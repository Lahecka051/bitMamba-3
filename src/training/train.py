"""Training loop for BitMamba-3 using upstream state-spaces/mamba MambaLMHeadModel.

Design:
  - Build upstream `MambaLMHeadModel` with `ssm_cfg={"layer": "Mamba3", ...}` so it
    instantiates Mamba-3 blocks from the vendored repo.
  - Post-init: swap each block's `in_proj` and `out_proj` with `BitLinear`.
  - AdamW + cosine LR + warmup, bf16 autocast, gradient checkpointing optional.
  - Logs to wandb if WANDB_API_KEY is set, else file-only.

This keeps upstream code intact; only Linear layers are replaced (BitMamba-2 style).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import BitLinear, ensure_mamba3_registered  # noqa: E402

ensure_mamba3_registered()  # Register Mamba3 layer in upstream create_block

from mamba_ssm.models.config_mamba import MambaConfig  # type: ignore  # noqa: E402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore  # noqa: E402

from training.dataset import TokenShardDataset  # type: ignore  # noqa: E402


# -------------- Model config presets --------------

PRESETS = {
    "30M": dict(
        d_model=384,
        n_layer=8,
        d_intermediate=0,
        vocab_size=50280,
        ssm_cfg=dict(layer="Mamba3", is_mimo=True, mimo_rank=4, d_state=64, headdim=64, chunk_size=8, rope_fraction=0.5),  # chunk_size=8 for Blackwell shmem budget
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    ),
    "130M": dict(
        d_model=768,
        n_layer=24,
        d_intermediate=0,
        vocab_size=50280,
        ssm_cfg=dict(layer="Mamba3", is_mimo=True, mimo_rank=4, d_state=128, headdim=64, chunk_size=8, rope_fraction=0.5),  # chunk_size=8 for Blackwell shmem budget
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    ),
    "370M": dict(
        d_model=1024,
        n_layer=48,
        d_intermediate=0,
        vocab_size=50280,
        ssm_cfg=dict(layer="Mamba3", is_mimo=True, mimo_rank=4, d_state=128, headdim=64, chunk_size=8, rope_fraction=0.5),  # chunk_size=8 for Blackwell shmem budget
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    ),
}


def bitify_model(model: nn.Module) -> tuple[int, int]:
    """Replace all Mamba3 `in_proj` / `out_proj` with `BitLinear` in place.
    Returns (n_layers_modified, total_ternary_params).
    """
    n_modified = 0
    n_ternary_params = 0
    for m in model.modules():
        for attr in ("in_proj", "out_proj"):
            lin = getattr(m, attr, None)
            if isinstance(lin, nn.Linear) and not isinstance(lin, BitLinear):
                new = BitLinear(
                    lin.in_features,
                    lin.out_features,
                    bias=lin.bias is not None,
                    device=lin.weight.device,
                    dtype=lin.weight.dtype,
                )
                setattr(m, attr, new)
                n_modified += 1
                n_ternary_params += new.weight.numel()
    return n_modified, n_ternary_params


def build_model(preset: str, device: str, dtype: torch.dtype) -> MambaLMHeadModel:
    cfg = MambaConfig(**PRESETS[preset])
    model = MambaLMHeadModel(cfg, device=device, dtype=dtype)
    n_replaced, n_tern = bitify_model(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"[build] preset={preset} total_params={total/1e6:.1f}M, "
          f"replaced_linears={n_replaced}, ternary_params={n_tern/1e6:.1f}M "
          f"({100*n_tern/total:.1f}%)")
    return model


# -------------- LR schedule --------------

def cosine_lr(step, max_steps, base_lr, warmup_steps, min_lr=1e-5):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


# -------------- Training loop --------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="30M")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--base_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--wandb_project", default="bitmamba3")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # wandb
    try:
        import wandb
        if os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc")):
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            use_wandb = True
        else:
            print("[wandb] no API key; file-only logging")
            use_wandb = False
    except ImportError:
        use_wandb = False

    # Data
    ds = TokenShardDataset(args.data_dir, seqlen=args.seqlen, samples_per_epoch=args.max_steps * args.batch_size * args.grad_accum + 1000)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    # Model
    model = build_model(args.preset, device, dtype)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable = True  # user hint
    model.train()

    # Optimizer
    no_decay = [n for n, p in model.named_parameters() if n.endswith(".bias") or "norm" in n or "_bias" in n or getattr(p, "_no_weight_decay", False)]
    params = [
        {"params": [p for n, p in model.named_parameters() if n not in no_decay], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.base_lr, betas=(0.9, 0.95))

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"[resume] from step {step}")

    t_start = time.time()
    loss_accum = 0.0
    micro_step = 0
    opt.zero_grad(set_to_none=True)

    for input_ids, labels in loader:
        if step >= args.max_steps:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids).logits  # (B, L, V)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)), labels.reshape(-1))
            loss = loss / args.grad_accum

        loss.backward()
        loss_accum += loss.item()
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = cosine_lr(step, args.max_steps, args.base_lr, args.warmup_steps, args.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                tok_s = args.batch_size * args.grad_accum * args.seqlen * (step + 1) / (time.time() - t_start)
                print(f"step={step:>6}  loss={loss_accum:.4f}  lr={lr:.2e}  tok/s={tok_s:.0f}")
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": loss_accum, "train/lr": lr, "train/tok_per_s": tok_s, "step": step})

            if step % args.save_interval == 0 and step > 0:
                path = out_dir / f"ckpt_{step:06d}.pt"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, path)
                print(f"[save] {path}")

            step += 1
            loss_accum = 0.0

    # Final save
    path = out_dir / f"ckpt_final_{step:06d}.pt"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)}, path)
    print(f"[save] final {path}")


if __name__ == "__main__":
    main()
