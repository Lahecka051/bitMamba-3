"""Parity task: state-tracking benchmark from Mamba-3 paper.

Given a sequence of bits (0/1 tokens), predict whether the running XOR up to
each position is 0 or 1. Requires real state tracking (cannot be solved by
real-valued linear SSMs per Mamba-3 paper).

Mamba-3 paper result: Mamba-2 ~0.90% acc (random), Mamba-3 100% at small scale.

This benchmark uses a synthetic mini-model trained from scratch on pure parity
sequences. Not a standard lm-eval task; run standalone.

Usage:
    python src/evaluation/parity_task.py --arch mamba3_mimo --n_steps 5000 --seqlen 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import BitLinear, ensure_mamba3_registered  # noqa: E402
ensure_mamba3_registered()

from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore  # noqa: E402
from mamba_ssm.modules.mamba3 import Mamba3  # type: ignore  # noqa: E402


def make_parity_batch(batch, seqlen, device):
    """Generate random bit sequences and their running XOR labels."""
    bits = torch.randint(0, 2, (batch, seqlen), device=device, dtype=torch.long)
    labels = torch.zeros_like(bits)
    running = torch.zeros((batch,), device=device, dtype=torch.long)
    for t in range(seqlen):
        running = running ^ bits[:, t]
        labels[:, t] = running
    return bits, labels


class ParityModel(nn.Module):
    """Tiny 2-token embedding + single SSM block + 2-class head."""

    def __init__(self, arch="mamba3_mimo", d_model=128, bitize=False, depth=1, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed = nn.Embedding(2, d_model, **factory_kwargs)
        self.blocks = nn.ModuleList()
        for li in range(depth):
            if arch == "mamba3_mimo":
                blk = Mamba3(d_model=d_model, d_state=64, expand=2, headdim=32, is_mimo=True, mimo_rank=4, chunk_size=8, rope_fraction=0.5, layer_idx=li, **factory_kwargs)
            elif arch == "mamba3_siso":
                blk = Mamba3(d_model=d_model, d_state=64, expand=2, headdim=32, is_mimo=False, chunk_size=64, rope_fraction=0.5, layer_idx=li, **factory_kwargs)
            elif arch == "mamba2":
                blk = Mamba2(d_model=d_model, d_state=64, expand=2, headdim=32, chunk_size=64, layer_idx=li, **factory_kwargs)
            else:
                raise ValueError(f"Unknown arch: {arch}")
            self.blocks.append(blk)
        # Backward-compat alias for older code paths that reference self.ssm
        self.ssm = self.blocks[0]
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        self.head = nn.Linear(d_model, 2, **factory_kwargs)

        if bitize:
            # Replace Linear layers inside each SSM block with BitLinear
            for blk in self.blocks:
                for m in blk.modules():
                    for attr in ("in_proj", "out_proj"):
                        lin = getattr(m, attr, None)
                        if isinstance(lin, nn.Linear) and not isinstance(lin, BitLinear):
                            new = BitLinear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                            device=lin.weight.device, dtype=lin.weight.dtype)
                            setattr(m, attr, new)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = h + blk(h)  # residual
        h = self.norm(h)
        return self.head(h)  # (B, L, 2)


@torch.no_grad()
def eval_acc(model, batch=64, seqlen=256, device="cuda"):
    bits, labels = make_parity_batch(batch, seqlen, device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(bits)
    pred = logits.argmax(dim=-1)
    acc_total = (pred == labels).float().mean().item()
    acc_last = (pred[:, -1] == labels[:, -1]).float().mean().item()
    return acc_total, acc_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mamba3_mimo", choices=["mamba3_mimo", "mamba3_siso", "mamba2"])
    ap.add_argument("--bitize", action="store_true", help="Apply BitLinear ternary wrap (BitMamba-3/BitMamba-2 style)")
    ap.add_argument("--n_steps", type=int, default=3000)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=1, help="Number of stacked SSM blocks")
    ap.add_argument("--out_json", default="results/tables/parity_task.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    np_seed = args.seed
    try:
        import numpy as np
        np.random.seed(np_seed)
    except Exception:
        pass

    device = "cuda"
    # Build directly at bf16 so that Mamba3's explicitly-fp32 params
    # (B_bias, C_bias, mimo_x/z/o, dt_bias, D) retain fp32 (matching kernel expectations).
    model = ParityModel(args.arch, d_model=args.d_model, bitize=args.bitize, depth=args.depth, device=device, dtype=torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tag = f"{args.arch}{'_bit' if args.bitize else ''}_seed{args.seed}_d{args.d_model}"
    if args.depth > 1:
        tag += f"_depth{args.depth}"
    print(f"=== Parity task: {tag} | d_model={args.d_model}, seqlen={args.seqlen} ===")

    history = []
    t0 = time.time()
    for step in range(args.n_steps):
        bits, labels = make_parity_batch(args.batch, args.seqlen, device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(bits)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_interval == 0 or step == args.n_steps - 1:
            model.eval()
            acc_tot, acc_last = eval_acc(model, batch=64, seqlen=args.seqlen, device=device)
            model.train()
            elapsed = time.time() - t0
            print(f"  step={step:>5}  loss={loss.item():.4f}  acc_all={acc_tot:.4f}  acc_last_tok={acc_last:.4f}  ({elapsed:.1f}s)")
            history.append({"step": step, "loss": loss.item(), "acc_all": acc_tot, "acc_last": acc_last})

    # Final evaluation on longer seqlen if possible
    print("\nFinal eval at 2x train seqlen:")
    model.eval()
    try:
        acc_tot, acc_last = eval_acc(model, batch=64, seqlen=args.seqlen * 2, device=device)
        print(f"  seqlen={args.seqlen*2}: acc_all={acc_tot:.4f} acc_last={acc_last:.4f}")
        history.append({"step": "final_2x", "seqlen": args.seqlen * 2, "acc_all": acc_tot, "acc_last": acc_last})
    except Exception as e:
        print(f"  2x eval failed: {e}")

    out_path = _root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing results if present
    try:
        existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    except Exception:
        existing = {}
    existing[tag] = {"args": vars(args), "history": history}
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nWrote {tag} results to {out_path}")


if __name__ == "__main__":
    main()
