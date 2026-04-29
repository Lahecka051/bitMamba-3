"""
Train Mamba-3 180M using OFFICIAL mamba_ssm infrastructure.
Uses MambaLMHeadModel + Block + create_block pattern.
Only modification: add Mamba3 support to create_block.

Key differences from our broken Mamba3LM:
  1. Official Block: proper residual management (Add→LN→Mixer)
  2. _init_weights: out_proj scaled by 1/√(2*n_layer)
  3. fused_add_norm: fused Add+LayerNorm
  4. residual_in_fp32: residual path stays FP32
  5. tie_embeddings: weight tying via config
"""
import os, sys, json, math, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from functools import partial
from collections import namedtuple
from dataclasses import dataclass, field

DEVICE = 'cuda'
WORKDIR = Path(__file__).resolve().parent.parent

# Import official components
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ============================================================
# Config (matching paper's 180M)
# ============================================================
@dataclass
class Mamba3Config:
    d_model: int = 768
    d_intermediate: int = 0  # No MLP (SSM has internal gating)
    n_layer: int = 24
    vocab_size: int = 50257  # GPT-2 tokenizer
    ssm_cfg: dict = field(default_factory=lambda: {
        "layer": "Mamba3",
        "d_state": 128,
        "headdim": 64,
    })
    attn_layer_idx: list = field(default_factory=lambda: [5, 11, 17, 23])  # 5:1 ratio
    attn_cfg: dict = field(default_factory=lambda: {
        "num_heads": 12,
        "causal": True,
    })
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


# ============================================================
# Modified create_block to support Mamba3
# ============================================================
def create_block_mamba3(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg)
        ssm_layer = ssm_cfg.pop("layer", "Mamba3")

        if ssm_layer == "Mamba3":
            mixer_cls = partial(
                Mamba3,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
        elif ssm_layer == "Mamba2":
            from mamba_ssm.modules.mamba2 import Mamba2
            mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        else:
            from mamba_ssm.modules.mamba_simple import Mamba
            mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# ============================================================
# Official _init_weights (from mixer_seq_simple.py)
# ============================================================
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ============================================================
# Mamba3 LM Head Model (following official pattern)
# ============================================================
class Mamba3LMHeadModel(nn.Module, GenerationMixin):
    def __init__(self, config: Mamba3Config, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config

        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (vocab_size % config.pad_vocab_size_multiple)

        self.backbone = Mamba3Backbone(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            residual_in_fp32=config.residual_in_fp32,
            fused_add_norm=config.fused_add_norm,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False, **factory_kwargs)

        # Apply official init
        self.apply(
            partial(_init_weights, n_layer=config.n_layer,
                    n_residuals_per_layer=1 if config.d_intermediate == 0 else 2)
        )
        self.tie_weights()

        # Count params
        n_params = sum(p.numel() for p in self.parameters())
        n_ssm = sum(1 for i in range(config.n_layer) if i not in config.attn_layer_idx)
        n_attn = len(config.attn_layer_idx)
        print(f"Mamba3LMHeadModel: {n_params/1e6:.1f}M params, "
              f"{n_ssm}×SSM + {n_attn}×Attention ({n_ssm}:{n_attn})")

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids, targets=None, inference_params=None):
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


class Mamba3Backbone(nn.Module):
    def __init__(self, d_model, n_layer, d_intermediate, vocab_size,
                 ssm_cfg, attn_layer_idx, attn_cfg,
                 rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                 device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.layers = nn.ModuleList([
            create_block_mamba3(
                d_model,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                norm_epsilon=1e-5,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=1e-5, **factory_kwargs
        )

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


# ============================================================
# Data Loading (same as before)
# ============================================================
class FineWebEduDataset:
    def __init__(self, tokenizer, seq_len=2048):
        from datasets import load_dataset
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []
        self.iter = iter(self.dataset)

    def get_batch(self, batch_size):
        while len(self.buffer) < batch_size * (self.seq_len + 1):
            try:
                sample = next(self.iter)
                tokens = self.tokenizer.encode(sample["text"])
                self.buffer.extend(tokens)
            except StopIteration:
                self.iter = iter(self.dataset)
                sample = next(self.iter)
                tokens = self.tokenizer.encode(sample["text"])
                self.buffer.extend(tokens)

        input_ids = []
        targets = []
        for _ in range(batch_size):
            chunk = self.buffer[:self.seq_len + 1]
            self.buffer = self.buffer[self.seq_len + 1:]
            input_ids.append(torch.tensor(chunk[:self.seq_len], dtype=torch.long))
            targets.append(torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long))

        return torch.stack(input_ids), torch.stack(targets)


# ============================================================
# Validation
# ============================================================
def validate_model(model, tokenizer, tag=""):
    model.eval()
    print(f"\n  {'='*50}")
    print(f"  Validation Checkpoint {tag}")
    print(f"  {'='*50}")

    texts = [
        "The solar system consists of the Sun and eight planets orbiting around it.",
        "Machine learning algorithms identify patterns in data without explicit programming.",
        "The French Revolution began in 1789 and changed the course of modern history.",
        "Quantum computing uses superposition and entanglement to process information.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    ]
    total_loss = 0
    total_tokens = 0
    for text in texts:
        ids = tokenizer.encode(text)
        x = torch.tensor([ids[:-1]]).to(DEVICE)
        y = torch.tensor([ids[1:]]).to(DEVICE)
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * len(ids)
        total_tokens += len(ids)
    unseen_ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))

    import random
    rand_ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(128)]
    x = torch.tensor([rand_ids[:-1]]).to(DEVICE)
    y = torch.tensor([rand_ids[1:]]).to(DEVICE)
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    random_ppl = math.exp(min(loss.item(), 20))

    x = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]]).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
    copy_matches = sum(1 for i in range(7) if preds[0, i].item() == x[0, i + 1].item())

    print(f"  Unseen PPL:  {unseen_ppl:.2f}  (expect 20-80)")
    print(f"  Random PPL:  {random_ppl:.2f}  (expect >1000)")
    print(f"  Copy ratio:  {copy_matches}/7  (expect 0-1)")

    status = "PASS" if unseen_ppl > 10 and random_ppl > 100 and copy_matches <= 2 else "FAIL"
    print(f"  Status: {status}")
    print(f"  {'='*50}\n")
    model.train()
    return {"unseen_ppl": unseen_ppl, "random_ppl": random_ppl, "copy_ratio": copy_matches, "status": status}


# ============================================================
# Training
# ============================================================
def train(total_tokens=100_000_000_000, stop_at_tokens=None,
          batch_size=4, grad_accum=4, resume_from=None):

    config = Mamba3Config()
    seq_len = 2048
    effective_batch = batch_size * grad_accum
    tokens_per_step = effective_batch * seq_len

    print(f"{'='*60}")
    print(f"Training Mamba-3 180M (Official Architecture)")
    print(f"{'='*60}")
    print(f"  Effective batch: {batch_size} × {grad_accum} = {effective_batch}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {total_tokens/1e9:.0f}B (LR schedule)")
    if stop_at_tokens:
        print(f"  Stop at: {stop_at_tokens/1e9:.0f}B")
    print()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = len(tokenizer)

    model = Mamba3LMHeadModel(config).to(DEVICE)
    dataset = FineWebEduDataset(tokenizer, seq_len)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1,
    )

    total_steps = total_tokens // tokens_per_step
    warmup_steps = min(2000, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler(device='cuda')

    start_step = 0
    total_tokens_seen = 0
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        total_tokens_seen = ckpt["tokens_seen"]
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resumed from step {start_step}, {total_tokens_seen/1e9:.2f}B tokens")

    ckpt_dir = WORKDIR / "ppq" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_interval = 100
    save_interval = 10000
    running_loss = 0
    t0 = time.time()
    actual_stop = stop_at_tokens if stop_at_tokens else total_tokens

    model.train()
    optimizer.zero_grad()
    step = start_step

    while total_tokens_seen < actual_stop:
        for micro_step in range(grad_accum):
            input_ids, targets = dataset.get_batch(batch_size)
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(input_ids, targets)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        step += 1
        total_tokens_seen += tokens_per_step

        if step % log_interval == 0:
            elapsed = time.time() - t0
            tokens_since = total_tokens_seen - (start_step * tokens_per_step if resume_from else 0)
            tps = tokens_since / max(elapsed, 1)
            loss_avg = running_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            ppl = math.exp(min(loss_avg * grad_accum, 20))
            progress = total_tokens_seen / total_tokens * 100

            print(f"  Step {step:>7d} | Loss {loss_avg*grad_accum:.4f} | PPL {ppl:.1f} | "
                  f"LR {lr:.2e} | {tps/1e3:.1f}K tok/s | "
                  f"{total_tokens_seen/1e9:.2f}B/{total_tokens/1e9:.0f}B ({progress:.1f}%)",
                  flush=True)
            running_loss = 0

        if step % save_interval == 0:
            ckpt_path = ckpt_dir / f"mamba3_official_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "tokens_seen": total_tokens_seen,
                "config": config.__dict__,
            }, str(ckpt_path))
            print(f"  Saved: {ckpt_path.name}")

    # Save at stop
    stop_path = ckpt_dir / f"mamba3_official_{total_tokens_seen/1e9:.0f}B.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": total_tokens_seen,
        "config": config.__dict__,
    }, str(stop_path))

    elapsed = time.time() - t0
    print(f"\nTraining stopped at {total_tokens_seen/1e9:.1f}B tokens in {elapsed/3600:.1f}h")
    print(f"Saved: {stop_path}")

    val_results = validate_model(model, tokenizer, tag=f"{total_tokens_seen/1e9:.0f}B")
    return model, val_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=float, default=100e9)
    parser.add_argument("--stop", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    args = parser.parse_args()

    train(total_tokens=int(args.tokens),
          stop_at_tokens=int(args.stop) if args.stop else None,
          batch_size=args.batch, grad_accum=args.accum,
          resume_from=args.resume)
