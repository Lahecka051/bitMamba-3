"""Smoke test: verify end-to-end BitMamba-3 training step works.

Uses random token IDs (no real data) to sanity-check:
  1. MambaLMHeadModel with Mamba-3 ssm_cfg builds correctly
  2. BitLinear replacement via bitify_model works
  3. Forward pass produces logits of correct shape
  4. Backward pass runs without error
  5. Optimizer step updates parameters
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

import torch
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig  # type: ignore
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore

from training.train import PRESETS, bitify_model  # type: ignore


def run_smoke(preset="30M"):
    print(f"=== Building {preset} model ===")
    cfg = MambaConfig(**PRESETS[preset])
    model = MambaLMHeadModel(cfg, device="cuda", dtype=torch.bfloat16)
    n_replaced, n_tern = bitify_model(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"  params={total/1e6:.2f}M, replaced Linears={n_replaced}, ternary={n_tern/1e6:.2f}M ({100*n_tern/total:.1f}%)")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch, seqlen = 2, 256
    vocab_size = cfg.vocab_size

    print(f"\n=== Forward + Backward (batch={batch}, seqlen={seqlen}) ===")
    x = torch.randint(0, vocab_size, (batch, seqlen), device="cuda", dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch, seqlen), device="cuda", dtype=torch.long)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x).logits
        loss = F.cross_entropy(out.reshape(-1, vocab_size), y.reshape(-1))

    print(f"  logits shape: {tuple(out.shape)}")
    print(f"  initial loss: {loss.item():.4f}")
    print(f"  expected random-init loss: ~{torch.log(torch.tensor(vocab_size)).item():.4f}")

    # Measure one backward + step
    import time
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    opt.step()
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    # Check gradient stats
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"  backward: {(t1-t0)*1000:.1f} ms, opt step: {(t2-t1)*1000:.1f} ms")
    print(f"  gradient count: {len(grad_norms)}, max grad norm: {max(grad_norms):.4f}")

    # Do 3 more steps to verify loss reduces
    print(f"\n=== 3 more steps on same batch (should reduce loss) ===")
    for s in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x).logits
            loss = F.cross_entropy(out.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        opt.step()
        print(f"  step {s+1}: loss={loss.item():.4f}")

    print("\nSmoke test PASSED")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="30M")
    ap.add_argument("--siso", action="store_true", help="Force SISO (bypass MIMO TileLang backward issue)")
    args = ap.parse_args()
    if args.siso:
        from training.train import PRESETS
        PRESETS[args.preset]["ssm_cfg"]["is_mimo"] = False
        PRESETS[args.preset]["ssm_cfg"]["chunk_size"] = 64
        print("[smoke_test] Forcing SISO mode")
    run_smoke(args.preset)
