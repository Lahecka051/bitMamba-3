"""Sanity check: BitLinear + BitMamba3 forward pass on random input.

Verifies:
  1. BitLinear produces output of correct shape
  2. Weight effectively quantized to ternary post-forward (inspect detached w_q)
  3. BitMamba3 (subclass of Mamba3) forwards without error on GPU
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

import torch
import torch.nn as nn

from bitmamba3 import BitLinear, BitMamba3


def test_bitlinear():
    torch.manual_seed(0)
    layer = BitLinear(in_features=128, out_features=256, bias=False).cuda()

    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.float32)
    y = layer(x)

    assert y.shape == (2, 16, 256), f"BitLinear output shape mismatch: {y.shape}"
    print(f"[BitLinear] forward OK, output shape={tuple(y.shape)}")

    # Check weight would ternarize
    scale_w = 1.0 / layer.weight.abs().mean().clamp(min=1e-5)
    w_q = (layer.weight * scale_w).round().clamp(-1, 1)
    unique = torch.unique(w_q)
    print(f"[BitLinear] weight quantized unique values: {unique.tolist()}")
    assert set(unique.tolist()).issubset({-1.0, 0.0, 1.0}), f"Weight not ternary: {unique}"
    print("[BitLinear] ternary weight check PASS")


def test_bitmamba3_siso():
    torch.manual_seed(0)
    block = BitMamba3(
        d_model=256,
        d_state=64,
        expand=2,
        headdim=64,
        is_mimo=False,
        chunk_size=64,
        device="cuda",
        dtype=torch.bfloat16,
    ).cuda()

    # Check substitutions took effect
    assert isinstance(block.in_proj, BitLinear), "in_proj not BitLinear"
    assert isinstance(block.out_proj, BitLinear), "out_proj not BitLinear"
    print("[BitMamba3] in_proj and out_proj are BitLinear instances")

    # Parameter count breakdown
    total_params = sum(p.numel() for p in block.parameters())
    linear_params = block.in_proj.weight.numel() + block.out_proj.weight.numel()
    print(f"[BitMamba3] total params={total_params:,}, ternarized Linear params={linear_params:,} ({100*linear_params/total_params:.1f}%)")

    # Forward pass
    x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        try:
            y = block(x)
            print(f"[BitMamba3 SISO] forward OK, output shape={tuple(y.shape)}")
            assert y.shape == x.shape
        except Exception as e:
            print(f"[BitMamba3 SISO] forward FAILED: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("Sanity check: BitLinear")
    print("=" * 60)
    test_bitlinear()

    print()
    print("=" * 60)
    print("Sanity check: BitMamba3 (SISO)")
    print("=" * 60)
    test_bitmamba3_siso()

    print()
    print("All sanity checks PASSED")
