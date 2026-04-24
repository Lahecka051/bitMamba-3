"""Sanity check: BitMamba3 MIMO forward pass.

MIMO is the key Mamba-3 novelty (+1.2pt accuracy per paper).
Verifies:
  1. BitMamba3 constructs with is_mimo=True
  2. mimo_x / mimo_z / mimo_o parameters exist
  3. Forward pass on GPU produces expected shape
  4. Inspect which params are ternarized vs FP
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

import torch

from bitmamba3 import BitLinear, BitMamba3


def summarize_params(block, label):
    print(f"\n[{label}] Parameter breakdown:")
    total = 0
    ternary = 0
    fp = 0
    details = []
    for name, p in block.named_parameters():
        n = p.numel()
        total += n
        is_ternary = (
            name.startswith("in_proj.") or name.startswith("out_proj.")
        )
        if is_ternary:
            ternary += n
        else:
            fp += n
        details.append((name, tuple(p.shape), n, "TERN" if is_ternary else "FP  "))

    for name, shape, n, tag in sorted(details, key=lambda t: -t[2])[:15]:
        print(f"  {tag}  {name:40s}  shape={str(shape):25s}  params={n:>10,}")

    print(f"  TOTAL     {total:>12,}")
    print(f"  TERNARY   {ternary:>12,}  ({100*ternary/total:.1f}%)")
    print(f"  FP        {fp:>12,}  ({100*fp/total:.1f}%)")


def test_mimo(mimo_rank=4):
    print(f"\n{'='*60}\nBitMamba3 MIMO (rank={mimo_rank}) sanity\n{'='*60}")
    torch.manual_seed(0)
    block = BitMamba3(
        d_model=256,
        d_state=64,
        expand=2,
        headdim=64,
        is_mimo=True,
        mimo_rank=mimo_rank,
        chunk_size=64 // mimo_rank,
        device="cuda",
        dtype=torch.bfloat16,
    ).cuda()

    # Check MIMO-specific params exist
    assert hasattr(block, "mimo_x"), "MIMO mode missing mimo_x"
    assert hasattr(block, "mimo_z"), "MIMO mode missing mimo_z"
    assert hasattr(block, "mimo_o"), "MIMO mode missing mimo_o"
    assert isinstance(block.in_proj, BitLinear), "in_proj not BitLinear"
    assert isinstance(block.out_proj, BitLinear), "out_proj not BitLinear"

    print(f"in_proj.out_features = {block.in_proj.out_features}")
    print(f"mimo_x.shape = {tuple(block.mimo_x.shape)}")
    print(f"mimo_z.shape = {tuple(block.mimo_z.shape)}")
    print(f"mimo_o.shape = {tuple(block.mimo_o.shape)}")

    summarize_params(block, f"MIMO rank={mimo_rank}")

    # Forward
    x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            y = block(x)
        print(f"\n[MIMO forward] OK, output shape={tuple(y.shape)}")
        assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    except Exception as e:
        print(f"\n[MIMO forward] FAILED: {type(e).__name__}: {e}")
        raise


def test_siso_for_comparison():
    print(f"\n{'='*60}\nBitMamba3 SISO (for comparison)\n{'='*60}")
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
    summarize_params(block, "SISO")

    x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y = block(x)
    print(f"\n[SISO forward] OK, output shape={tuple(y.shape)}")


if __name__ == "__main__":
    test_siso_for_comparison()
    test_mimo(mimo_rank=4)
    test_mimo(mimo_rank=2)
    print("\n\nAll MIMO sanity tests PASSED")
