"""Test MIMO backward across rank/chunk_size configs to find one that works on RTX 5090.

The TileLang backward kernel needs dynamic shared memory, and at default
configs it exceeds Blackwell (SM 12.0) limits. Try smaller configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

import torch
import torch.nn.functional as F

from bitmamba3 import BitMamba3


def try_config(d_model, d_state, headdim, mimo_rank, chunk_size, batch=2, seqlen=128):
    label = f"d_model={d_model} d_state={d_state} headdim={headdim} rank={mimo_rank} chunk={chunk_size}"
    print(f"\n[trying] {label}")
    try:
        block = BitMamba3(
            d_model=d_model,
            d_state=d_state,
            expand=2,
            headdim=headdim,
            is_mimo=True,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            device="cuda",
            dtype=torch.bfloat16,
        ).cuda()
        x = torch.randn(batch, seqlen, d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = block(x)
        loss = (y ** 2).mean()
        loss.backward()
        print(f"  OK  (output shape={tuple(y.shape)}, loss={loss.item():.4f})")
        return True
    except Exception as e:
        msg = str(e)[:160]
        print(f"  FAIL  {type(e).__name__}: {msg}")
        return False


if __name__ == "__main__":
    results = []
    # Start with the 30M preset defaults
    configs = [
        dict(d_model=384, d_state=64, headdim=64, mimo_rank=4, chunk_size=16),
        dict(d_model=384, d_state=64, headdim=64, mimo_rank=2, chunk_size=32),
        dict(d_model=384, d_state=64, headdim=64, mimo_rank=2, chunk_size=16),
        dict(d_model=384, d_state=32, headdim=32, mimo_rank=2, chunk_size=16),
        dict(d_model=256, d_state=32, headdim=32, mimo_rank=2, chunk_size=16),
        dict(d_model=384, d_state=64, headdim=64, mimo_rank=4, chunk_size=8),
        dict(d_model=384, d_state=64, headdim=64, mimo_rank=4, chunk_size=32),
    ]
    for c in configs:
        ok = try_config(**c)
        results.append((c, ok))

    print("\n=== Summary ===")
    for c, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {c}")
