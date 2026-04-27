"""D — d=512 / depth=4 parity sweep, 5 seeds, 5K steps cosine LR.

Hypothesis: at this even larger scale, the ternary inductive-bias effect
should remain (or strengthen). If FP also reaches 0.95 here, the d=256
finding was scale-specific. If FP stays at chance and ternary stays at 0.95,
the inductive bias is robust to model size.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered  # noqa: E402
ensure_mamba3_registered()

from scripts.run_parity_larger_bg import run_one  # type: ignore  # noqa: E402


def main():
    out_path = _root / "results/tables/parity_d512.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("mamba3_siso", False),
        ("mamba3_siso", True),
        ("mamba3_mimo", False),
        ("mamba3_mimo", True),
    ]
    seeds = [0, 1, 2, 3, 4]

    results = []
    for arch, bitize in configs:
        for seed in seeds:
            summary = run_one(arch, bitize, seed,
                              n_steps=5000, d_model=512, seqlen=128,
                              batch=32, base_lr=1e-3, depth=4)
            results.append(summary)
            out_path.write_text(json.dumps(results, indent=2))

    print(f"\nAll {len(results)} d=512 runs complete -> {out_path}")


if __name__ == "__main__":
    main()
