"""BitMamba-3 block: Mamba-3 with BitLinear projections.

Subclasses the upstream Mamba-3 module from state-spaces/mamba v2.3.1 and
swaps `in_proj` and `out_proj` for BitLinear (1.58-bit ternary weight,
INT8 activation). SSM kernels, RoPE, RMSNorm, and biases remain FP.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing upstream Mamba-3 module from vendored third_party
_repo_root = Path(__file__).resolve().parents[2]
_mamba_src = _repo_root / "third_party" / "state-spaces-mamba"
if str(_mamba_src) not in sys.path:
    sys.path.insert(0, str(_mamba_src))

from mamba_ssm.modules.mamba3 import Mamba3  # type: ignore  # noqa: E402

from .bitlinear import BitLinear  # noqa: E402


class BitMamba3(Mamba3):
    """Mamba-3 block with ternary in_proj and out_proj weights."""

    def __init__(self, *args, ternarize_mimo: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype

        self.in_proj = BitLinear(
            self.in_proj.in_features,
            self.in_proj.out_features,
            bias=self.in_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        self.out_proj = BitLinear(
            self.out_proj.in_features,
            self.out_proj.out_features,
            bias=self.out_proj.bias is not None,
            device=device,
            dtype=dtype,
        )

        self.ternarize_mimo = ternarize_mimo and self.is_mimo
        # Optional: ternarize MIMO projection tensors via ablation flag.
        # Deferred — requires custom quantization of einsum weights.
