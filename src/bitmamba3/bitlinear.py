"""BitNet-style ternary Linear with straight-through estimator.

Reference: BitMamba-2 (Zhayr1/BitMamba-2), BitNet b1.58 (arXiv:2402.17764).

Weight: ternary {-1, 0, +1} per-tensor absmean scaling.
Activation: INT8 per-token absmax scaling, with pre-RMSNorm.
Gradient: straight-through estimator (STE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with BitNet b1.58 quantization.

    Same constructor signature as nn.Linear. During forward:
      1. RMSNorm activation
      2. Per-token absmax INT8 quantize activation with STE
      3. Per-tensor absmean ternary quantize weight with STE
      4. Fake-quant F.linear (effective matmul is FP; HW backend would
         use true ternary/INT8 kernel)

    Notes:
      - `bias` is kept in FP and is not quantized.
      - RMSNorm has no learnable affine (unit gamma) to keep ternarization
        strategy simple; the weight scale absorbs magnitude information.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.rms_norm(x, normalized_shape=(x.size(-1),))

        abs_max = x_n.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale_x = 127.0 / abs_max
        x_q = (x_n * scale_x).round().clamp(-128, 127) / scale_x
        x_eff = x_n + (x_q - x_n).detach()

        abs_mean = self.weight.abs().mean().clamp(min=1e-5)
        scale_w = 1.0 / abs_mean
        w_q = (self.weight * scale_w).round().clamp(-1, 1) / scale_w
        w_eff = self.weight + (w_q - self.weight).detach()

        return F.linear(x_eff, w_eff, self.bias)
