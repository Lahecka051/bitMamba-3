"""INT4 weight-only PTQ Linear layer.

Used as a fair-baseline comparison vs BitLinear (ternary). Both replace
the full-precision Linear with a quantized variant; INT4 is the de-facto
deployment standard (GPTQ, AWQ-class methods all target ~3-4 bits).

This implementation is the simplest possible INT4 PTQ:
  - Per-tensor symmetric range [-8, 7]
  - absmax scale: scale = max(|W|) / 7
  - Round-to-nearest, no Hessian or activation reweighting
  - Weight-only (activations stay BF16)
  - Applied post-hoc to a trained FP checkpoint (no fine-tuning)

This is intentionally a simple baseline. More sophisticated INT4 methods
(GPTQ, AWQ, NF4) typically give 1-3% better PPL but require calibration
data and are framework-specific. Round-to-nearest is the universal floor.

Usage: load an FP model, replace its in_proj/out_proj with Int4Linear,
copy the FP weights into the new layer (weights are quantized on first
forward via the round-trip).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int4Linear(nn.Linear):
    """INT4 PTQ Linear: per-tensor symmetric weight quantization.

    Drop-in replacement for nn.Linear. Like BitLinear, the weights remain
    stored in FP (so we can reload from any FP checkpoint), and on every
    forward we round-trip the weights through the INT4 grid before the
    matmul. Activations are NOT quantized (matches BitsAndBytes 4-bit and
    most weight-only PTQ methods).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        absmax = self.weight.abs().amax().clamp(min=1e-5)
        scale = absmax / 7.0  # INT4 symmetric: [-8, 7], use 7 for symmetric mid
        w_q = (self.weight / scale).round().clamp(-8, 7) * scale
        # No STE needed (PTQ, no training)
        return F.linear(w_q, x.transpose(-1, -2)).transpose(-1, -2) if False else F.linear(x, w_q, self.bias)


def int4_quantize_model(model: nn.Module) -> tuple[int, int]:
    """Replace all in_proj / out_proj nn.Linear with Int4Linear in place.
    Mirrors bitify_model from training.train but for INT4."""
    n_modified = 0
    n_params = 0
    for m in model.modules():
        for attr in ("in_proj", "out_proj"):
            lin = getattr(m, attr, None)
            if isinstance(lin, nn.Linear) and not isinstance(lin, Int4Linear):
                # Reuse existing tensor (just change forward semantics)
                new = Int4Linear(
                    lin.in_features, lin.out_features,
                    bias=lin.bias is not None,
                    device=lin.weight.device, dtype=lin.weight.dtype,
                )
                with torch.no_grad():
                    new.weight.copy_(lin.weight)
                    if lin.bias is not None:
                        new.bias.copy_(lin.bias)
                setattr(m, attr, new)
                n_modified += 1
                n_params += new.weight.numel()
    return n_modified, n_params
