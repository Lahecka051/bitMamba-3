"""Runtime patch: register 'Mamba3' as a valid ssm_layer in state-spaces/mamba.

Upstream v2.3.1 ships `mamba3.py` module but `mixer_seq_simple.create_block`
still rejects `ssm_cfg["layer"] == "Mamba3"`. This module injects Mamba3 into
the allow-list at import time so that `MambaLMHeadModel` can instantiate
Mamba-3 blocks without modifying upstream source code.

Usage (import BEFORE building MambaLMHeadModel):
    from bitmamba3.mamba3_lm_patch import ensure_mamba3_registered
    ensure_mamba3_registered()
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel(MambaConfig(ssm_cfg={"layer": "Mamba3", ...}), ...)

Paper note: this is a registration stub, not an algorithmic change. The Mamba3
module class itself is used unmodified from state-spaces/mamba.
"""

from __future__ import annotations

import copy
from functools import partial

_PATCHED = False


def ensure_mamba3_registered():
    global _PATCHED
    if _PATCHED:
        return

    import mamba_ssm.models.mixer_seq_simple as _m
    from mamba_ssm.modules.mamba3 import Mamba3
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mha import MHA
    from mamba_ssm.modules.mlp import GatedMLP
    from mamba_ssm.modules.block import Block
    import torch.nn as nn

    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except ImportError:
        RMSNorm = None

    _SSM_CLASSES = {"Mamba1": Mamba, "Mamba2": Mamba2, "Mamba3": Mamba3}

    def create_block(
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
            ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
            ssm_layer = ssm_cfg.pop("layer", "Mamba1")
            if ssm_layer not in _SSM_CLASSES:
                raise ValueError(
                    f"Invalid ssm_layer: {ssm_layer}, supported: {list(_SSM_CLASSES.keys())}"
                )
            mixer_cls = partial(
                _SSM_CLASSES[ssm_layer],
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs,
            )
        else:
            mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm,
            eps=norm_epsilon,
            **factory_kwargs,
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

    _m.create_block = create_block
    _PATCHED = True
