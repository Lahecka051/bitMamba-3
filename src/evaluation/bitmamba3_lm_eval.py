"""lm-evaluation-harness wrapper for BitMamba-3 local checkpoints.

Upstream `lm_eval.models.mamba_lm.MambaLMWrapper` loads Mamba checkpoints from
HuggingFace via `MambaLMHeadModel.from_pretrained`. Since our BitMamba-3
checkpoints are local .pt files with Mamba3 blocks + BitLinear swap, we subclass
and override `_create_model` to build from our preset config and state_dict.

Usage:
    lm_eval --model bitmamba3 --model_args \
        "preset=30M,ckpt=/path/to/ckpt.pt,tokenizer=EleutherAI/gpt-neox-20b" \
        --tasks lambada_openai,hellaswag ...

Or programmatically via `evaluate_checkpoint()` below.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered  # noqa: E402
ensure_mamba3_registered()

from lm_eval.api.registry import register_model  # type: ignore  # noqa: E402
from lm_eval.models.mamba_lm import MambaLMWrapper  # type: ignore  # noqa: E402

from mamba_ssm.models.config_mamba import MambaConfig  # type: ignore  # noqa: E402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore  # noqa: E402

from training.train import PRESETS, bitify_model  # type: ignore  # noqa: E402


@register_model("bitmamba3")
class BitMamba3LMWrapper(MambaLMWrapper):
    """Local-checkpoint wrapper for BitMamba-3.

    model_args expected keys:
        preset: One of "30M", "130M", "370M"
        ckpt: Path to local checkpoint .pt (saved by src/training/train.py)
        tokenizer: HF tokenizer id (default EleutherAI/gpt-neox-20b)
    """

    def __init__(
        self,
        preset: str = "30M",
        ckpt: Optional[str] = None,
        tokenizer: str = "EleutherAI/gpt-neox-20b",
        max_length: int = 2048,
        batch_size: int = 1,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs,
    ):
        self._preset = preset
        self._ckpt_path = ckpt
        self._dtype_str = dtype
        # Sidestep upstream HF loader; call grandparent HFLM with manual model injection
        # by monkey-patching _create_model. We do this via init overrides.
        self.is_hf = False
        # Ensure HFLM init runs after we've prepped our model
        super_kwargs = dict(kwargs)
        super_kwargs.setdefault("backend", "causal")
        super_kwargs.setdefault("max_length", max_length)
        super_kwargs.setdefault("batch_size", batch_size)
        super_kwargs.setdefault("device", device)
        super_kwargs.setdefault("dtype", dtype)
        # MambaLMWrapper uses `pretrained` for HF path; we use a dummy and override _create_model.
        super_kwargs["pretrained"] = f"bitmamba3-{preset}-local"
        super_kwargs["tokenizer"] = tokenizer
        super().__init__(**super_kwargs)

    def _get_config(self, pretrained: str, **kwargs) -> None:
        # Synthesize a config from our preset; upstream code only reads a few fields.
        preset_cfg = PRESETS[self._preset]
        self._config = MambaConfig(**preset_cfg)

    def _create_model(
        self,
        pretrained: str,
        dtype: str | torch.dtype | None = "bfloat16",
        **kwargs,
    ) -> None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        cfg = MambaConfig(**PRESETS[self._preset])
        self._model = MambaLMHeadModel(cfg, device=self.device, dtype=dtype)
        bitify_model(self._model)

        if self._ckpt_path:
            state = torch.load(self._ckpt_path, map_location=self.device)
            if "model" in state:
                state = state["model"]
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            if missing:
                print(f"[bitmamba3] missing keys: {len(missing)} (first 3: {missing[:3]})")
            if unexpected:
                print(f"[bitmamba3] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

        self._model.eval()
