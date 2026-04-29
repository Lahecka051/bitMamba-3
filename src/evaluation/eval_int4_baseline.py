"""Apply INT4 PTQ to a FP Mamba-3 checkpoint and evaluate on WikiText-103.

This is the fair-baseline counterpart to eval_fp_baseline.py. It loads the
Mamba-3 FP checkpoint, applies post-hoc INT4 weight quantization (per-tensor
symmetric absmax), and measures PPL on the same WikiText-103 slice.

Compared metrics (130M / 480M tokens):
  - Mamba-3 FP16     : PPL 61.86  (eval_fp_baseline.py)
  - Mamba-3 INT4 PTQ : PPL ???    (this script)
  - BitMamba-3 ternary: PPL 69.4   (quick_eval.py)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered
ensure_mamba3_registered()

from transformers import AutoTokenizer
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from training.train import PRESETS
from bitmamba3.int4_linear import int4_quantize_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="FP checkpoint to quantize post-hoc")
    ap.add_argument("--preset", default="130M")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda"

    print(f"Loading FP {args.ckpt} (preset={args.preset})")
    cfg = MambaConfig(**PRESETS[args.preset])
    model = MambaLMHeadModel(cfg, device=device, dtype=torch.bfloat16)
    state = torch.load(args.ckpt, map_location=device)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

    n_replaced, n_params = int4_quantize_model(model)
    print(f"  Applied INT4 PTQ to {n_replaced} Linear layers ({n_params/1e6:.1f}M params quantized)")

    model.eval()

    # WikiText eval (same slice as quick_eval.py / eval_fp_baseline.py)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=False)
    texts = [ex["text"] for ex in ds.select(range(min(500, len(ds)))) if ex["text"].strip()]
    enc = tok("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    print(f"  {len(texts)} non-empty rows, {enc.size(1)} tokens")

    seqlen, stride = 1024, 512
    nlls = []
    prev = 0
    with torch.no_grad():
        for begin in range(0, enc.size(1), stride):
            end = min(begin + seqlen, enc.size(1))
            trg = end - prev
            ids = enc[:, begin:end]
            labels = ids.clone()
            labels[:, :-trg] = -100
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids).logits
                loss = F.cross_entropy(
                    logits[..., :-1, :].reshape(-1, logits.size(-1)),
                    labels[..., 1:].reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            nlls.append(loss.item())
            prev = end
            if end == enc.size(1):
                break
    ppl = math.exp(sum(nlls) / enc.size(1))
    print(f"\nMamba-3 INT4 PTQ WikiText-103 PPL: {ppl:.2f}")

    out = {
        "ckpt": args.ckpt, "preset": args.preset, "quantization": "INT4 PTQ (per-tensor symmetric, RTN)",
        "wikitext103_ppl": ppl, "wikitext103_tokens": int(enc.size(1)),
        "n_quantized_layers": n_replaced, "n_quantized_params": n_params,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
