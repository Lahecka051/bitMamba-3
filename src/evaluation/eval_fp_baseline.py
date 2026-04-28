"""Evaluate the Mamba-3 FP baseline (no bitify) on WikiText-103.

Used to quantify the pure quantization cost of BitMamba-3 vs an
identically-trained Mamba-3 FP model.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="130M")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    cfg = MambaConfig(**PRESETS[args.preset])
    model = MambaLMHeadModel(cfg, device="cuda", dtype=torch.bfloat16)
    state = torch.load(args.ckpt, map_location="cuda")
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=False)
    texts = [ex["text"] for ex in ds.select(range(min(500, len(ds)))) if ex["text"].strip()]
    enc = tok("\n\n".join(texts), return_tensors="pt").input_ids.cuda()
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
    print(f"WikiText-103 PPL: {ppl:.2f}")

    out = {"ckpt": args.ckpt, "preset": args.preset,
           "wikitext103_ppl": ppl, "wikitext103_tokens": int(enc.size(1))}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
