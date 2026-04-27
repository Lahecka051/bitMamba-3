"""Evaluate a HuggingFace Mamba-2 baseline on WikiText-103 for PPL.

Loads any state-spaces/mamba2-* checkpoint via from_pretrained and computes
sliding-window PPL on the same WikiText-103 slice as our BitMamba-3 evals
(`src/evaluation/quick_eval.py`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, seqlen=1024, stride=512, device="cuda"):
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    nlls = []
    prev_end = 0
    for begin in range(0, encodings.size(1), stride):
        end = min(begin + seqlen, encodings.size(1))
        trg_len = end - prev_end
        input_ids = encodings[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                   shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
            nlls.append(loss.item())
        prev_end = end
        if end == encodings.size(1):
            break
    total = encodings.size(1)
    return math.exp(sum(nlls) / max(total, 1)), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba2-130m")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = "cuda"
    print(f"Loading {args.model}...")
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params/1e6:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    from datasets import load_dataset
    print("Loading WikiText-103 (first 500 rows)...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=False)
    texts = [ex["text"] for ex in ds.select(range(min(500, len(ds)))) if ex["text"].strip()]
    print(f"  {len(texts)} non-empty lines")

    t0 = time.time()
    ppl, tokens = compute_ppl(model, tokenizer, texts, seqlen=1024, stride=512)
    print(f"  WikiText-103 PPL: {ppl:.2f} ({tokens} tokens, {time.time()-t0:.1f}s)")

    name = args.model.replace("/", "_")
    out = Path(args.out_json) if args.out_json else _root / f"results/tables/baseline_{name}_wikitext.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "params_M": round(n_params / 1e6, 1),
        "wikitext103_ppl": ppl, "wikitext103_tokens": tokens,
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
