"""Quick-turnaround evaluation for a BitMamba-3 checkpoint.

Metrics:
  - WikiText-103 / LAMBADA token-level NLL and perplexity
  - Basic generation quality check (greedy continuation of fixed prompts)
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
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered
ensure_mamba3_registered()

from transformers import AutoTokenizer
from mamba_ssm.models.config_mamba import MambaConfig  # type: ignore
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore

from training.train import PRESETS, bitify_model  # type: ignore


def load_ckpt(ckpt_path, preset, device="cuda", dtype=torch.bfloat16):
    cfg = MambaConfig(**PRESETS[preset])
    model = MambaLMHeadModel(cfg, device=device, dtype=dtype)
    bitify_model(model)
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


@torch.no_grad()
def compute_ppl(model, tokenizer, texts, seqlen=2048, stride=1024, device="cuda"):
    """Compute sliding-window perplexity."""
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    max_length = min(seqlen, encodings.size(1))
    nlls = []
    prev_end = 0
    for begin in range(0, encodings.size(1), stride):
        end = min(begin + max_length, encodings.size(1))
        trg_len = end - prev_end
        input_ids = encodings[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            nlls.append(loss.item())
        prev_end = end
        if end == encodings.size(1):
            break
    total_tokens = encodings.size(1)
    ppl = math.exp(sum(nlls) / max(total_tokens, 1))
    return ppl, total_tokens


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_new=50, device="cuda"):
    outputs = []
    for p in prompts:
        ids = tokenizer.encode(p, return_tensors="pt").to(device)
        for _ in range(max_new):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids).logits
            next_id = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=1)
        outputs.append(tokenizer.decode(ids[0]))
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="30M")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    model = load_ckpt(args.ckpt, args.preset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- WikiText-103 PPL (use a small slice for speed) ---
    from datasets import load_dataset
    print("\nLoading WikiText-103 (first 500 rows)...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=False)
        texts = [ex["text"] for ex in ds.select(range(min(500, len(ds)))) if ex["text"].strip()]
        print(f"  {len(texts)} non-empty lines")
        t0 = time.time()
        ppl, tokens = compute_ppl(model, tokenizer, texts, seqlen=1024, stride=512)
        print(f"  WikiText-103 slice PPL: {ppl:.2f} ({tokens} tokens, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  WikiText eval failed: {e}")
        ppl, tokens = None, None

    # --- Generation samples ---
    print("\nGeneration samples (greedy):")
    prompts = [
        "The quick brown fox",
        "In the beginning, there was",
        "Machine learning models can",
    ]
    samples = generate_samples(model, tokenizer, prompts, max_new=30)
    for s in samples:
        print(f"  > {s}")

    result = {
        "ckpt": args.ckpt,
        "preset": args.preset,
        "wikitext103_ppl": ppl,
        "wikitext103_tokens": tokens,
        "generation_samples": samples,
    }
    out_path = _root / (args.out_json or f"results/tables/quick_eval_{Path(args.ckpt).stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
