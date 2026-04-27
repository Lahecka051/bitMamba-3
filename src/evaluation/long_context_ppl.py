"""Long-context perplexity evaluation on PG19 (book-length text).

PG19 (Project Gutenberg books from 1919) is a standard long-document
language modeling benchmark. We compute sliding-window PPL at multiple
context lengths to expose any long-range modeling advantage.

Usage:
    python src/evaluation/long_context_ppl.py \
        --ckpt checkpoints/bitmamba3_370M_mimo/ckpt_final_030000.pt \
        --preset 370M --max_seqlen 8192
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

from bitmamba3 import ensure_mamba3_registered  # noqa: E402
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
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def ppl_at_seqlen(model, tokenizer, texts, seqlen, stride=None, device="cuda"):
    if stride is None:
        stride = seqlen // 2
    text_blob = "\n\n".join(texts)
    enc = tokenizer(text_blob, return_tensors="pt").input_ids.to(device)
    nlls, total = [], 0
    prev_end = 0
    for begin in range(0, enc.size(1), stride):
        end = min(begin + seqlen, enc.size(1))
        trg_len = end - prev_end
        ids = enc[:, begin:end]
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ids).logits
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=-100, reduction="sum",
            )
        nlls.append(loss.item())
        total = end
        prev_end = end
        if end == enc.size(1):
            break
    return math.exp(sum(nlls) / max(total, 1)), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--n_books", type=int, default=10, help="Number of PG19 books to evaluate")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = "cuda"
    print(f"Loading {args.ckpt} (preset={args.preset})")
    model = load_ckpt(args.ckpt, args.preset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading PG19 first {args.n_books} books...")
    from datasets import load_dataset
    try:
        ds = load_dataset("emozilla/pg19", split="test", streaming=False)
    except Exception:
        # fallback: emozilla/pg19-test, deepmind/pg19
        try:
            ds = load_dataset("deepmind/pg19", split="test", streaming=False)
        except Exception as e:
            print(f"Could not load PG19: {e}")
            return
    texts = [ex["text"] for ex in ds.select(range(min(args.n_books, len(ds))))]
    char_total = sum(len(t) for t in texts)
    print(f"  {len(texts)} books, {char_total:,} chars")

    results = {}
    for L in args.seqlens:
        t0 = time.time()
        ppl, tokens = ppl_at_seqlen(model, tokenizer, texts, seqlen=L, stride=L // 2)
        elapsed = time.time() - t0
        results[L] = {"ppl": ppl, "tokens": tokens, "elapsed_s": elapsed}
        print(f"  L={L:>5}: PPL={ppl:.2f} ({tokens} tokens, {elapsed:.1f}s)")

    out_path = Path(args.out_json) if args.out_json else _root / f"results/tables/long_context_{args.preset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "preset": args.preset,
        "n_books": args.n_books, "seqlens": args.seqlens,
        "results": results,
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
