"""Prepare training data: download + tokenize + shard into memmap files.

Uses SlimPajama-627B (or a subset) via HuggingFace datasets streaming.
Tokenized with GPT-NeoX tokenizer (Mamba-2 default).

Output format: data/tokens_<split>.bin (int32 memmap), aligned to seqlen boundaries.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cerebras/SlimPajama-627B")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--target_tokens", type=int, default=1_000_000_000, help="1B default")
    p.add_argument("--out_dir", default="data/slimpajama_1B")
    p.add_argument("--shard_size", type=int, default=100_000_000, help="tokens per shard")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = _root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {args.tokenizer}...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tok.vocab_size
    print(f"  vocab_size={vocab_size}")

    print(f"Loading dataset {args.dataset} (streaming)...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    dtype = np.uint32 if vocab_size > 65535 else np.uint16
    print(f"  using dtype={dtype} (vocab fits={np.iinfo(dtype).max+1})")

    shard_idx = 0
    shard_buf = []
    shard_count = 0
    total_tokens = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        if total_tokens >= args.target_tokens:
            break
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(tok.eos_token_id if tok.eos_token_id is not None else 0)
        shard_buf.extend(ids)
        shard_count += len(ids)
        total_tokens += len(ids)

        if shard_count >= args.shard_size:
            arr = np.array(shard_buf, dtype=dtype)
            path = out_dir / f"shard_{shard_idx:04d}.bin"
            arr.tofile(path)
            elapsed = time.time() - t0
            print(f"  shard {shard_idx}: {shard_count:,} tokens -> {path.name}  "
                  f"(total={total_tokens/1e6:.1f}M, {total_tokens/1e6/elapsed:.1f}M tok/s)")
            shard_buf = []
            shard_count = 0
            shard_idx += 1

    if shard_buf:
        arr = np.array(shard_buf, dtype=dtype)
        path = out_dir / f"shard_{shard_idx:04d}.bin"
        arr.tofile(path)
        print(f"  final shard {shard_idx}: {len(shard_buf):,} tokens -> {path.name}")

    meta = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "total_tokens": total_tokens,
        "num_shards": shard_idx + (1 if shard_buf else 0),
        "shard_size_target": args.shard_size,
    }
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone: {total_tokens:,} tokens across {meta['num_shards']} shards in {out_dir}")


if __name__ == "__main__":
    main()
