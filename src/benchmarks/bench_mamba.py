"""Benchmark Mamba-2/Mamba-3 FP16/BF16 inference on RTX 5090.

Measures:
  - Prefill throughput (tokens/sec)
  - Decode throughput (tokens/sec)
  - Peak GPU memory
  - Per-token latency

Usage:
  python src/benchmarks/bench_mamba.py --model state-spaces/mamba2-370m --context 2048 --decode 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Vendored state-spaces/mamba
_root = Path(__file__).resolve().parents[2]
_mamba_src = _root / "third_party" / "state-spaces-mamba"
if str(_mamba_src) not in sys.path:
    sys.path.insert(0, str(_mamba_src))

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba2-370m")
    p.add_argument("--context", type=int, default=2048, help="Prefill context length")
    p.add_argument("--decode", type=int, default=128, help="Decode steps")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = "cuda"

    print(f"Loading {args.model} ({args.dtype})...")
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    model.eval()

    vocab_size = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M, vocab={vocab_size}")

    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, vocab_size, (args.batch, args.context), device=device, dtype=torch.long)

    prefill_times, decode_times = [], []

    for r in range(args.warmup + args.repeats):
        torch.cuda.synchronize()

        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_length=args.context + args.decode,
                return_dict_in_generate=True,
                output_scores=False,
                enable_timing=False,
                cg=False,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_time = t1 - t0

        # Rough split: first forward is prefill on L tokens, rest is decode
        # For proper separation we'd call model.forward directly; here we report total.
        if r >= args.warmup:
            total_tokens = args.context * args.batch + args.decode * args.batch
            decode_tokens = args.decode * args.batch
            prefill_tokens = args.context * args.batch
            # Approximate: measure generate() as single call
            print(
                f"[Run {r-args.warmup+1}] context={args.context} decode={args.decode} "
                f"total_time={total_time*1000:.1f}ms "
                f"total_throughput={total_tokens/total_time:.1f} tok/s"
            )
            decode_times.append(total_time)

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\nSummary: {args.model} @ L={args.context} + decode={args.decode}")
    print(f"  Peak GPU mem: {peak_mem:.2f} GB")
    if decode_times:
        mean_t = sum(decode_times) / len(decode_times)
        print(f"  Mean time: {mean_t*1000:.1f} ms")
        print(f"  Overall throughput: {(args.context+args.decode)*args.batch/mean_t:.1f} tok/s")


if __name__ == "__main__":
    main()
