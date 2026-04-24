"""Matrix benchmark: Mamba-2 FP16 baseline at multiple model sizes x context lengths.

Separates prefill and decode throughput by calling forward() and step() directly
via the model's generate path, measuring each phase independently.

Outputs CSV to results/tables/bench_mamba_baseline.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
_mamba_src = _root / "third_party" / "state-spaces-mamba"
if str(_mamba_src) not in sys.path:
    sys.path.insert(0, str(_mamba_src))

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore  # noqa: E402
from mamba_ssm.utils.generation import InferenceParams  # type: ignore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["state-spaces/mamba2-130m"])
    p.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    p.add_argument("--decode_steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out_csv", default="results/tables/bench_mamba_baseline.csv")
    return p.parse_args()


@torch.no_grad()
def measure(model, vocab_size, L, decode_steps, batch, device, warmup, repeats):
    """Return (prefill_tok_s, decode_tok_s, peak_mem_gb)."""
    dtype = next(model.parameters()).dtype
    torch.cuda.reset_peak_memory_stats()

    prefill_ts, decode_ts = [], []

    for r in range(warmup + repeats):
        input_ids = torch.randint(0, vocab_size, (batch, L), device=device, dtype=torch.long)

        inference_params = InferenceParams(max_seqlen=L + decode_steps, max_batch_size=batch)

        # Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(input_ids, inference_params=inference_params).logits
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_params.seqlen_offset = L

        # Decode
        next_tok = logits[:, -1:, :].argmax(dim=-1)
        t2 = time.perf_counter()
        for _ in range(decode_steps):
            out = model(next_tok, inference_params=inference_params)
            inference_params.seqlen_offset += 1
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if r >= warmup:
            prefill_ts.append(t1 - t0)
            decode_ts.append(t3 - t2)

    prefill_tok_s = (L * batch) / (sum(prefill_ts) / len(prefill_ts))
    decode_tok_s = (decode_steps * batch) / (sum(decode_ts) / len(decode_ts))
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return prefill_tok_s, decode_tok_s, peak_mem


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = "cuda"

    out_path = _root / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_id in args.models:
        print(f"\n=== Loading {model_id} ===")
        try:
            model = MambaLMHeadModel.from_pretrained(model_id, device=device, dtype=dtype)
            model.eval()
        except Exception as e:
            print(f"  SKIP: {type(e).__name__}: {e}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        vocab_size = model.config.vocab_size
        print(f"  params={n_params/1e6:.1f}M, vocab={vocab_size}, dtype={dtype}")

        for L in args.contexts:
            try:
                prefill_ts, decode_ts, peak_mem = measure(
                    model, vocab_size, L, args.decode_steps, args.batch, device,
                    args.warmup, args.repeats
                )
                print(
                    f"  L={L:>6} : prefill={prefill_ts:>8.1f} tok/s  "
                    f"decode={decode_ts:>7.1f} tok/s  peak_mem={peak_mem:.2f} GB"
                )
                rows.append({
                    "model": model_id,
                    "params_M": round(n_params / 1e6, 1),
                    "dtype": args.dtype,
                    "context_L": L,
                    "decode_steps": args.decode_steps,
                    "batch": args.batch,
                    "prefill_tok_per_s": round(prefill_ts, 2),
                    "decode_tok_per_s": round(decode_ts, 2),
                    "peak_mem_gb": round(peak_mem, 3),
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  L={L:>6} : OOM")
                torch.cuda.empty_cache()
                rows.append({
                    "model": model_id,
                    "params_M": round(n_params / 1e6, 1),
                    "dtype": args.dtype,
                    "context_L": L,
                    "decode_steps": args.decode_steps,
                    "batch": args.batch,
                    "prefill_tok_per_s": "OOM",
                    "decode_tok_per_s": "OOM",
                    "peak_mem_gb": "OOM",
                })
            except Exception as e:
                print(f"  L={L:>6} : FAIL {type(e).__name__}: {str(e)[:120]}")
                rows.append({
                    "model": model_id,
                    "params_M": round(n_params / 1e6, 1),
                    "dtype": args.dtype,
                    "context_L": L,
                    "decode_steps": args.decode_steps,
                    "batch": args.batch,
                    "prefill_tok_per_s": "FAIL",
                    "decode_tok_per_s": "FAIL",
                    "peak_mem_gb": "FAIL",
                })

        del model
        torch.cuda.empty_cache()

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
