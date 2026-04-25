"""Run lm-evaluation-harness on a BitMamba-3 local checkpoint.

Uses our `BitMamba3LMWrapper` registered via `bitmamba3` model name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

# Trigger model registration (BitMamba3LMWrapper)
from evaluation import bitmamba3_lm_eval  # noqa: F401, E402

from lm_eval import simple_evaluate  # type: ignore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="30M")
    ap.add_argument("--tasks", nargs="+",
                    default=["lambada_openai", "hellaswag", "arc_easy", "arc_challenge",
                             "winogrande", "piqa", "boolq", "openbookqa"])
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit examples per task (None=all)")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    out_path = Path(args.out_json) if args.out_json else _root / f"results/tables/lm_eval_{Path(args.ckpt).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running lm-eval on {args.ckpt} (preset={args.preset})")
    print(f"Tasks: {args.tasks}")

    results = simple_evaluate(
        model="bitmamba3",
        model_args=f"preset={args.preset},ckpt={args.ckpt},tokenizer=EleutherAI/gpt-neox-20b,dtype=bfloat16,batch_size={args.batch_size},max_length=2048",
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # Save trimmed results
    saved = {
        "ckpt": args.ckpt,
        "preset": args.preset,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "results": results.get("results", {}) if results else {},
        "configs": results.get("configs", {}) if results else {},
    }
    out_path.write_text(json.dumps(saved, indent=2, default=str))
    print(f"\nSaved to {out_path}")
    if "results" in saved:
        for task, scores in saved["results"].items():
            print(f"  {task}: {scores}")


if __name__ == "__main__":
    main()
