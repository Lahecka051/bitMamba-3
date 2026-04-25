"""Aggregate `results/tables/parity_multiseed.json` into a paper-ready table.

Computes per-config (arch + bitize) statistics across seeds:
  - mean ± std of peak_acc_all
  - mean ± std of final_acc_all
  - mean ± std of acc_2x_seqlen_all
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

_root = Path(__file__).resolve().parents[2]


def main():
    in_path = _root / "results/tables/parity_multiseed.json"
    if not in_path.exists():
        print(f"Missing {in_path}. Run scripts/run_parity_multiseed_bg.py first.")
        return

    rows = json.loads(in_path.read_text())
    by_config = {}
    for r in rows:
        key = f"{r['arch']}{'_bit' if r.get('bitize') else ''}"
        by_config.setdefault(key, []).append(r)

    print("\n## Multi-seed Parity Task Results")
    print(f"  d_model={rows[0]['d_model']}, seqlen={rows[0]['seqlen']}, n_steps={rows[0]['n_steps']}\n")

    cols = ["config", "n_seeds", "peak_mean", "peak_std", "final_mean", "final_std",
            "2x_mean", "2x_std"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")

    for key, runs in sorted(by_config.items()):
        peaks = [r["peak_acc_all"] for r in runs if r.get("peak_acc_all") is not None]
        finals = [r["final_acc_all"] for r in runs if r.get("final_acc_all") is not None]
        twox = [r["acc_2x_seqlen_all"] for r in runs if r.get("acc_2x_seqlen_all") is not None]

        def mean_std(xs):
            if not xs:
                return ("N/A", "N/A")
            m = statistics.mean(xs)
            s = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return (f"{m:.4f}", f"{s:.4f}")

        pm, ps = mean_std(peaks)
        fm, fs = mean_std(finals)
        tm, ts = mean_std(twox)
        print(f"| {key} | {len(runs)} | {pm} | {ps} | {fm} | {fs} | {tm} | {ts} |")


if __name__ == "__main__":
    main()
