"""Generate paper-ready figures from accumulated results.

Outputs PNG / PDF figures to results/figures/ for use in paper.

Figures:
  Fig 1: Mamba-2 baseline throughput vs context length (3 model sizes)
  Fig 2: Parity task — ternary vs FP comparison (d=256/depth=2 main result)
  Fig 3: Scaling: train loss / WikiText PPL vs model size
  Fig 4: Needle-in-haystack heatmap (130M)
  Fig 5: Parity learning curves (5 seeds overlaid)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).resolve().parents[2]
_figs = _root / "results" / "figures"
_figs.mkdir(parents=True, exist_ok=True)
_tables = _root / "results" / "tables"


def _save(fig, name):
    fig.savefig(_figs / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(_figs / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


def fig1_baseline_throughput():
    rows = list(csv.DictReader(open(_tables / "bench_mamba_baseline.csv")))
    by_model = {}
    for r in rows:
        m = r["model"]
        by_model.setdefault(m, []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, rs in sorted(by_model.items()):
        Ls = [int(r["context_L"]) for r in rs]
        prefill = [float(r["prefill_tok_per_s"]) for r in rs if r["prefill_tok_per_s"] != "OOM"]
        decode = [float(r["decode_tok_per_s"]) for r in rs if r["decode_tok_per_s"] != "OOM"]
        label = f"{model} ({rs[0]['params_M']}M)"
        ax1.plot(Ls[:len(prefill)], prefill, marker="o", label=label)
        ax2.plot(Ls[:len(decode)], decode, marker="s", label=label)

    for ax, title, ylabel in [
        (ax1, "Prefill throughput", "tokens/sec"),
        (ax2, "Decode throughput", "tokens/sec"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Mamba-2 FP16 {title} (RTX 5090)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    plt.suptitle("Fig 1. Mamba-2 baseline throughput")
    plt.tight_layout()
    _save(fig, "fig1_baseline_throughput")


def fig2_parity_ternary_vs_fp():
    larger = json.loads((_tables / "parity_larger.json").read_text())
    by_config = {}
    for r in larger:
        key = f"{r['arch']}{'_bit' if r.get('bitize') else ''}"
        by_config.setdefault(key, []).append(r["peak_acc_all"])

    configs = ["mamba3_siso", "mamba3_siso_bit", "mamba3_mimo", "mamba3_mimo_bit"]
    labels = ["SISO FP", "SISO + ternary", "MIMO FP", "MIMO + ternary"]
    means = [np.mean(by_config.get(c, [0])) for c in configs]
    stds = [np.std(by_config.get(c, [0]), ddof=1) for c in configs]
    colors = ["#888", "#1f77b4", "#aaa", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Peak parity accuracy (5 seeds, μ±σ)")
    ax.set_title("Fig 2. Ternary quantization as state-tracking inductive bias\n(d=256, depth=2, cosine LR, 5K steps)")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f"{m:.3f}\n±{s:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01),
                    ha="center", va="bottom", fontsize=8)

    _save(fig, "fig2_parity_ternary_vs_fp")


def fig3_scaling_curves():
    runs = []
    for ck_label, file in [
        ("30M @ 164M tok", "quick_eval_ckpt_final_010000.json"),
        ("30M @ 480M tok", "quick_eval_30M_long.json"),
        ("130M @ 480M tok", "quick_eval_130M.json"),
    ]:
        p = _tables / file
        if p.exists():
            d = json.loads(p.read_text())
            runs.append((ck_label, d.get("wikitext103_ppl")))

    if not runs:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [r[0] for r in runs]
    ppls = [r[1] for r in runs]
    ax.bar(labels, ppls, color=["#888", "#aaa", "#ff7f0e"])
    for i, p in enumerate(ppls):
        ax.annotate(f"{p:.1f}", (i, p), ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("WikiText-103 PPL (log)")
    ax.set_title("Fig 3. BitMamba-3 PPL scaling: model size + training tokens")
    ax.grid(axis="y", alpha=0.3, which="both")
    _save(fig, "fig3_scaling_curves")


def fig4_needle_heatmap():
    p = _tables / "needle_130M.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    results = d.get("results", {})
    Ls = sorted(results.keys(), key=int)
    depths = sorted({int(d) for L in results.values() for d in L})

    matrix = np.array([[results[L].get(str(d), {"mean_log_prob": np.nan})["mean_log_prob"]
                        for d in depths] for L in Ls])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-12, vmax=-3)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"{d}%" for d in depths])
    ax.set_yticks(range(len(Ls)))
    ax.set_yticklabels([f"L={L}" for L in Ls])
    ax.set_xlabel("Needle depth (% through context)")
    ax.set_ylabel("Context length")
    ax.set_title("Fig 4. Needle-in-Haystack 130M: avg log-prob (higher = better recall)")
    fig.colorbar(im, ax=ax, label="log-prob")

    for i, L in enumerate(Ls):
        for j, d in enumerate(depths):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=9)

    _save(fig, "fig4_needle_heatmap")


def fig5_parity_curves():
    p = _tables / "parity_larger.json"
    if not p.exists():
        return
    runs = json.loads(p.read_text())

    # Group by config
    by_config = {}
    for r in runs:
        key = f"{r['arch']}{'_bit' if r.get('bitize') else ''}"
        by_config.setdefault(key, []).append(r["history"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, sharex=True)
    config_to_ax = {
        "mamba3_siso": (0, 0, "Mamba-3 SISO FP", "#666"),
        "mamba3_siso_bit": (0, 1, "Mamba-3 SISO + ternary", "#1f77b4"),
        "mamba3_mimo": (1, 0, "Mamba-3 MIMO FP", "#aaa"),
        "mamba3_mimo_bit": (1, 1, "Mamba-3 MIMO + ternary", "#ff7f0e"),
    }
    for key, (r, c, title, color) in config_to_ax.items():
        ax = axes[r][c]
        if key in by_config:
            for hist in by_config[key]:
                steps = [h["step"] for h in hist]
                accs = [h["acc_all"] for h in hist]
                ax.plot(steps, accs, color=color, alpha=0.6)
            ax.axhline(0.5, color="red", linestyle="--", alpha=0.4)
        ax.set_title(title)
        ax.set_ylim(0.3, 1.05)
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("Parity accuracy (acc_all)")

    plt.suptitle("Fig 5. Parity learning curves — 5 seeds per config (d=256, depth=2)")
    plt.tight_layout()
    _save(fig, "fig5_parity_curves")


def main():
    print("Generating paper figures...")
    try:
        fig1_baseline_throughput()
    except Exception as e:
        print(f"  Fig 1 failed: {e}")
    try:
        fig2_parity_ternary_vs_fp()
    except Exception as e:
        print(f"  Fig 2 failed: {e}")
    try:
        fig3_scaling_curves()
    except Exception as e:
        print(f"  Fig 3 failed: {e}")
    try:
        fig4_needle_heatmap()
    except Exception as e:
        print(f"  Fig 4 failed: {e}")
    try:
        fig5_parity_curves()
    except Exception as e:
        print(f"  Fig 5 failed: {e}")
    print(f"\nDone. Figures in {_figs}")


if __name__ == "__main__":
    main()
