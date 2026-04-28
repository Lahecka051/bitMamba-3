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
        ("370M @ 480M tok", "quick_eval_370M.json"),
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


def fig6_parity_scaling_progression():
    """d=128 / d=256 / d=512 progression of MIMO+ternary effect."""
    sweeps = [
        ("d=128, depth=1", _tables / "parity_multiseed.json"),
        ("d=256, depth=2", _tables / "parity_larger.json"),
        ("d=512, depth=4", _tables / "parity_d512.json"),
    ]
    configs = ["mamba3_mimo", "mamba3_mimo_bit", "mamba3_siso", "mamba3_siso_bit"]
    labels = ["MIMO FP", "MIMO + ternary", "SISO FP", "SISO + ternary"]
    colors = ["#aaa", "#ff7f0e", "#888", "#1f77b4"]

    means = {c: [] for c in configs}
    stds = {c: [] for c in configs}

    for sweep_label, path in sweeps:
        if not path.exists():
            for c in configs:
                means[c].append(np.nan)
                stds[c].append(np.nan)
            continue
        runs = json.loads(path.read_text())
        by = {}
        for r in runs:
            key = f"{r['arch']}{'_bit' if r.get('bitize') else ''}"
            by.setdefault(key, []).append(r["peak_acc_all"])
        for c in configs:
            vals = by.get(c, [])
            if vals:
                means[c].append(float(np.mean(vals)))
                stds[c].append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            else:
                means[c].append(np.nan)
                stds[c].append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sweeps))
    width = 0.2
    for i, (c, label, color) in enumerate(zip(configs, labels, colors)):
        ax.bar(x + (i - 1.5) * width, means[c], width, yerr=stds[c],
               capsize=4, label=label, color=color, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in sweeps])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random")
    ax.set_ylabel("Peak parity accuracy (5 seeds, μ±σ)")
    ax.set_title("Fig 6. Inductive-bias effect strengthens with scale\n(d=128/256/512 — MIMO+ternary reaches 13σ at d=512)")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig6_parity_scaling_progression")


def fig7_bitmamba2_vs_bitmamba3():
    """BitMamba-2 vs BitMamba-3 130M comparison across metrics."""
    files = {
        "BitMamba-2": _tables / "lm_eval_130M_mamba2_ckpt_final_030000.json",
        "BitMamba-3": _tables / "lm_eval_130M.json",
    }
    quick = {
        "BitMamba-2": _tables / "quick_eval_bitmamba2_130M.json",
        "BitMamba-3": _tables / "quick_eval_130M.json",
    }
    for label, p in {**files, **quick}.items():
        if not p.exists():
            print(f"  fig7 missing: {p}")
            return

    bm2_le = json.loads(files["BitMamba-2"].read_text()).get("results", {})
    bm3_le = json.loads(files["BitMamba-3"].read_text()).get("results", {})
    bm2_qe = json.loads(quick["BitMamba-2"].read_text())
    bm3_qe = json.loads(quick["BitMamba-3"].read_text())

    metrics = [
        ("WikiText PPL", bm2_qe["wikitext103_ppl"], bm3_qe["wikitext103_ppl"], True),  # lower better
        ("LAMBADA PPL", bm2_le["lambada_openai"]["perplexity,none"], bm3_le["lambada_openai"]["perplexity,none"], True),
        ("LAMBADA acc", bm2_le["lambada_openai"]["acc,none"], bm3_le["lambada_openai"]["acc,none"], False),
        ("HellaSwag norm", bm2_le["hellaswag"]["acc_norm,none"], bm3_le["hellaswag"]["acc_norm,none"], False),
        ("ARC-Easy acc", bm2_le["arc_easy"]["acc,none"], bm3_le["arc_easy"]["acc,none"], False),
        ("PIQA acc", bm2_le["piqa"]["acc,none"], bm3_le["piqa"]["acc,none"], False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Left: PPLs (log scale, lower is better)
    ppl_metrics = [m for m in metrics if m[3]]
    other_metrics = [m for m in metrics if not m[3]]

    x = np.arange(len(ppl_metrics))
    w = 0.35
    axes[0].bar(x - w / 2, [m[1] for m in ppl_metrics], w, label="BitMamba-2 (Mamba-2 + ternary)", color="#888")
    axes[0].bar(x + w / 2, [m[2] for m in ppl_metrics], w, label="BitMamba-3 (Mamba-3 + ternary)", color="#ff7f0e")
    for i, (label, m2, m3, _) in enumerate(ppl_metrics):
        ratio = m2 / m3
        axes[0].annotate(f"{ratio:.2f}× ↓", (i, max(m2, m3) * 1.05), ha="center", fontsize=9)
        axes[0].annotate(f"{m2:.0f}", (i - w/2, m2), ha="center", va="bottom", fontsize=8)
        axes[0].annotate(f"{m3:.0f}", (i + w/2, m3), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m[0] for m in ppl_metrics])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("PPL (log, lower = better)")
    axes[0].set_title("Perplexity")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3, which="both")

    # Right: zero-shot accuracy (higher is better)
    x = np.arange(len(other_metrics))
    axes[1].bar(x - w / 2, [m[1] for m in other_metrics], w, label="BitMamba-2", color="#888")
    axes[1].bar(x + w / 2, [m[2] for m in other_metrics], w, label="BitMamba-3", color="#ff7f0e")
    for i, (label, m2, m3, _) in enumerate(other_metrics):
        axes[1].annotate(f"{m2:.3f}", (i - w/2, m2), ha="center", va="bottom", fontsize=8)
        axes[1].annotate(f"{m3:.3f}", (i + w/2, m3), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m[0] for m in other_metrics], rotation=15, fontsize=8)
    axes[1].set_ylabel("Accuracy (higher = better)")
    axes[1].set_title("Zero-shot downstream (200 samples)")
    axes[1].set_ylim(0, 0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Fig 7. BitMamba-2 vs BitMamba-3 at 130M (same 480M tokens, same training)")
    plt.tight_layout()
    _save(fig, "fig7_bitmamba2_vs_bitmamba3")


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
    try:
        fig6_parity_scaling_progression()
    except Exception as e:
        print(f"  Fig 6 failed: {e}")
    try:
        fig7_bitmamba2_vs_bitmamba3()
    except Exception as e:
        print(f"  Fig 7 failed: {e}")
    print(f"\nDone. Figures in {_figs}")


if __name__ == "__main__":
    main()
