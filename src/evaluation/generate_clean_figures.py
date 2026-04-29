"""Generate caption-free figures for Word document insertion.

Identical content to generate_paper_figures.py but with all set_title /
suptitle calls suppressed. Outputs to results/figures_clean/ as PNG and JPG.

Used when the figure caption will be supplied by the document (e.g., Word's
'Insert Caption' feature) rather than baked into the image itself.
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
_figs = _root / "results" / "figures_clean"
_figs.mkdir(parents=True, exist_ok=True)
_tables = _root / "results" / "tables"


def _save(fig, name):
    """Save figure as PNG only (caption-free) for Word manual captioning."""
    png = _figs / f"{name}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png.name}")


def fig1_baseline_throughput():
    rows = list(csv.DictReader(open(_tables / "bench_mamba_baseline.csv")))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, rs in sorted(by_model.items()):
        Ls = [int(r["context_L"]) for r in rs]
        prefill = [float(r["prefill_tok_per_s"]) for r in rs if r["prefill_tok_per_s"] != "OOM"]
        decode = [float(r["decode_tok_per_s"]) for r in rs if r["decode_tok_per_s"] != "OOM"]
        label = f"{model} ({rs[0]['params_M']}M)"
        ax1.plot(Ls[:len(prefill)], prefill, marker="o", label=label)
        ax2.plot(Ls[:len(decode)], decode, marker="s", label=label)

    for ax, ylabel in [(ax1, "Prefill tokens/sec"), (ax2, "Decode tokens/sec")]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

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

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Peak parity accuracy (5 seeds, μ±σ)")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random")
    # Higher ylim to keep "0.949 ±0.091" annotations inside the plot box
    ax.set_ylim(0.4, 1.20)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f"{m:.3f}\n±{s:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
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

    # Wider figure + rotated labels prevent x-tick overlap
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r[0] for r in runs]
    ppls = [r[1] for r in runs]
    colors = ["#888", "#aaa", "#ff7f0e", "#d62728"][:len(runs)]
    ax.bar(labels, ppls, color=colors, edgecolor="black")
    for i, p in enumerate(ppls):
        ax.annotate(f"{p:.1f}", (i, p), ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("WikiText-103 PPL (log, lower = better)")
    # Compact x labels to reduce overlap risk; rotate as backup
    ax.set_xticklabels([l.replace(" @ ", "\n@ ") for l in labels], rotation=0, fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
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
    for key, (r, c, sub_title, color) in config_to_ax.items():
        ax = axes[r][c]
        if key in by_config:
            for hist in by_config[key]:
                steps = [h["step"] for h in hist]
                accs = [h["acc_all"] for h in hist]
                ax.plot(steps, accs, color=color, alpha=0.6)
            ax.axhline(0.5, color="red", linestyle="--", alpha=0.4)
        # Sub-titles per panel are kept (they identify the panel content,
        # not a figure-level caption); paper captions can refer to them.
        ax.set_title(sub_title, fontsize=10)
        ax.set_ylim(0.3, 1.05)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("Parity accuracy (acc_all)")
    plt.tight_layout()
    _save(fig, "fig5_parity_curves")


def fig6_parity_scaling_progression():
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
                means[c].append(np.nan); stds[c].append(np.nan)
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
                means[c].append(np.nan); stds[c].append(np.nan)

    # Wider figure to hold legend outside the plot area (legend in plot
    # was overlapping the d=512 MIMO+ternary and SISO+ternary bars)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(len(sweeps))
    width = 0.2
    for i, (c, label, color) in enumerate(zip(configs, labels, colors)):
        ax.bar(x + (i - 1.5) * width, means[c], width, yerr=stds[c],
               capsize=4, label=label, color=color, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in sweeps])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random")
    ax.set_ylabel("Peak parity accuracy (5 seeds, μ±σ)")
    ax.set_ylim(0.4, 1.15)
    # Legend OUTSIDE the plot so the d=512 bars are fully visible
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig6_parity_scaling_progression")


def fig7_bitmamba2_vs_bitmamba3():
    files_130 = {
        "BitMamba-2 130M": _tables / "lm_eval_130M_mamba2_ckpt_final_030000.json",
        "BitMamba-3 130M": _tables / "lm_eval_130M.json",
    }
    files_370 = {
        "BitMamba-2 370M": _tables / "lm_eval_370M_mamba2_ckpt_final_030000.json",
        "BitMamba-3 370M": _tables / "lm_eval_370M.json",
    }
    quick_130 = {
        "BitMamba-2 130M": _tables / "quick_eval_bitmamba2_130M.json",
        "BitMamba-3 130M": _tables / "quick_eval_130M.json",
    }
    quick_370 = {
        "BitMamba-2 370M": _tables / "quick_eval_bitmamba2_370M.json",
        "BitMamba-3 370M": _tables / "quick_eval_370M.json",
    }
    files = {**files_130, **files_370}
    quick = {**quick_130, **quick_370}
    for label, p in {**files, **quick}.items():
        if not p.exists():
            return

    le = {k: json.loads(v.read_text()).get("results", {}) for k, v in files.items()}
    qe = {k: json.loads(v.read_text()) for k, v in quick.items()}

    scales = ["130M", "370M"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    x = np.arange(len(scales)); w = 0.35
    m2_w = [qe[f"BitMamba-2 {s}"]["wikitext103_ppl"] for s in scales]
    m3_w = [qe[f"BitMamba-3 {s}"]["wikitext103_ppl"] for s in scales]
    axes[0, 0].bar(x - w/2, m2_w, w, label="BitMamba-2", color="#888")
    axes[0, 0].bar(x + w/2, m3_w, w, label="BitMamba-3", color="#ff7f0e")
    for i, (m2, m3) in enumerate(zip(m2_w, m3_w)):
        axes[0, 0].annotate(f"{m2:.1f}", (i - w/2, m2), ha="center", va="bottom", fontsize=9)
        axes[0, 0].annotate(f"{m3:.1f}", (i + w/2, m3), ha="center", va="bottom", fontsize=9)
        axes[0, 0].annotate(f"{m2/m3:.2f}× ↓", (i, max(m2, m3) * 1.1), ha="center",
                            fontsize=9, color="darkgreen", weight="bold")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(scales)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("WikiText-103 PPL (log)")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(axis="y", alpha=0.3, which="both")

    m2_l = [le[f"BitMamba-2 {s}"]["lambada_openai"]["perplexity,none"] for s in scales]
    m3_l = [le[f"BitMamba-3 {s}"]["lambada_openai"]["perplexity,none"] for s in scales]
    axes[0, 1].bar(x - w/2, m2_l, w, label="BitMamba-2", color="#888")
    axes[0, 1].bar(x + w/2, m3_l, w, label="BitMamba-3", color="#ff7f0e")
    for i, (m2, m3) in enumerate(zip(m2_l, m3_l)):
        axes[0, 1].annotate(f"{m2:.0f}", (i - w/2, m2), ha="center", va="bottom", fontsize=9)
        axes[0, 1].annotate(f"{m3:.0f}", (i + w/2, m3), ha="center", va="bottom", fontsize=9)
        axes[0, 1].annotate(f"{m2/m3:.2f}× ↓", (i, max(m2, m3) * 1.1), ha="center",
                            fontsize=9, color="darkgreen", weight="bold")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(scales)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("LAMBADA PPL (log)")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(axis="y", alpha=0.3, which="both")

    other_tasks = [
        ("HellaSwag", "hellaswag", "acc_norm,none"),
        ("ARC-Easy", "arc_easy", "acc,none"),
        ("PIQA", "piqa", "acc,none"),
        ("LAMBADA", "lambada_openai", "acc,none"),
    ]
    for ax_idx, scale in enumerate(scales):
        ax = axes[1, ax_idx]
        x = np.arange(len(other_tasks))
        m2_vals = [le[f"BitMamba-2 {scale}"][t[1]][t[2]] for t in other_tasks]
        m3_vals = [le[f"BitMamba-3 {scale}"][t[1]][t[2]] for t in other_tasks]
        ax.bar(x - w/2, m2_vals, w, label="BitMamba-2", color="#888")
        ax.bar(x + w/2, m3_vals, w, label="BitMamba-3", color="#ff7f0e")
        for i, (m2, m3) in enumerate(zip(m2_vals, m3_vals)):
            ax.annotate(f"{m2:.2f}", (i - w/2, m2), ha="center", va="bottom", fontsize=8)
            ax.annotate(f"{m3:.2f}", (i + w/2, m3), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([t[0] for t in other_tasks], fontsize=9)
        ax.set_ylabel("Accuracy")
        # Sub-panel label kept (panel identification only)
        ax.set_title(f"{scale}", fontsize=10)
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig7_bitmamba2_vs_bitmamba3")


def fig8_quant_cost_decomposition():
    cells = {}
    files = {
        ("Mamba-3", "ternary"): _tables / "quick_eval_130M.json",
        ("Mamba-3", "FP"): _tables / "quick_eval_mamba3_130M_fp.json",
        ("Mamba-2", "ternary"): _tables / "quick_eval_bitmamba2_130M.json",
    }
    for (arch, quant), path in files.items():
        if path.exists():
            cells[(arch, quant)] = json.loads(path.read_text()).get("wikitext103_ppl")

    hf_path = _tables / "baseline_state-spaces_mamba2-130m_wikitext.json"
    hf_ppl = json.loads(hf_path.read_text()).get("wikitext103_ppl") if hf_path.exists() else None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    archs = ["Mamba-2", "Mamba-3"]
    width = 0.35
    x = np.arange(len(archs))

    ternary_ppls = [cells.get((a, "ternary")) or 0 for a in archs]
    fp_ppls = [cells.get((a, "FP")) or 0 for a in archs]

    b1 = ax.bar(x - width / 2, ternary_ppls, width, label="ternary",
                color=["#999", "#ff7f0e"], edgecolor="black")
    b2 = ax.bar(x + width / 2, fp_ppls, width, label="FP", hatch="//",
                color=["#bbb", "#ffaa55"], edgecolor="black")
    for bar, ppl in zip(b1, ternary_ppls):
        if ppl > 0:
            ax.annotate(f"{ppl:.1f}", (bar.get_x() + bar.get_width() / 2, ppl),
                        ha="center", va="bottom", fontsize=9)
    for bar, ppl in zip(b2, fp_ppls):
        if ppl > 0:
            ax.annotate(f"{ppl:.1f}", (bar.get_x() + bar.get_width() / 2, ppl),
                        ha="center", va="bottom", fontsize=9)
        else:
            ax.annotate("(not run)", (bar.get_x() + bar.get_width() / 2, 5),
                        ha="center", va="bottom", fontsize=8, style="italic")

    if hf_ppl:
        ax.axhline(hf_ppl, color="green", linestyle=":", alpha=0.7,
                   label=f"Mamba-2 130M FP @ 300B tokens (PPL {hf_ppl:.1f}, reference)")

    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("WikiText-103 PPL (lower = better)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig8_quant_cost_decomposition")


def fig9_int4_parity_control():
    p = _tables / "parity_int4_d512.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    by_config = {}
    for r in data:
        arch = r["arch"]
        mode = r.get("bitize_mode", False)
        if mode == "ternary" or mode is True: mode_str = "ternary"
        elif mode == "int4": mode_str = "int4"
        else: mode_str = "FP"
        by_config.setdefault(f"{arch}_{mode_str}", []).append(r["peak_acc_all"])

    archs = ["mamba3_siso", "mamba3_mimo"]
    arch_labels = ["SISO", "MIMO"]
    modes = ["FP", "int4", "ternary"]
    mode_labels = ["FP (16-bit)", "INT4 (4-bit)", "ternary (1.58-bit)"]
    colors = ["#bbb", "#7aa6c2", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(archs)); width = 0.25
    for i, (mode, mode_label, color) in enumerate(zip(modes, mode_labels, colors)):
        means, stds = [], []
        for arch in archs:
            vals = by_config.get(f"{arch}_{mode}", [0])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds,
                      capsize=4, label=mode_label, color=color, edgecolor="black")
        for bar, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.3f}\n±{s:.3f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01),
                        ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="random")
    ax.set_xticks(x); ax.set_xticklabels(arch_labels)
    ax.set_ylabel("Peak parity accuracy (3 seeds, μ±σ)")
    ax.set_ylim(0.4, 1.15)
    # Legend outside plot to avoid covering ternary bars
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig9_int4_parity_control")


def main():
    print(f"Generating clean (caption-free) figures into {_figs}")
    funcs = [
        fig1_baseline_throughput, fig2_parity_ternary_vs_fp,
        fig3_scaling_curves, fig4_needle_heatmap, fig5_parity_curves,
        fig6_parity_scaling_progression, fig7_bitmamba2_vs_bitmamba3,
        fig8_quant_cost_decomposition, fig9_int4_parity_control,
    ]
    for f in funcs:
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} failed: {e}")
    print(f"\nDone. {len(list(_figs.glob('*.png')))} PNG + {len(list(_figs.glob('*.jpg')))} JPG written.")


if __name__ == "__main__":
    main()
