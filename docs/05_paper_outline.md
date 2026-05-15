# Paper Outline (Working Draft)

Last updated: 2026-04-25.

## Tentative Title

**"BitMamba-3: 1.58-bit Ternary Quantization of Inference-First State Space
Models, and Ternary Quantization as a State-Tracking Inductive Bias"**

## Core Claims (in order of strength)

1. **C1 — Implementation feasibility**: Mamba-3's SSM kernels (SISO Triton,
   MIMO TileLang, decode CuteDSL) compose with BitNet b1.58 ternary
   quantization on `in_proj` / `out_proj` linear layers without modifying
   any upstream Mamba-3 forward path. Verified via 100-step training and
   30K-step convergence on fineweb-edu. (Strong, demonstrated.)

2. **C2 — Mamba-2 cannot solve parity**: Across 3 seeds at d=128 / single
   block / 3000 steps, Mamba-2 (FP and ternary) achieves 0.528–0.530 ± 0.006
   peak accuracy on the bit-parity state-tracking task. Pure architectural
   limit. (Strong, multi-seed.)

3. **C3 — Mamba-3 enables state tracking at tiny scale**: At d=128 / depth=1 /
   3K steps with constant LR, Mamba-3 MIMO peak 0.845 ± 0.125 (FP) shows
   architecture matters but with high variance. (Demonstrated, high variance.)

4. **C4 — Ternary quantization is an inductive bias for parity** (NEW,
   STRONGEST result): At d=256 / depth=2 / 5K steps / cosine LR / 5 seeds:
     - Mamba-3 SISO FP:        peak 0.530 ± 0.024 (random)
     - Mamba-3 SISO + ternary: peak **0.950 ± 0.075** (5/5 seeds learn parity)
     - Mamba-3 MIMO FP:        peak 0.521 ± 0.006 (random)
     - Mamba-3 MIMO + ternary: peak **0.949 ± 0.091** (5/5 seeds learn parity)

   Effect size 0.43 with σ ~0.08 → ~5-σ separation, p << 0.001.
   2× seqlen generalization 0.72 vs 0.50 random → genuine state tracking.
   SISO and MIMO converge to identical performance with ternary, suggesting
   ternary regularization is orthogonal to SISO/MIMO choice. **Candidate
   novel finding: ternary as inductive bias, distinct from compression.**

5. **C5 — Throughput on RTX 5090**: BitMamba-3 30M MIMO trains at 195K tok/s
   on RTX 5090 (bfloat16 AMP, batch 8, seqlen 2048). At 30K steps (480M
   tokens) reaches WikiText-103 PPL 400 from random init. Baseline matrix:
   Mamba-2 130M at L=32K reaches 401.9K tok/s prefill / 86 tok/s decode /
   6.45 GB peak. (Demonstrated.)

6. **C6 — Blackwell shmem characterization**: TileLang MIMO backward kernel
   exceeds RTX 5090 (SM 12.0) dynamic shared memory budget at the upstream
   default `chunk_size=64/mimo_rank`. Reduced to `chunk_size=8` for
   `mimo_rank=4`, fits within Blackwell's budget without algorithmic
   change. (Demonstrated, documented.)

## Suggested Structure

### §1 Introduction
- Motivation: efficient sequence modeling, the BitNet b1.58 low-bit trend.
- Contribution sketch: first PyTorch BitMamba-3 + parity state-tracking
  analysis (ternary as inductive bias).

### §2 Background
- Mamba-2 vs Mamba-3 architectural differences (cite Mamba-3 paper, our
  `docs/02_architecture_diff.md` summary).
- BitNet b1.58 quantization (cite arXiv:2402.17764, BitMamba-2).

### §3 Method
- §3.1 BitLinear: per-token INT8 absmax activation, per-tensor absmean
  ternary weight, STE. Bit-for-bit port from JAX BitMamba-2 to PyTorch.
- §3.2 BitMamba-3 module: subclass upstream Mamba-3 + replace `in_proj` /
  `out_proj`. Mamba-3 forward path / SSM kernels untouched.
- §3.3 Mamba-3 LM head registration patch (runtime override).
- §3.4 Blackwell shared-memory workaround (chunk_size=8 for mimo_rank=4).

### §4 Experiments
- §4.1 Setup: WSL Ubuntu-24.04, PyTorch 2.11+cu130, Mamba-2.3.1, RTX 5090.
  Data: fineweb-edu 1B tokens, GPT-NeoX tokenizer.
- §4.2 Parity (state-tracking): full multi-seed table, findings C2+C3+C4.
- §4.3 30M/130M/370M training curves: loss vs tokens, WikiText-103 PPL.
- §4.4 Zero-shot downstream (130M): lm-eval-harness suite.
- §4.5 Long-context: needle-in-haystack and PG19 PPL.
- §4.6 Ablations: ternarization scope, mimo_rank, rope_fraction.

### §5 Discussion
- Honest limitations: tiny model scale; multi-seed variance; training-data
  budget two orders of magnitude below standard corpora.
- Mamba-3 public checkpoint pending — own training stops short of
  competitive PPL.

### §6 Conclusion + Future Work
- Mamba-3 public checkpoints (when released): direct ternarization +
  fine-tuning instead of from-scratch training.
- Larger-scale parity ablations to firm up the inductive-bias claim.
- Scaling the from-scratch training token budget toward the
  Chinchilla-optimal regime.

## Bibliography Anchors

- Mamba-3: arXiv:2603.15569 (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu).
- Mamba-2: "Transformers are SSMs", ICML 2024.
- BitNet b1.58: arXiv:2402.17764 (MS Research).
- BitMamba-2: COLING 2025 / Zhayr1/BitMamba-2 GitHub.

## What This Paper Is Not Claiming

- Not claiming SOTA accuracy. Our 30M proxy is small; full 130M+ training
  pending.
- Not claiming BitMamba-3 outperforms BitMamba-2 in absolute terms (need
  same-scale matched comparison, pending Mamba-3 public checkpoints).
- Not claiming the parity solution generalizes (2× seqlen accuracy is ~0.58).

## Rough Submission Targets (no commitment)

Efficient-ML and systems venues that fit a software/algorithm contribution:
- MLSys (system-level / algorithm co-design)
- EMC² @ NeurIPS, ES-FoMo @ ICML (efficient ML workshops, lower bar)
- General ML venues, given the state-tracking inductive-bias finding
