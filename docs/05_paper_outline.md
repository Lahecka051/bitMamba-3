# Paper Outline (Working Draft)

Last updated: 2026-04-25.

## Tentative Title

**"BitMamba-3: 1.58-bit Ternary Quantization of Inference-First State Space
Models for Energy-Efficient Edge Deployment"**

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

7. **C7 — Single-board FPGA implementation feasibility**: 6 RTL modules
   (`bit_mac`, `rope_engine`, `rmsnorm_int8`, `selective_scan_mimo`,
   `mimo_matmul`, `top_bitmamba3_block`) targeting Zybo Z7-20 with AXI4-HP
   DMA interface. `bit_mac` is functionally complete with 3-stage pipeline,
   bit-exact verified against PyTorch reference (100 vectors). RoPE LUT
   (1024-entry FP16 sin/cos) generated. RMSNorm + selective scan use FP16
   IP placeholders pending integration. (Partial, RTL-level.)

8. **C8 — Energy efficiency vs RTX 5090** (target, not yet measured): 8×
   Zybo aggregate ≈ 40W TDP vs RTX 5090 450W. Tokens/sec/W projection
   pending Zybo bring-up. (Future work.)

## Suggested Structure

### §1 Introduction
- Motivation: edge LLM inference, energy efficiency, the BitNet b1.58 trend.
- Contribution sketch: first PyTorch BitMamba-3 + first Zybo RTL design +
  parity state-tracking analysis.

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
- §3.5 RTL design for Zybo Z7-20.

### §4 Experiments — Software
- §4.1 Setup: WSL Ubuntu-24.04, PyTorch 2.11+cu130, Mamba-2.3.1, RTX 5090.
  Data: fineweb-edu 1B tokens, GPT-NeoX tokenizer.
- §4.2 Parity (state-tracking): full 18-run multi-seed table, finding 1+2+3+4.
- §4.3 30M training curves: loss vs tokens, WikiText-103 PPL.
- §4.4 130M training (planned).
- §4.5 Ablations: ternarization scope, mimo_rank, rope_fraction.

### §5 Experiments — Hardware
- §5.1 RTL micro-architecture diagrams.
- §5.2 Verilator bit-exact verification results.
- §5.3 Vivado synthesis + Zybo Z7-20 utilization.
- §5.4 PetaLinux bring-up + end-to-end FPGA inference.
- §5.5 Throughput + energy measurements.

### §6 Discussion
- Honest limitations: tiny model scale; multi-seed variance; FPGA RTL
  partly placeholder.
- Mamba-3 public checkpoint pending — own training stops short of
  competitive PPL.

### §7 Conclusion + Future Work
- Multi-board scaling (8× Zybo cluster, deferred).
- Bigger model scales (1B+).
- ASIC area/energy projection.

## Bibliography Anchors

- Mamba-3: arXiv:2603.15569 (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu).
- Mamba-2: "Transformers are SSMs", ICML 2024.
- BitNet b1.58: arXiv:2402.17764 (MS Research).
- BitMamba-2: COLING 2025 / Zhayr1/BitMamba-2 GitHub.
- TerEffic: arXiv:2502.16473 (FPGA ternary LLM, Alveo U280).
- TeLLMe / TeLLMe v2: arXiv:2504.16266 / 2510.15926 (KV260 / edge FPGA).
- LightMamba: arXiv:2502.15260 (FPGA Mamba 4-bit, Versal).
- FastMamba: arXiv:2505.18975 (FPGA Mamba2 8-bit).

## What This Paper Is Not Claiming

- Not claiming SOTA accuracy. Our 30M proxy is small; full 130M+ training
  pending.
- Not claiming BitMamba-3 outperforms BitMamba-2 in absolute terms (need
  same-scale matched comparison, pending Mamba-3 public checkpoints).
- Not claiming parity solution generalizes (2× seqlen accuracy is ~0.58).
- Not claiming the FPGA is faster than RTX 5090 (it isn't and won't be).
  Energy efficiency angle only.

## Rough Submission Targets (no commitment)

Algorithm + HW co-design venues that fit:
- FCCM, FPL, FPGA (HW-focused, FPGA implementation valued)
- MLSys (system-level co-design)
- EMC² @ NeurIPS, ES-FoMo @ ICML (efficient ML workshops, lower bar)
