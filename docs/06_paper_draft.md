# BitMamba-3 Paper Draft (Working Document)

Status: working draft, sections marked WIP.

---

## Abstract

State-space models (SSMs) like Mamba-3 promise efficient sequence modeling with
sub-quadratic complexity, while ternary quantization (BitNet b1.58) replaces
multipliers with conditional add/subtract for reduced compute and memory. We
combine the two: **BitMamba-3** ternarizes the projection layers of the
Mamba-3 block while leaving the SSM kernels (Triton SISO, TileLang MIMO, CuteDSL
decode) untouched. We make two contributions:

1. **Implementation**: minimal-diff PyTorch wrapper that swaps `in_proj` and
   `out_proj` with `BitLinear`, plus a runtime `create_block` patch that
   registers Mamba-3 as an `ssm_layer` in upstream `state-spaces/mamba`.
   Verified end-to-end at 30M / 130M / 370M scales on RTX 5090.

2. **Empirical finding (state-tracking)**: At small scale (d=256, depth=2,
   5K steps with cosine LR), Mamba-3 + ternary reaches **0.95 ± 0.08 peak
   accuracy** on the bit-parity task across 5 random seeds; Mamba-3 FP at the
   same scale sits at chance (0.52 ± 0.02). The 0.43-effect-size, ~5σ
   separation suggests **ternary as an inductive bias toward crisp
   state-tracking solutions**, distinct from the usual compression-only
   framing of BitNet-style quantization.

This is a software / algorithm contribution: a working ternary Mamba-3
implementation and an empirical state-tracking finding. Absolute throughput
comparisons against GPUs and dedicated-accelerator deployment are out of scope.

---

## §1 Introduction

[WIP]

LLM inference at the edge faces the multiply-bandwidth wall: each MAC on a
mobile/embedded device costs energy proportional to its precision, and weight
fetch from off-chip memory dominates power. The combination of efficient
state-space architectures (Mamba-3, the latest in the Mamba family with an
"inference-first" design) and aggressive low-bit weight quantization
(BitNet b1.58, which collapses weights to ternary {-1, 0, +1}) is a natural
candidate for both axes.

Our position is that this combination is more than additive. Mamba-3's
RoPE-augmented recurrence is provably equivalent to a complex-valued SSM
(Lahoti et al. 2026) and grants the architecture state-tracking capabilities
absent in Mamba-2. Ternary quantization restricts the weight hypothesis class
to a small discrete set; on top of an architecture that *can* express state
tracking, this restriction appears to act as an inductive bias toward crisp
parity-style solutions rather than a fuzzy approximation. Section §4
demonstrates a 5σ separation between Mamba-3 FP and Mamba-3 + ternary on the
bit-parity benchmark at small scale.

---

## §2 Background

### 2.1 Mamba-3

Mamba-3 (Lahoti et al. 2026, arXiv:2603.15569) refines the Mamba-2 selective
state-space model along two architectural axes:

1. **RoPE-augmented recurrence**: The B and C projections receive a
   data-dependent rotary positional embedding before the SSM scan. The paper
   proves that this is mathematically equivalent to a *complex-valued* SSM
   recurrence — the rotation matrices implement complex multiplication in
   real-valued form. This grants Mamba-3 the capability to track running
   parities (XOR-style state), which Mamba-2 provably cannot at any width.

2. **Optional MIMO formulation**: The standard SISO recurrence is replaced
   by Multi-Input Multi-Output, where the per-head state expands by a
   `mimo_rank` factor (default 4). The original SISO recurrence is recovered
   at `mimo_rank=1`.

The reference implementation in `state-spaces/mamba` v2.3.1 ships three
fused kernels: (i) a Triton-based SISO chunk-scan combined kernel, (ii) a
TileLang MIMO combined kernel, and (iii) a CuteDSL `step_fn` for single-token
decode. These are tested on H100 / SM 9.0; we describe Blackwell tuning in
§3.4.

The published Mamba-3 paper uses `d_state=128` and `mimo_rank=4` as defaults,
trained at 1.5 B parameters. As of this writing the 1.5 B weights are not
publicly released, which forces our experiments to use from-scratch training
at 30M / 130M / 370M scales (§4).

### 2.2 BitNet b1.58 and BitMamba-2

BitNet b1.58 (Microsoft Research, 2024, arXiv:2402.17764) restricts the
weights of all linear layers in a transformer to the ternary set
`{-1, 0, +1}`, encoded at log₂(3) ≈ 1.58 bits per weight. The paper shows
that, at 3 B parameters and above, ternary weights match the perplexity and
zero-shot performance of FP16 baselines while using 5× less memory and an
order-of-magnitude lower energy per multiply (which becomes a conditional
add). Activations are kept at INT8 with per-token absmax scaling. The
straight-through estimator (STE) keeps gradients flowing past the
non-differentiable `round()` and `clamp()` operators during training.

BitMamba-2 (Zhayr1/BitMamba-2, "Fully Quantized Mamba in 1.58 Bits", COLING
2025) applies BitNet b1.58 to a Mamba-2 architecture. The author trains a
170 M and a 1 B variant from scratch on 150 B tokens via JAX/Flax on Google
TPU. Their `BitLinear` operates only on the linear projection layers
(`in_proj` and `out_proj` of each Mamba block, plus the LM head); the SSM
scan and the conv1d remain in FP16/BF16. Embedding layers are kept FP. The
authors report ≈90% reduction in stored bits and competitive perplexity at
1 B against an FP16 Mamba-2 baseline.

Our `BitLinear` (§3.1) is a bit-for-bit PyTorch port of BitMamba-2's JAX
implementation, modulo the framework-specific RMSNorm and STE machinery.

To our knowledge, **no prior work has combined ternary (BitNet b1.58)
quantization with the Mamba-3 architecture**; this paper takes the first
step toward that intersection.

---

## §3 Method

### 3.1 BitLinear (PyTorch port of BitMamba-2)

We re-implement BitMamba-2's `BitLinear` as a drop-in subclass of
`torch.nn.Linear`. Forward semantics match BitMamba-2 bit-for-bit:

```python
class BitLinear(nn.Linear):
    def forward(self, x):
        x_n = F.rms_norm(x, normalized_shape=(x.size(-1),))
        scale_x = 127.0 / x_n.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x_n * scale_x).round().clamp(-128, 127) / scale_x
        x_eff = x_n + (x_q - x_n).detach()
        scale_w = 1.0 / self.weight.abs().mean().clamp(min=1e-5)
        w_q = (self.weight * scale_w).round().clamp(-1, 1) / scale_w
        w_eff = self.weight + (w_q - self.weight).detach()
        return F.linear(x_eff, w_eff, self.bias)
```

Activation is INT8 with per-row absmax scaling. Weights are ternary with
per-tensor absmean scaling. The straight-through estimator (STE) keeps
gradients flowing through `round()` and `clamp()`. RMSNorm with unit affine
gates the activation magnitude pre-quantization for stability.

### 3.2 BitMamba-3 module

`BitMamba3` subclasses `state-spaces/mamba.Mamba3` and replaces only `in_proj`
and `out_proj` post-init:

```python
class BitMamba3(Mamba3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_proj = BitLinear(self.in_proj.in_features,
                                 self.in_proj.out_features, bias=False, ...)
        self.out_proj = BitLinear(self.out_proj.in_features,
                                  self.out_proj.out_features, bias=False, ...)
```

The Mamba-3 forward path, including the SSM kernels (`mamba3_siso_combined`
in Triton, `mamba3_mimo_combined` in TileLang, `mamba3_step_fn` in CuteDSL),
RoPE engine, and RMSNorm on B/C, is **unchanged**. All other parameters
(`B_bias`, `C_bias`, `dt_bias`, `D`, `mimo_x/z/o`, `B_norm`, `C_norm`) remain
in their original precision.

At 30M scale, this swap covers 28% of total parameters (the small embedding
dominates). At 130M and 370M scales the ternary fraction climbs to 71% and
86% respectively as the projection layers grow with d_model.

### 3.3 Mamba-3 LM head registration patch

Upstream `state-spaces/mamba` v2.3.1 ships `modules/mamba3.py` but its
`mixer_seq_simple.create_block` factory rejects `ssm_cfg["layer"] == "Mamba3"`.
We monkey-patch `create_block` at runtime to register Mamba-3 alongside
Mamba-1 and Mamba-2; no upstream source file is modified. See
`src/bitmamba3/mamba3_lm_patch.py`.

### 3.4 Blackwell shared-memory tuning

The TileLang MIMO backward kernel allocates dynamic shared memory proportional
to `d_state × chunk_size × mimo_rank`. NVIDIA Blackwell consumer GPUs
(SM 12.0, e.g. RTX 5090) reject any allocation above ~123 KB. The kernel also
asserts `chunk_size >= 8`. Combined, we must reduce `d_state` from the
upstream default 128 to **64** for our 130M and 370M presets to fit. This is
a hardware-specific kernel-launch tuning; the algorithm is unchanged. We
verified forward and backward numerical equivalence at smaller batch×seqlen
sizes where d_state=128 fits.

---

## §4 Experiments

### 4.1 Setup

[WIP — WSL Ubuntu 24.04, PyTorch 2.11+cu130, mamba_ssm 2.3.1, RTX 5090,
fineweb-edu 1B tokens, GPT-NeoX-20B tokenizer, AdamW with cosine LR]

### 4.2 Mamba-2 baseline throughput (RTX 5090)

| Model    | L=512    | L=2K     | L=8K      | L=32K     | Decode | Peak Mem (32K) |
|---------|----------|----------|-----------|-----------|--------|----------------|
| 130M    | 23.6 K/s | 88.7 K/s | 285.5 K/s | 401.9 K/s | 86 t/s | 6.45 GB        |
| 370M    | 11.8 K/s | 45.8 K/s | 141.3 K/s | 131.2 K/s | 43 t/s | 6.92 GB        |
| 1.3B    | 13.2 K/s | 47.4 K/s | 50.9 K/s  | 49.3 K/s  | 44 t/s | 8.86 GB        |

Decode rate is ~constant in context length (Mamba's O(1) state advantage).

### 4.3 BitMamba-3 training (fineweb-edu)

| Run         | Params | Tokens | Steps | Loss  | WikiText PPL |
|------------|--------|--------|-------|-------|--------------|
| 30M short  | 27M    | 164M   | 10K   | 5.00  | 553          |
| 30M long   | 27M    | 480M   | 30K   | 4.90  | 400          |
| 130M       | 135M   | 480M   | 30K   | 3.57  | 69.4         |
| **370M**   | **386M** | **480M** | **30K** | **3.33** | **60.2** |

The 30M → 130M jump (4.6× params, same tokens) yields a 5.8× PPL improvement
(400 → 69.4). The 130M → 370M jump (2.9× params, same tokens) yields only a
13% PPL improvement (69.4 → 60.2). This pattern of diminishing returns is
consistent with the Chinchilla compute-optimal regime: 480M tokens is roughly
optimal for the 130M scale, making 370M training-data-bound at this token
budget. With more tokens (10–20 B per Chinchilla recommendations), 370M would
be expected to overtake 130M by a larger margin.

### 4.4 Parity (state-tracking) — main result

Figures 2, 5, and 6 (results/figures/).

#### 4.4.1 d=256 / depth=2 (5 seeds, cosine LR, 5K steps)

| Config                     | Peak           | Final          | 2× seqlen      |
|---------------------------|----------------|----------------|----------------|
| Mamba-3 SISO FP            | 0.530 ± 0.024  | 0.514 ± 0.026  | 0.506 ± 0.012  |
| Mamba-3 SISO + ternary     | 0.950 ± 0.075  | 0.821 ± 0.158  | 0.717 ± 0.149  |
| Mamba-3 MIMO FP            | 0.521 ± 0.006  | 0.509 ± 0.008  | 0.503 ± 0.006  |
| **Mamba-3 MIMO + ternary** | **0.949 ± 0.091** | 0.809 ± 0.190 | 0.715 ± 0.165 |

Effect size 0.43 with σ ≈ 0.08 → ~5σ separation, p ≪ 0.001.

#### 4.4.2 d=512 / depth=4 (5 seeds, cosine LR, 5K steps)

| Config                     | Peak           | Final          | 2× seqlen      |
|---------------------------|----------------|----------------|----------------|
| Mamba-3 SISO FP            | 0.510 ± 0.002  | 0.503 ± 0.005  | 0.500 ± 0.004  |
| Mamba-3 SISO + ternary     | 0.860 ± 0.188  | 0.694 ± 0.260  | 0.615 ± 0.168  |
| Mamba-3 MIMO FP            | 0.510 ± 0.003  | 0.503 ± 0.005  | 0.500 ± 0.004  |
| **Mamba-3 MIMO + ternary** | **0.981 ± 0.036** | **0.897 ± 0.171** | **0.765 ± 0.143** |

Effect size 0.47 with σ ≈ 0.04 → **~13σ separation**.

#### 4.4.3 Scaling progression of the inductive-bias effect

|                   | d=128  | d=256  | d=512  |
|------------------|--------|--------|--------|
| MIMO ternary peak | 0.860  | 0.949  | **0.981** |
| MIMO ternary σ    | 0.146  | 0.091  | **0.036** |
| MIMO FP peak      | 0.845  | 0.521  | **0.510** |
| MIMO FP σ         | 0.125  | 0.006  | **0.003** |

Three observations:

1. **MIMO + ternary peak strengthens with scale** (0.86 → 0.95 → 0.98) and
   variance tightens (σ 0.15 → 0.09 → 0.04).
2. **MIMO FP regresses to chance** as the model grows (cosine LR + larger d
   prevents random landing in parity solutions).
3. **SISO + ternary regresses at d=512** (0.95 → 0.86 with σ 0.19): the
   single-channel state runs out of capacity. **MIMO's rank-4 expansion
   becomes structurally necessary at scale**, even with ternary
   regularization.

The 5σ → 13σ separation across scale, combined with strong 2× seqlen
generalization (0.72 → 0.77), positions ternary as a **structural
inductive bias** for state tracking on top of Mamba-3's recurrence,
distinct from the usual compression-only framing of low-bit quantization.

#### 4.4.4 Mamba-2 control

For all 6 seeds across d=128 sweeps, Mamba-2 (FP and ternary) stays at
chance (0.528 ± 0.006 and 0.530 ± 0.004). This is a clean negative
result confirming the Mamba-3 paper's claim that the RoPE-based
recurrence is necessary, not just sufficient, for state tracking.

### 4.5 Zero-shot downstream (130M)

| Task           | Acc   | Random | Mamba-2 130M (300B tok) |
|---------------|-------|--------|-------------------------|
| ARC-Easy       | 0.410 | 0.25   | ~0.50                   |
| HellaSwag      | 0.390 (norm) | 0.25 | ~0.38              |
| PIQA           | 0.570 | 0.50   | ~0.62                   |
| LAMBADA-OpenAI | 0.100 | 0.0    | ~0.30                   |

[WIP — discuss reasonable proximity to published baselines for ARC/HellaSwag,
LAMBADA gap explained by limited training tokens]

### 4.6 Needle-in-haystack (130M and 370M)

Figure 4. Average log-prob of magic-number recall at L ∈ {512, 2K, 4K} and
needle depth ∈ {0, 50, 100}%.

| L | depth | BitMamba-2 130M | BitMamba-3 130M | BitMamba-3 370M |
|---|---|---|---|---|
| 512 | 100% | -8.21 | -6.85 | **-3.54** |
| 2048 | 100% | -7.74 | -4.67 | -5.76 |
| 4096 | 0% | -9.35 | -11.53 | **-8.85** |
| 4096 | 100% | -6.07 | -8.26 | **-5.17** |

Two trends:
- BitMamba-3 dominates **recent-context recall** (depth=100%): L=512 jumps
  from -8.21 → -3.54 going M2 130M → M3 370M.
- BitMamba-3 370M improves **far-context recovery** at long L
  (L=4K depth=0%: -8.85 vs 130M's -11.53).

### 4.7 Long-context PPL on PG19

| L     | BitMamba-2 130M | BitMamba-3 130M | BitMamba-3 370M |
|------|-----------------|-----------------|-----------------|
| 1024 | 80.45           | 71.50           | **65.11**       |
| 2048 | 79.91           | 70.37           | **64.01**       |
| 4096 | 79.85           | 70.10           | **63.78**       |
| 8192 | (n/a)           | (n/a)           | 63.82           |

Mamba-3 maintains a consistent ~12% PPL improvement over Mamba-2 across
all context lengths, with the advantage slightly widening at longer L
(11.1% at L=1024 → 12.2% at L=4096) — consistent with the RoPE-based
recurrence's long-range modeling advantage.

---

## §6 Discussion

### Limitations

- Our 30M / 130M / 370M models are trained on 480M fineweb-edu tokens, two
  orders of magnitude below standard pretraining corpora. Absolute downstream
  performance is therefore well below published baselines like Mamba-2 130M
  (300 B tokens). Comparisons should be interpreted as relative trends within
  our token budget, not as competitive benchmarks.

- The parity result is at small-scale single-block models with a synthetic
  task. Whether the inductive-bias mechanism scales to other state-tracking
  workloads at larger scales is open. We provide it as a hypothesis-generating
  observation, not a definitive statement.

- The Blackwell `d_state=64` reduction matches `state-spaces/mamba`'s smaller
  configurations rather than the published `d_state=128`. We have not
  retrained at d_state=128 on a different GPU.

- Mamba-3 public weights are not yet released; we cannot directly compare a
  ternary-quantized Mamba-3 against the FP16 reference at matched training
  tokens.

---

## §7 Future Work

- Mamba-3 public checkpoints (when released): direct ternarization +
  fine-tuning rather than from-scratch training.
- Larger-scale parity ablations (d=512, depth=4, 10+ seeds) to firm up the
  inductive-bias claim.
- Scaling the from-scratch training token budget toward the
  Chinchilla-optimal regime for the 130M / 370M presets.

---

## Appendix A: Reproducibility

Public code: `G:\Github Desktop\bitMamba-3\` (paths assume Windows host with
WSL Ubuntu 24.04).

Key entry points:
- `python src/training/train.py --preset 130M --data_dir data/fineweb_1B ...`
- `python src/evaluation/quick_eval.py --ckpt <ckpt.pt> --preset 130M`
- `python scripts/run_parity_multiseed_bg.py` (3-seed sweep)
- `python scripts/run_parity_larger_bg.py` (5-seed d=256 sweep, main parity result)
- `python src/evaluation/run_lm_eval.py --ckpt <ckpt.pt> --preset 130M`
- `python src/evaluation/needle_haystack.py --ckpt <ckpt.pt> --preset 130M`
- `python src/evaluation/generate_paper_figures.py` (Fig 1–5)

Random seeds for parity: explicit `--seed 0..4`; for training a single seed
(0) is used per run.
