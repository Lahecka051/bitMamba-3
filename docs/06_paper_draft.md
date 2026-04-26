# BitMamba-3 Paper Draft (Working Document)

Status: working draft, sections marked WIP.

---

## Abstract

State-space models (SSMs) like Mamba-3 promise efficient sequence modeling with
sub-quadratic complexity, while ternary quantization (BitNet b1.58) replaces
multipliers with conditional add/subtract for reduced compute and memory. We
combine the two: **BitMamba-3** ternarizes the projection layers of the
Mamba-3 block while leaving the SSM kernels (Triton SISO, TileLang MIMO, CuteDSL
decode) untouched. We make three contributions:

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

3. **Single-board FPGA prototype** (Zybo Z7-20): six RTL modules
   (`bit_mac`, `rope_engine`, `rmsnorm_int8`, `selective_scan_mimo`,
   `mimo_matmul`, `top_bitmamba3_block`) targeting the AXI4-HP DMA path
   from the prior `axi_dot_hp` and GGUF accelerators on the same board.
   `bit_mac` (128-lane ternary MAC) is bit-exact verified against PyTorch.
   The remaining FP16 modules use Xilinx FP IP placeholders pending
   integration.

We do not claim absolute throughput superiority over GPUs. Energy efficiency
projection and full FPGA bring-up are deferred to future work pending Mamba-3
public checkpoint release and physical board access.

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

### 2.3 Prior FPGA inference accelerators

BitNet-style ternary inference has been demonstrated on multiple FPGA
platforms in the past 18 months:

| Accelerator | FPGA | Bit width | Notable feature |
|---|---|---|---|
| TerEffic (arXiv:2502.16473) | Alveo U280 | 1.6-bit | First FPGA BitNet, 1.3 B fully on-chip |
| TeLLMe (arXiv:2504.16266) | Kria KV260 | 1.58-bit | Edge FPGA, prefill+decode, 25 tok/s @ 5 W |
| TeLLMe v2 (arXiv:2510.15926) | Kria KV260 | 1.58-bit | Table-lookup matmul, fused norm/quant |
| TOM (arXiv:2602.20662) | TBD | 1-bit | Ternary ROM, 3306 TPS on BitNet-2B |
| ternaryLLM (ETH ZH) | various | ternary | Open-source addition-based GEMM |
| Ternary-NanoCore | Artix-7 | ternary | NN accelerator on the same FPGA family as Zybo |

State-space-model FPGA accelerators have separately appeared:

| Accelerator | FPGA | Target | Notable feature |
|---|---|---|---|
| LightMamba (arXiv:2502.15260) | Versal VCK190 | Mamba | Rotation-assisted 4-bit quantization |
| FastMamba (arXiv:2505.18975) | various | Mamba2 | Hadamard-transform 8-bit, 68× CPU speedup |
| SpecMamba (arXiv:2509.19873) | FPGA | Mamba | First Mamba + speculative decoding FPGA |
| eMamba (arXiv:2508.10370) | edge FPGA | Mamba | Edge-focused framework |
| TVMamba (OpenReview) | FPGA | Visual Mamba | Ternary SSM blocks, *vision* not language |

To our knowledge, **no prior work has combined ternary quantization with
Mamba-3** on any platform; this paper takes the first step toward that
intersection.

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

### 3.5 RTL for Zybo Z7-20

We design six Verilog/SystemVerilog modules targeting the Zynq-7020
(`xc7z020clg400-1`) on the Digilent Zybo Z7-20 board. The board has 220
DSP48E1 slices, 53,200 LUTs, 140 BRAM blocks (4.9 Mb), and four PS-side AXI
HP DMA channels giving ≈3 GB/s aggregate to off-chip DDR3.

| Module | LUT (target) | DSP (target) | BRAM (target) | Status |
|---|---|---|---|---|
| `bit_mac` (128-lane ternary MAC) | ~8K | 0 | 4 | functional, bit-exact verified |
| `rope_engine` (data-dependent rotation) | ~2K | 8 | 2 | skeleton + LUT |
| `rmsnorm_int8` (RMSNorm + INT8 quantizer) | ~3K | 20 | 0 | skeleton |
| `selective_scan_mimo` (Mamba-3 SSM state update) | ~5K | 40 | 20 | skeleton |
| `mimo_matmul` (mimo_x/z/o einsum) | ~3K | 32 | 0 | skeleton |
| `top_bitmamba3_block` (AXI wrapper) | ~5K | 0 | 4 | structural |
| **Total** | **~26K (49%)** | **~100 (45%)** | **~30 (21%)** | comfortable fit |

The bit-exact-verified `bit_mac` realises the BitLinear weight×activation
inner product as 128 lanes of conditional add/sub (no multiplications),
producing INT24 partial sums in three pipeline stages (decode → group reduce →
final reduce). 100 random vectors generated by the PyTorch reference all
match the RTL output exactly.

The remaining FP16 modules (rope, rmsnorm, selective_scan, mimo_matmul)
declare their port-level interfaces and high-level pipeline structure. Their
internal floating-point datapath is intended to be filled in via Xilinx
Floating-Point Operator IP instances at integration time. This is a deliberate
choice: re-implementing FP16 arithmetic from scratch would consume disproportionate
engineering effort relative to the algorithmic novelty of the project.

#### Reuse of existing infrastructure

Our prior FPGA project on the same board (G:\Xilinx\) provides field-tested
modules for AXI HP DMA (4-channel interleaved, 3 GB/s effective), AXI Lite
CSR with start/done/perf-counter registers, and a softmax LUT. The
`top_bitmamba3_block` integrates these as plug-in submodules rather than
re-implementing them. This reduces the new-RTL surface area by approximately
1,000 lines and inherits the bring-up history (BOOT.BIN, PetaLinux device
tree reservations, SDK image) of the prior `axi_dot_hp` and GGUF Q4_0 / Q8_0
projects on the same physical board.

#### PetaLinux host driver

A C-language host program (`petalinux/test_programs/test_bitmamba3.c`)
mmaps the AXI Lite CSR at `0x43C0_0000` and a 256 MB DDR scratch region
reserved at `0x3000_0000` via the device tree. The host loads
ternary-packed weights and INT8 activations into DDR, programs the base
addresses into the CSR, kicks off the FSM, and polls the status register
until done. Performance counters (cycle count, AXI burst counts) are
read back. This mirrors the structure of the existing `test_dot_hp`
program from the same board's prior axi-DMA accelerator project.

Physical Zybo bring-up and end-to-end bit-exact verification against the
PyTorch reference are deferred to follow-on work; the present paper
documents the RTL artifacts and the integration plan but does not include
on-board measurements.

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

Figure 2 (results/figures/fig2_parity_ternary_vs_fp.pdf):

| Config                     | Peak (5 seeds) | Final          | 2× seqlen      |
|---------------------------|----------------|----------------|----------------|
| Mamba-3 SISO FP            | 0.530 ± 0.024  | 0.514 ± 0.026  | 0.506 ± 0.012  |
| **Mamba-3 SISO + ternary** | **0.950 ± 0.075** | 0.821 ± 0.158 | 0.717 ± 0.149 |
| Mamba-3 MIMO FP            | 0.521 ± 0.006  | 0.509 ± 0.008  | 0.503 ± 0.006  |
| **Mamba-3 MIMO + ternary** | **0.949 ± 0.091** | 0.809 ± 0.190 | 0.715 ± 0.165 |

[WIP — narrative about the 0.43 separation, 2× seqlen generalization,
inductive-bias interpretation]

### 4.5 Zero-shot downstream (130M)

| Task           | Acc   | Random | Mamba-2 130M (300B tok) |
|---------------|-------|--------|-------------------------|
| ARC-Easy       | 0.410 | 0.25   | ~0.50                   |
| HellaSwag      | 0.390 (norm) | 0.25 | ~0.38              |
| PIQA           | 0.570 | 0.50   | ~0.62                   |
| LAMBADA-OpenAI | 0.100 | 0.0    | ~0.30                   |

[WIP — discuss reasonable proximity to published baselines for ARC/HellaSwag,
LAMBADA gap explained by limited training tokens]

### 4.6 Needle-in-haystack (130M)

Figure 4. Average log-prob of magic-number recall at L ∈ {512, 2K, 4K} and
needle depth ∈ {0, 50, 100}%. Recent-context recall is strong (-4.67 at
L=2K depth=100%); early-context recall fades at long L.

---

## §5 Hardware

The hardware contribution of this paper is the RTL design and integration
plan documented in §3.5, with one bit-exact-verified module (`bit_mac`)
and five structural modules awaiting FP16 IP integration. Concrete
Vivado synthesis, timing closure, PetaLinux bring-up, and on-board
throughput / energy measurements are work-in-progress. We summarise the
status here for completeness.

### 5.1 Module-level verification

`bit_mac` was verified by Verilator against a Python golden reference
generated by `sim/verilator/gen_vectors_bit_mac.py`. All 100 random
INT8 × ternary dot-product vectors at 128-lane width matched the RTL
output exactly (`sim/verilator/verify_bit_mac_python.py` reports
`ok=100, bad=0`). The same pattern is set up for `rope_engine`
(`sim/verilator/gen_vectors_rope.py` produces 1000 vectors, 983 pass
within FP16 ULP tolerance pending the rotation datapath substitution).

### 5.2 Block-design integration (Vivado, pending)

`rtl/bd/create_bitmamba3_bd.tcl` constructs a Vivado IPI block design that
instantiates `top_bitmamba3_block` as a Module Reference, wires its
AXI-Lite slave port to PS GP0 and reserves AXI HP0 for DDR access. The TCL
also pulls in the field-tested `axi_hp_dma.sv` and `axi_lite_csr.sv` from
the prior project on the same board. Synthesis and timing closure on
`xc7z020clg400-1` at 100 MHz are pending Vivado access.

### 5.3 PetaLinux + on-board measurement (pending)

The PetaLinux project structure and BOOT.BIN flow follow the pattern of
the prior `axi_dot_hp` project on this board. End-to-end bit-exact
comparison of the on-board accelerator against the PyTorch reference
(forward pass on a single Mamba-3 block at d_model=384, n_layer=8,
i.e. our 30M training preset) is the immediate next milestone after
synthesis. Throughput and energy measurements (Kill-A-Watt at the
barrel-jack PSU) will follow.

### 5.4 What this paper claims about hardware

We claim:
- A complete RTL plan for a single-board BitMamba-3 accelerator on a
  commodity Zybo Z7-20.
- One module (`bit_mac`) bit-exact verified.
- A clear path from the documented modules to a working bit-stream that
  reuses ~1000 lines of validated AXI plumbing from prior projects on
  the same board.

We do **not** claim:
- Synthesised LUT/DSP/BRAM utilisation numbers (resources are estimates).
- Timing closure at 100 MHz.
- On-board throughput numbers.
- Energy efficiency vs RTX 5090.

These are honestly future work. The §3.5 module table provides the
target resource budget and §6 explicitly flags this gap.

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

- Our FPGA RTL is at skeleton level for FP16 modules; only the ternary MAC
  is bit-exact verified. Throughput and energy numbers on physical Zybo
  hardware are future work.

---

## §7 Future Work

- Mamba-3 public checkpoints (when released): direct ternarization +
  fine-tuning rather than from-scratch training.
- Multi-board Zybo cluster for distributed inference.
- ASIC area / energy projection from synthesized RTL.
- Larger-scale parity ablations (d=512, depth=4, 10+ seeds) to firm up the
  inductive-bias claim.
- Energy efficiency measurement (Zybo wall-plug vs RTX 5090) once FPGA
  bring-up completes.

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
