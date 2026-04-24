# Experimental Matrix

Scope: **single Zybo Z7-20** FPGA board. All multi-board experiments deferred.

## Track A: Algorithm (RTX 5090, PyTorch)

### A1. Baseline inference benchmarks (no training)

Reproduce published baselines for comparison numbers.

| Baseline | Platform | Metrics |
|---|---|---|
| Mamba-3 FP16 (public ckpt) | RTX 5090 | PPL, tok/s, tok/s/W at L=512..131K |
| Mamba-2 FP16 (public ckpt) | RTX 5090 | Same, for Mamba-3 gain quantification |
| BitMamba-2 C++ AVX2 | Intel 285K CPU | tok/s, tok/s/W |
| Transformer equiv. (TinyLlama-1.1B) | RTX 5090 | Long-context scaling comparison |

### A2. BitMamba-3 training (from scratch)

| Exp | Model | Tokens | Data | Purpose |
|---|---|---|---|---|
| A2-a | 30M proxy | 1B | SlimPajama subset | Convergence verification |
| A2-b | 130M main | 10B | SlimPajama | Primary result |
| A2-c | 370M scaling | 20B | SlimPajama | Scaling trend |

### A3. Ablation sweep (on 130M or 30M as scope permits)

| Axis | Values |
|---|---|
| Ternarization scope | weight-only / +embedding / +projection / +all |
| RoPE position | pre-projection / post-projection / fused |
| MIMO group size | 1 (SISO) / 2 / 4 / 8 |
| State size | full Mamba-2 / half (Mamba-3 default) |
| Activation bits | INT4 / INT8 / FP8 |
| STE variant | standard / clipped / learned-scale |

### A4. Evaluation suite

- **Perplexity**: WikiText-103, C4, The Pile (slice)
- **Zero-shot (lm-eval-harness)**: LAMBADA, HellaSwag, ARC-e/c, Winogrande, PIQA, BoolQ, OpenBookQA
- **State tracking**: Parity task (Mamba-3 paper benchmark — key claim)
- **Long context**: Needle-in-Haystack at L=4K / 32K / 128K
- **Generation quality** (370M only, if time): MMLU, TriviaQA

## Track B: RTL (Zybo Z7-20 single board)

### B1. Module RTL with Verilator bit-exact testbench

| Module | I/O | Verification vectors |
|---|---|---|
| Ternary MAC (conditional add/sub/skip) | 128-wide int8 × ternary → int24 | 10,000 random |
| MIMO matmul array | (d_mimo × N) × (N × d_mimo) | 1,000 random |
| RoPE engine | (x, θ) → rotated pair, sin/cos LUT | 10,000 random |
| Selective scan pipeline | sequential state update | 1,000 sequence |
| Causal conv1d | width-4 window | 10,000 random |
| Activation INT8 quantizer | FP → INT8 + scale | 10,000 random |

Bit-exact reference: PyTorch BitMamba-3 forward pass.

### B2. Single-layer integration

- Full Mamba-3 block RTL
- Vivado synthesis on Zybo Z7-20 (xc7z020clg400-1)
- Target: WNS > 0 @ 100 MHz
- Report: LUT / DSP / BRAM utilization
- Goal: fit 1~3 Mamba-3 layers per Zybo

### B3. End-to-end single-board inference

- PetaLinux BOOT.BIN
- N layer stacked (N = fit max)
- Bit-exact vs PyTorch reference on token-by-token hidden states

## Track C: System Measurements (single Zybo + RTX 5090 baseline)

### C1. Throughput matrix

| L (context) | Zybo tok/s | RTX 5090 tok/s (Mamba-3 FP16) | CPU tok/s (BitMamba-2) |
|---|---|---|---|
| 512 | ? | ? | ? |
| 2K | ? | ? | ? |
| 8K | ? | ? | ? |
| 32K | ? | ? | ? |
| 128K | ? | ? | ? |

### C2. Energy measurements

- Zybo: wall-plug power meter (Kill-A-Watt) or INA219 on 5V rail
- RTX 5090: `nvidia-smi --query-gpu=power.draw` + system-wide meter
- CPU (BitMamba-2): HWInfo or wall meter
- Metric: **tokens/sec/W** at each context length

### C3. Latency breakdown

Per-token decomposition:
- Compute (layer × N) on Zybo
- DDR3 access (weight read)
- Activation propagation
- Host↔Zybo comm (if offloaded)

Tools: Vivado ILA on-chip, external logic analyzer, Linux `perf`.

## Data Collection Checklist

### Tables for paper
- [ ] Table 1: Perplexity (Mamba-3 FP16, Mamba-2 FP16, BitMamba-2, BitMamba-3 at 30M/130M/370M)
- [ ] Table 2: Downstream zero-shot accuracy
- [ ] Table 3: Throughput across platforms × context
- [ ] Table 4: Energy efficiency (tok/s/W)
- [ ] Table 5: FPGA resource utilization
- [ ] Table 6: Ablation study results

### Figures
- [ ] Fig 1: BitMamba-3 architecture diagram
- [ ] Fig 2: Ternary MIMO MAC micro-architecture
- [ ] Fig 3: RTL pipeline timing diagram
- [ ] Fig 4: Training loss curves (ablations)
- [ ] Fig 5: Energy comparison bar chart
- [ ] Fig 6: Long-context needle-in-haystack heatmap
- [ ] Fig 7: Perplexity vs model size (scaling law)

## Deferred / Future Work

- Multi-board pipeline parallelism (8+ Zybos)
- 1B+ model training (requires multi-GPU or cloud)
- ASIC projection / synthesis
- Mobile/edge deployment scenarios
