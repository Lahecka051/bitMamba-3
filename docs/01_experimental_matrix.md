# Experimental Matrix

Scope: software / algorithm experiments for BitMamba-3 (1.58-bit ternary
Mamba-3). All experiments run on RTX 5090 + PyTorch (with an Intel 285K CPU
baseline for BitMamba-2).

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
- **Long context**: Needle-in-Haystack at L=4K / 32K / 128K; PG19 long-context PPL
- **Generation quality** (370M only, if time): MMLU, TriviaQA

## Data Collection Checklist

### Tables for paper
- [ ] Table 1: Perplexity (Mamba-3 FP16, Mamba-2 FP16, BitMamba-2, BitMamba-3 at 30M/130M/370M)
- [ ] Table 2: Downstream zero-shot accuracy
- [ ] Table 3: Throughput across context length (RTX 5090; CPU BitMamba-2 baseline)
- [ ] Table 4: Energy efficiency (tok/s/W) on RTX 5090 and CPU
- [ ] Table 5: Ablation study results

### Figures
- [ ] Fig 1: BitMamba-3 architecture diagram
- [ ] Fig 2: Training loss curves (ablations)
- [ ] Fig 3: Energy comparison bar chart (RTX 5090 vs CPU)
- [ ] Fig 4: Long-context needle-in-haystack heatmap
- [ ] Fig 5: Perplexity vs model size (scaling law)
- [ ] Fig 6: Parity accuracy — ternary vs FP across seeds and scales

## Deferred / Future Work

- 1B+ model training (requires multi-GPU or cloud)
- Larger-scale parity ablations (d=512, depth=4, 10+ seeds)
- Direct ternarization + fine-tuning once Mamba-3 public checkpoints release
