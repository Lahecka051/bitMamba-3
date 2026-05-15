# BitMamba-3: 1.58-bit Ternary Mamba-3

Research prototype combining **1.58-bit ternary quantization** (BitNet b1.58)
with the **Mamba-3** inference-first state space model (arXiv:2603.15569).
BitMamba-3 ternarizes the projection layers of the Mamba-3 block while leaving
the SSM kernels (Triton SISO, TileLang MIMO, CuteDSL decode) untouched, and
studies what that quantization does to the model's state-tracking ability.

This is a **software / algorithm** project: a PyTorch implementation, training
at 30M/130M/370M scales, and an empirical state-tracking analysis.

## Status

Software stack complete — minimal-diff BitMamba-3 wrapper, 30M/130M/370M
from-scratch training, and the parity state-tracking finding.

## Headline Results

- **Parity (state-tracking) — ~5σ separation**: At d=256 / depth=2 / cosine LR
  / 5 seeds, Mamba-3 + ternary reaches **0.95 ± 0.08** peak parity accuracy
  while Mamba-3 FP stays at chance (0.52 ± 0.02). At d=512 / depth=4 the
  separation widens to ~13σ (MIMO+ternary 0.98 ± 0.04). Mamba-2 (FP and
  ternary) cannot solve parity at any seed. This positions **ternary
  quantization as an inductive bias for state tracking**, distinct from the
  usual compression-only framing. See `docs/04_parity_unexpected_finding.md`
  and `results/figures/`.

- **Training (from scratch, fineweb-edu, 480M tokens)**: BitMamba-3 130M
  reaches WikiText-103 PPL **69.4**; 370M reaches **60.2**. The 30M → 130M
  jump (same token budget) is a 5.8× PPL improvement from scale alone.

- **Zero-shot (130M, 200 samples/task)**: ARC-Easy 0.41, HellaSwag 0.39 (norm),
  PIQA 0.57 — all above chance, approaching published Mamba-2 130M (trained on
  ~600× more tokens) on ARC-Easy and HellaSwag.

- **Throughput**: BitMamba-3 30M trains at 195K tok/s, 130M at 44K tok/s on
  RTX 5090. Mamba-2 FP16 baseline matrix (130M / 370M / 1.3B × L=512..32K)
  saved to `results/tables/bench_mamba_baseline.csv`.

## Directory Layout

```
bitMamba-3/
├── docs/                    # architecture analysis, paper outline + draft, surveys
├── src/
│   ├── bitmamba3/           # BitNet-wrapped Mamba-3 modules (PyTorch)
│   ├── training/            # training scripts
│   ├── evaluation/          # lm-eval-harness wrappers, custom benchmarks
│   └── benchmarks/          # throughput measurement harnesses
├── ppq/                     # post-training quantization experiments
├── scripts/                 # training / parity / evaluation entry points
├── results/
│   ├── tables/              # paper tables (perplexity, throughput)
│   ├── figures/             # paper figures
│   └── logs/                # training / eval logs
├── paper/                   # paper generator (generate_paper.js)
├── third_party/             # external deps: state-spaces/mamba, BitMamba-2 ref
├── experiments/             # experiment configs / outputs
├── data/                    # training data (fineweb-edu, WikiText, PG19)
└── checkpoints/             # trained model weights
```

(`third_party/`, `experiments/`, `data/`, `checkpoints/`, and large binaries
are gitignored.)

## Environment

- **OS**: Windows 11 + WSL Ubuntu-24.04
- **GPU**: NVIDIA RTX 5090 (32 GB, Blackwell, SM 12.0)
- **Python env**: `taq-vidssm` in WSL miniconda (`~/miniconda3/envs/taq-vidssm/`)
  - Python 3.11.15, PyTorch 2.11.0+cu130, mamba_ssm 2.3.1, causal_conv1d 1.6.1
- **CUDA**: Toolkit 12.8, Driver 591.86

See `docs/00_environment.md` for the full specification.

## Reference

- Mamba-3 paper: arXiv:2603.15569 (ICLR 2026)
- state-spaces/mamba: github.com/state-spaces/mamba (v2.3.1)
- BitMamba-2: github.com/Zhayr1/BitMamba-2
- BitNet b1.58: arXiv:2402.17764

## Reproducibility — Quick Commands

All commands assume WSL Ubuntu 24.04 with the `taq-vidssm` conda env, run from
the repository root.

```bash
# Sanity test (BitLinear + BitMamba-3 forward pass)
./scripts/wsl-run.sh python scripts/sanity_check_bitlinear.py
./scripts/wsl-run.sh python scripts/sanity_check_mimo.py

# Reproduce parity 5-seed sweep at d=256 (main result, ~80 min on RTX 5090)
./scripts/wsl-run.sh python scripts/run_parity_larger_bg.py
./scripts/wsl-run.sh python src/evaluation/analyze_parity_multiseed.py

# Train BitMamba-3 30M proxy on 1B fineweb-edu (~14 min)
./scripts/wsl-run.sh python src/training/prepare_data.py \
    --dataset HuggingFaceFW/fineweb-edu --target_tokens 1000000000 \
    --out_dir data/fineweb_1B --shard_size 100000000
./scripts/wsl-run.sh python src/training/train.py --preset 30M \
    --data_dir data/fineweb_1B --out_dir checkpoints/30M

# Evaluate
./scripts/wsl-run.sh python src/evaluation/quick_eval.py \
    --ckpt checkpoints/30M/ckpt_final_010000.pt --preset 30M
./scripts/wsl-run.sh python src/evaluation/run_lm_eval.py \
    --ckpt checkpoints/30M/ckpt_final_010000.pt --preset 30M --limit 200

# Generate paper figures from accumulated data
./scripts/wsl-run.sh python src/evaluation/generate_paper_figures.py
```

See `docs/05_paper_outline.md` and `docs/06_paper_draft.md` for the working
paper structure, and `results/tables/` for the consolidated result tables.
