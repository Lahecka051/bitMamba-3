# BitMamba-3: 1.58-bit Ternary Mamba-3 on FPGA

Research prototype implementing 1.58-bit ternary quantization of the Mamba-3 state space model architecture (arXiv:2603.15569) with FPGA acceleration on Zybo Z7-20.

## Status

**Phase 1 complete** — software stack + 30M/130M/370M training + state-tracking finding + RTL skeletons.

## Headline Results

- **Parity (state-tracking) — 5σ separation**: At d=256/depth=2/cosine LR/5 seeds, Mamba-3 + ternary reaches **0.95 ± 0.08** peak parity accuracy across 5 seeds while Mamba-3 FP stays at chance (0.52 ± 0.02). Mamba-2 (FP and ternary) cannot solve parity at any seed (σ < 0.01). See `docs/04_parity_unexpected_finding.md` and `results/figures/fig2_parity_ternary_vs_fp.pdf`.

- **Training**: BitMamba-3 130M MIMO trained from scratch on 480M fineweb-edu tokens reaches WikiText-103 PPL **69.4** (vs 30M's 400 at the same token count — a 5.8× improvement from scale alone).

- **Zero-shot (130M, 200 samples/task)**: ARC-Easy 0.41, HellaSwag 0.39, PIQA 0.57. All above chance. Approaches published Mamba-2 130M (300 B tokens) on ARC-Easy and HellaSwag despite our 0.16% training-data fraction.

- **Throughput**: Mamba-2 FP16 baseline matrix on RTX 5090 (130M / 370M / 1.3B × L=512..32K) saved to `results/tables/bench_mamba_baseline.csv`. BitMamba-3 30M trains at 195 K tok/s, 130M at 44 K tok/s.

- **RTL**: 6 Verilog/SystemVerilog modules targeting Zybo Z7-20. `bit_mac` (128-lane ternary MAC) bit-exact verified by Verilator. Plug-in integration plan with prior `/g/Xilinx/` AXI HP DMA + AXI Lite CSR (~1000 lines saved + bring-up history).

## Scope

Single-board FPGA implementation. Multi-board / distributed extensions are future work.

## Directory Layout

```
bitMamba-3/
├── docs/                    # Experiment plans, architecture specs, paper drafts
├── src/
│   ├── bitmamba3/           # BitNet-wrapped Mamba-3 modules (PyTorch)
│   ├── training/            # Training scripts
│   ├── evaluation/          # lm-eval-harness wrappers, custom benchmarks
│   └── benchmarks/          # Throughput/energy measurement harnesses
├── third_party/             # External deps (state-spaces/mamba, BitMamba-2 ref)
├── experiments/
│   ├── 30M_proxy/           # Convergence verification
│   ├── 130M_main/           # Primary results
│   ├── 370M_scaling/        # Model size scaling
│   ├── ablations/           # Ternarization scope, MIMO size, etc.
│   └── baselines/           # Mamba-3 FP16, BitMamba-2 CPU
├── data/                    # Training data (SlimPajama, C4, WikiText)
├── checkpoints/             # Trained model weights
├── results/
│   ├── tables/              # Paper tables (perplexity, throughput, energy)
│   ├── figures/             # Paper figures
│   └── logs/                # Training/eval logs
├── rtl/
│   ├── cores/bitmamba3/     # Verilog RTL (ternary MAC, RoPE, SSM)
│   ├── bd/                  # Vivado block design TCL
│   └── vivado_project/      # Vivado project files
├── sim/verilator/           # Verilator testbenches
├── petalinux/               # BOOT.BIN, test programs for Zybo
└── scripts/                 # Utility scripts
```

## Environment

- **OS**: Windows 11 + WSL Ubuntu-24.04
- **GPU**: NVIDIA RTX 5090 (32GB, Blackwell)
- **Python env**: `taq-vidssm` in WSL miniconda (`~/miniconda3/envs/taq-vidssm/`)
  - Python 3.11.15, PyTorch 2.11.0+cu130, mamba_ssm 2.3.1, causal_conv1d 1.6.1
- **CUDA**: Toolkit 12.8, Driver 591.86 (runtime up to 13.1)
- **FPGA**: Zybo Z7-20 (Zynq-7020), single board
- **Prior infrastructure**: `G:\Xilinx\` (RTL, PetaLinux, BOOT.BIN for axi_dot_hp, gguf_q4_0, gguf_q8_0)

## Reference

- Mamba-3 paper: arXiv:2603.15569 (ICLR 2026)
- state-spaces/mamba: github.com/state-spaces/mamba (v2.3.1)
- BitMamba-2: github.com/Zhayr1/BitMamba-2
- BitNet b1.58: arXiv:2402.17764

## Reproducibility — Quick Commands

All commands assume WSL Ubuntu 24.04 with the `taq-vidssm` conda env and that
the working directory is `G:\Github Desktop\bitMamba-3`.

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
paper structure and `results/tables/SUMMARY.md` for the consolidated results
table.
