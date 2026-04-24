# BitMamba-3: 1.58-bit Ternary Mamba-3 on FPGA

Research prototype implementing 1.58-bit ternary quantization of the Mamba-3 state space model architecture (arXiv:2603.15569) with FPGA acceleration on Zybo Z7-20.

## Status

**Phase 0** — Environment setup and baseline reproduction (current)

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
