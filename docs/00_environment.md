# Environment Specification

Confirmed 2026-04-25.

## Host Machine

| Component | Spec |
|---|---|
| CPU | Intel Core Ultra 9 285K |
| GPU | NVIDIA RTX 5090 (32GB VRAM, Blackwell, SM 12.0) |
| RAM | (TBD — check) |
| Disk G: | 1.9 TB total, 989 GB free |
| Disk C: | 1.9 TB total, 577 GB free |
| OS | Windows 11 Pro 10.0.26200 |

## GPU / CUDA

| Item | Value |
|---|---|
| Driver | 591.86 |
| Runtime CUDA | up to 13.1 |
| CUDA Toolkit | 12.8 (primary), 12.1 (legacy) |
| nvcc path | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe` |

## Python Environments

Primary environment: **WSL Ubuntu-24.04 `taq-vidssm`** at `/home/qerti/miniconda3/envs/taq-vidssm`

| Package | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.11.0+cu130 |
| torchaudio | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| mamba_ssm | 2.3.1 |
| causal_conv1d | 1.6.1 |

CUDA test: `torch.cuda.is_available() == True`, GPU `NVIDIA GeForce RTX 5090`, compute capability `(12, 0)`.

Other envs (not used): `kivi` (WSL), `base`, `iopaint`, `parking_macro`, `rfdetr_env`, `swin_lora`, `udm-build`, `vividdet`, `yolo_env` (Windows conda).

## Invocation Pattern

Always invoke Python via WSL with clean environment to avoid Windows PATH contamination:

```bash
wsl -d Ubuntu-24.04 --exec bash -lc 'env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c "source /home/qerti/miniconda3/etc/profile.d/conda.sh && conda activate taq-vidssm && <command>"'
```

Reason: Windows conda script inherits Windows PATH containing `(x86)` directories that break bash `eval` syntax when conda initializes.

## Working Directory

- Host path: `G:\Github Desktop\bitMamba-3\`
- WSL path: `/mnt/g/Github\ Desktop/bitMamba-3/`

## Git

- User: Lahecka051
- Email: gtbbknq2001@gmail.com
- Git version: 2.51.2.windows.1

## FPGA Tools

- **Vivado**: Install path TBD (not found in expected locations)
- **Prior Zybo artifacts** at `G:\Xilinx\`:
  - `docs/edge_kv_accel/` — prior project session handoff docs (08 files)
  - `petalinux_zybo/images_dot_hp/BOOT.BIN` — AXI HP DMA dot product baseline (preserved)
  - `petalinux_zybo/images_gguf_q4_0/BOOT.BIN` — GGUF Q4_0 accelerator
  - `petalinux_zybo/images_gguf_q8_0/BOOT.BIN` — GGUF Q8_0 accelerator
  - `rtl/cores/` — Verilog modules, reusable for new BitMamba-3 RTL
- **Zybo Z7-20 board**: Single unit (multi-board scope deferred)

## Dependencies To Add

Packages to install into `taq-vidssm`:

```bash
pip install datasets transformers accelerate wandb
pip install lm-eval  # lm-evaluation-harness
pip install tiktoken sentencepiece  # tokenizers
# Optional, for JAX BitMamba-2 reference:
# pip install jax flax optax
```
