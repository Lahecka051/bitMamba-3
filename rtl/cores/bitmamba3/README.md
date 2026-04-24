# BitMamba-3 RTL Modules for Zybo Z7-20

Target FPGA: `xc7z020clg400-1` (Zynq-7020). Single board, AXI-HP DMA interface.

## Module Plan

Each module has a Verilator testbench under `sim/verilator/` that compares
against a PyTorch golden reference generated from `src/bitmamba3/`.

### 1. `bit_mac.v` — Ternary weight × INT8 activation MAC unit

- 128-wide conditional add/sub based on 2-bit weight encoding (00=zero, 01=+1, 10=-1, 11=reserved)
- Input activation: INT8 (8 bits)
- Output accumulator: INT24 (sufficient headroom for 128-wide sum with INT8+ternary)
- 3-stage pipeline (similar to prior GGUF Q4_0 MAC)

### 2. `rope_engine.v` — Data-dependent RoPE rotation

- Input: (x_real, x_imag, theta) where (x_real, x_imag) is a pair, theta = per-token angle
- Output: (x_real * cos(theta) - x_imag * sin(theta), x_real * sin(theta) + x_imag * cos(theta))
- sin/cos via small BRAM LUT (256-entry, FP16 values)
- FP16 arithmetic (or INT16 with post-scale)

### 3. `rmsnorm_int8.v` — RMSNorm + INT8 activation quantizer (pre-BitLinear)

- Computes RMS over last dim (128 elements at a time)
- Divides, then scales to INT8 using per-row absmax / 127
- Single pipeline, uses DSP for reciprocal sqrt

### 4. `selective_scan_mimo.v` — Mamba-3 MIMO SSM state update

- Sequential recurrence: `state = state * A_discrete + B * x` (vector form)
- MIMO rank = 4 (paper default)
- State storage in BRAM (per head × rank × d_state × headdim)
- One token per cycle after pipeline fill

### 5. `mimo_matmul.v` — MIMO projection matrix multiplication

- `mimo_x / mimo_z / mimo_o` applied as einsum: (bhp, hrp → brhp) reshape then multiply
- FP16 matmul (these weights stay FP in our design; ablation: ternary)

### 6. `top_bitmamba3_block.v` — Single Mamba-3 MIMO block top-level

- AXI4-HP interface for DDR3 weight streaming
- AXI4-Lite slave for control/status registers
- Sequences: in_proj (BitLinear) → split → RMSNorm on B/C → RoPE → selective_scan → mimo_out → out_proj (BitLinear)

## Resource Target (Zybo Z7-20)

| Module | LUT budget | DSP budget | BRAM (36Kb blocks) |
|---|---|---|---|
| bit_mac × 4 instances | 8,000 | 0 | 4 |
| rope_engine | 2,000 | 8 | 2 |
| rmsnorm_int8 | 3,000 | 20 | 0 |
| selective_scan_mimo | 5,000 | 40 | 20 |
| mimo_matmul | 3,000 | 32 | 0 |
| top FSM + AXI | 5,000 | 0 | 4 |
| **Total** | **~26,000** (49% of 53K) | **~100** (45% of 220) | **~30** (21% of 140) |

Comfortable fit on Zybo Z7-20.

## Phase Plan

1. **Phase 1**: bit_mac + activation quantizer Verilator testbench
2. **Phase 2**: RoPE engine Verilator testbench
3. **Phase 3**: Selective scan MIMO Verilator testbench (most complex)
4. **Phase 4**: Top-level integration + Vivado synthesis
5. **Phase 5**: PetaLinux bring-up + bit-exact verification vs PyTorch
