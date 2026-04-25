# Reusable RTL Infrastructure from `G:\Xilinx\rtl\`

The user's prior FPGA project (`/g/Xilinx/`) on the same Zybo Z7-20 board has
working RTL modules and BD TCLs that BitMamba-3 should reuse rather than
re-write. This document catalogues the reuse opportunities and the
adaptation strategy.

## Existing Cores

### `/g/Xilinx/rtl/cores/attention_engine/` (most valuable)

| File | Status | Reuse |
|---|---|---|
| `axi_hp_dma.sv` | ✅ working | **Direct reuse** — drives DDR3 reads/writes via PS AXI HP0 |
| `axi_lite_csr.sv` | ✅ working | **Direct reuse** — start/done/addr CSRs for the BitMamba-3 top |
| `attention_fsm.sv` | working FSM | Template for `top_bitmamba3_block` orchestration |
| `attention_engine_top.sv` | full top | Reference for AXI port wiring |
| `systolic_16x8.sv` | 16×8 systolic array | Possibly reuse for `mimo_matmul` |

### `/g/Xilinx/rtl/cores/dequant/`

| File | Status | Reuse |
|---|---|---|
| `dequant_int4_to_fp16.sv` | working | Adapt: replace INT4 unpack with ternary 2-bit unpack for our `bit_mac` consumer |

### `/g/Xilinx/rtl/cores/softmax/`

| File | Status | Reuse |
|---|---|---|
| `softmax_lut.sv` | working LUT-based softmax | Reuse inside `selective_scan_mimo` if we want a fully-on-chip softmax |

### `/g/Xilinx/rtl/cores/gguf_accel/`

| File | Status | Reuse |
|---|---|---|
| `axi_gguf_q4_0_dot.v` | ✅ field-tested | **Template** for `top_bitmamba3_block`'s AXI dot-product front-end |
| `q4_0_int_mac.v` | ✅ working 3-stage MAC | Architectural template — our `bit_mac` is similar (2-bit weight vs 4-bit) |
| `q4_K_int_mac.v` | working | Reference for multi-block decoded MAC |
| `fp16_x_fp16_x_int_to_fp32.v` | working mixed precision | Reuse for the post-MAC FP scale step in BitLinear path |
| `q4_K_scale_decoder.v` | working 6-bit scale decoder | Reference for the BitLinear `scale_w / scale_x` decode |

### `/g/Xilinx/rtl/cores/kv_l1_cache/`

| File | Reuse |
|---|---|
| KV cache modules | Reusable if we extend to Mamba-3 inference state caching |

### `/g/Xilinx/rtl/cores/min_accel/`, `quant_pack/`

Less directly relevant but available.

## Existing Block Designs

### `/g/Xilinx/rtl/bd/`

| TCL | Project | Status | Reuse |
|---|---|---|---|
| `create_dot_hp_bd.tcl` | `axi_dot_hp` baseline | ✅ working | **Template** — our `create_bitmamba3_bd.tcl` mirrors this |
| `create_gguf_q4_0_bd.tcl` | GGUF Q4_0 accel | ✅ working | Template for axi-stream-style int MAC accel |
| `create_gguf_q8_0_bd.tcl` | GGUF Q8_0 accel | ✅ working | Same |
| `create_kv_cache.tcl` | KV cache | working | Reference for state-keeping designs |
| `create_min_accel_bd.tcl` | minimal accel | working | Compact wrapper template |

### `/g/Xilinx/petalinux_zybo/`

| Folder | Reuse |
|---|---|
| `images_dot_hp/BOOT.BIN` | **Working baseline** — proven Zybo bring-up |
| `images_gguf_q4_0/BOOT.BIN` | Working with custom RTL |
| `images_gguf_q8_0/BOOT.BIN` | Working |
| `test_programs/test_dot_hp.c` | **Template** for `test_bitmamba3.c` |

## Adaptation Strategy for BitMamba-3

Rather than developing from scratch, we **plug** BitMamba-3-specific datapath
modules (`bit_mac.v`, `rope_engine.v`, `selective_scan_mimo.v`) into the
existing AXI HP DMA + AXI Lite CSR + FSM scaffold.

Concretely, `top_bitmamba3_block.v` should be rewritten to instantiate:

```
top_bitmamba3_block:
  ├── axi_hp_dma           ← reuse from /g/Xilinx/rtl/cores/attention_engine/axi_hp_dma.sv
  ├── axi_lite_csr         ← reuse from /g/Xilinx/rtl/cores/attention_engine/axi_lite_csr.sv
  ├── bitmamba3_fsm        ← new, modeled on attention_fsm.sv
  ├── rmsnorm_int8         ← our new module (placeholder pending FP16 IP)
  ├── bit_mac              ← our new module, fully functional
  ├── rope_engine          ← our new module (placeholder pending FP16 IP)
  ├── selective_scan_mimo  ← our new module (placeholder)
  ├── mimo_matmul          ← our new module, can use systolic_16x8.sv pattern
  └── bit_mac (out_proj)   ← reuse our bit_mac instance
```

This **reduces the new RTL surface area** to roughly the BitMamba-3-specific
datapath, while **preserving the field-validated AXI plumbing** from prior
projects.

## Estimated Reuse Savings

| Component | Lines saved (vs scratch) |
|---|---|
| AXI HP DMA | ~500 |
| AXI Lite CSR | ~200 |
| FSM template | ~150 |
| Softmax LUT | ~100 |
| Mixed-precision MAC reference | ~100 |
| **Total** | **~1,050 lines + 6 months of debugging** |

## Next Action

Update `rtl/cores/bitmamba3/top_bitmamba3_block.v` to instantiate the existing
`axi_hp_dma` and `axi_lite_csr` modules from `G:\Xilinx\rtl\cores\attention_engine\`.
Update `rtl/bd/create_bitmamba3_bd.tcl` to add these source files to the project
file list. (Pending verification once Vivado is available.)
