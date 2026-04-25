# PetaLinux Integration Plan for BitMamba-3 on Zybo Z7-20

Reuses the prior infrastructure from `G:\Xilinx\petalinux_zybo\` (which has
working BOOT.BIN images for `axi_dot_hp`, `gguf_q4_0`, `gguf_q8_0`).

## Plan

1. **XSA export**: After `vivado -source rtl/bd/create_bitmamba3_bd.tcl`
   completes, export hardware as `gguf_bitmamba3.xsa`.

2. **PetaLinux project**: clone the existing `axi_dot_hp` PetaLinux project
   tree as `petalinux/bitmamba3/` and update the hardware description:
   ```
   cd petalinux/bitmamba3
   petalinux-config --get-hw-description=../../rtl/vivado_project/gguf_bitmamba3_proj/gguf_bitmamba3.xsa
   ```

3. **Test program**: write `petalinux/bitmamba3/test_programs/test_bitmamba3.c`
   that mmaps the AXI Lite control region, programs weight/activation DDR
   addresses, kicks off the start signal, and reads result back.

4. **Bit-exact verification**: drive the FPGA with PyTorch-generated
   activation tensors and compare per-token hidden state output to the
   PyTorch BitMamba-3 reference. Tolerance: bit-exact for ternary MAC
   path, FP16 ULP-tolerant for selective_scan path.

5. **Throughput measurement**: cycle counter on FPGA + wall-clock host
   timing for throughput in tokens/sec at L = 512, 2K, 8K, 32K, 131K.

6. **Energy measurement**: external Kill-A-Watt or INA219 on Zybo's 5V
   barrel jack; sample power during sustained-throughput runs.

## Status

This README is a forward-looking plan. RTL implementation in
`rtl/cores/bitmamba3/*.v` is at skeleton level — the FP16 datapath is
declared as ports but functional substitution with FP16 IP / numerical
units is pending.

## Files (to be created)

- `petalinux/bitmamba3/test_programs/test_bitmamba3.c`
- `petalinux/bitmamba3/test_programs/Makefile`
- `petalinux/bitmamba3/scripts/build_boot_bin.sh`
- `petalinux/bitmamba3/images/BOOT.BIN`         (after build)
- `petalinux/bitmamba3/images/image.ub`         (kernel)
