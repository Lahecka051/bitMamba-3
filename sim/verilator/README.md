# Verilator Testbenches

Each `tb_<module>.cpp` file drives its RTL unit with vectors produced by a
PyTorch golden reference in `src/bitmamba3/` and asserts bit-exact match.

## Prerequisites

Verilator 5.0+. Install in WSL:

```bash
sudo apt-get install verilator   # may be Verilator 4.x on Ubuntu 24.04
```

For Verilator 5 (recommended):

```bash
git clone https://github.com/verilator/verilator.git
cd verilator
autoconf
./configure
make -j8
sudo make install
```

## Testbenches

- `tb_bit_mac/` — conditional add/sub/skip ternary MAC (128-lane, INT8 activation)
- `tb_rope_engine/` — data-dependent rotation (FP16 pair + theta LUT)
- `tb_rmsnorm_int8/` — RMSNorm + INT8 quantizer pre-stage
- `tb_selective_scan/` — Mamba-3 MIMO recurrence (FP16 state accumulator)

## Golden Reference Generation

Each module has a Python script that emits vectors to `vectors/` directory:

```bash
python src/bitmamba3/gen_vectors_bit_mac.py --n 1000 --lanes 128 --out vectors/bit_mac.npz
```

Each .npz contains numpy arrays for inputs and expected outputs in the same
fixed-point / FP16 encoding as the RTL port widths.
