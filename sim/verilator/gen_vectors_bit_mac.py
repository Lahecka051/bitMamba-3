"""Generate bit-exact test vectors for bit_mac.v from PyTorch reference.

Emits `vectors/bit_mac.bin` (raw binary) with repeated tuples of:
    LANES bytes (INT8 activations)
    LANES/4 bytes (2-bit ternary weights, packed 4 per byte)
    4 bytes (INT32 expected accumulator output)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parents[2]


def generate_ternary_weights(lanes, rng):
    """Random ternary weights in {-1, 0, +1}."""
    return rng.integers(-1, 2, size=(lanes,), dtype=np.int8)


def encode_weights_2bit(w):
    """Encode ternary weights (-1/0/+1) into 2-bit codes (00=0, 01=+1, 10=-1)."""
    codes = np.zeros(len(w), dtype=np.uint8)
    codes[w == 1] = 0b01
    codes[w == -1] = 0b10
    # Pack 4 codes per byte (little-endian inside byte)
    packed = np.zeros(len(w) // 4, dtype=np.uint8)
    for i in range(len(w) // 4):
        b = 0
        for k in range(4):
            b |= (int(codes[i * 4 + k]) & 0b11) << (k * 2)
        packed[i] = b
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--lanes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sim/verilator/vectors/bit_mac.bin")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert args.lanes % 4 == 0, "LANES must be multiple of 4 for 2-bit packing"

    with open(out_path, "wb") as f:
        for _ in range(args.n):
            act = rng.integers(-128, 128, size=(args.lanes,), dtype=np.int8)
            w = generate_ternary_weights(args.lanes, rng)
            w_packed = encode_weights_2bit(w)
            expected = int(np.sum(act.astype(np.int32) * w.astype(np.int32)))

            f.write(act.tobytes())
            f.write(w_packed.tobytes())
            f.write(struct.pack("<i", expected))

    print(f"Wrote {args.n} vectors to {out_path}, expected size={out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
