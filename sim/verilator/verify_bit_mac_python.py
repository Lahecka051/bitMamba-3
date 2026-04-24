"""Pure-Python verification of bit_mac test vectors (no Verilator required).

Reads sim/verilator/vectors/bit_mac.bin and re-computes expected accumulator
from the bundled INT8 activations + ternary weights. Confirms the vector
generator is self-consistent and matches the RTL semantics documented in
`bit_mac.v`.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]


def unpack_2bit_codes(packed: bytes, n_lanes: int) -> np.ndarray:
    """Unpack 4-per-byte 2-bit codes."""
    codes = np.zeros(n_lanes, dtype=np.int8)
    for i in range(n_lanes):
        code = (packed[i // 4] >> ((i % 4) * 2)) & 0x3
        if code == 0b01:
            codes[i] = 1
        elif code == 0b10:
            codes[i] = -1
        else:
            codes[i] = 0
    return codes


def main():
    lanes = 128
    act_bytes = lanes
    w_bytes = lanes // 4
    exp_bytes = 4

    path = _root / "sim/verilator/vectors/bit_mac.bin"
    if not path.exists():
        print(f"Missing {path}. Run gen_vectors_bit_mac.py first.")
        sys.exit(1)

    with open(path, "rb") as f:
        blob = f.read()

    tuple_size = act_bytes + w_bytes + exp_bytes
    n = len(blob) // tuple_size
    if len(blob) % tuple_size != 0:
        print(f"File size {len(blob)} not divisible by tuple size {tuple_size}")
        sys.exit(1)

    ok = 0
    bad = 0
    for i in range(n):
        off = i * tuple_size
        act = np.frombuffer(blob[off:off + act_bytes], dtype=np.int8)
        packed = blob[off + act_bytes:off + act_bytes + w_bytes]
        expected = struct.unpack("<i", blob[off + act_bytes + w_bytes:off + tuple_size])[0]

        codes = unpack_2bit_codes(packed, lanes)
        recomputed = int(np.sum(act.astype(np.int32) * codes.astype(np.int32)))
        if recomputed == expected:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"MISMATCH vec={i}: recomputed={recomputed} expected={expected}")

    print(f"Verified {n} vectors: ok={ok} bad={bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
