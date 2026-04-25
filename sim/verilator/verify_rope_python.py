"""Pure-Python verification of rope test vectors."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]


def main():
    path = _root / "sim/verilator/vectors/rope.bin"
    if not path.exists():
        print(f"Missing {path}")
        sys.exit(1)

    with open(path, "rb") as f:
        blob = f.read()

    rec_size = 2 + 2 + 2 + 2 + 2  # x0, x1, theta_q13, x0p, x1p (uint16/int16 each)
    n = len(blob) // rec_size
    print(f"Read {n} records from {path}")

    ok = bad = 0
    for i in range(n):
        off = i * rec_size
        x0_u, x1_u, theta_q13 = struct.unpack("<HHh", blob[off:off + 6])
        x0p_u, x1p_u = struct.unpack("<HH", blob[off + 6:off + 10])

        x0 = np.frombuffer(np.array([x0_u], dtype=np.uint16).tobytes(), dtype=np.float16)[0]
        x1 = np.frombuffer(np.array([x1_u], dtype=np.uint16).tobytes(), dtype=np.float16)[0]
        x0p = np.frombuffer(np.array([x0p_u], dtype=np.uint16).tobytes(), dtype=np.float16)[0]
        x1p = np.frombuffer(np.array([x1p_u], dtype=np.uint16).tobytes(), dtype=np.float16)[0]

        theta_rad = theta_q13 / 8192.0
        c = np.float32(np.cos(theta_rad))
        s = np.float32(np.sin(theta_rad))
        x0p_re = np.float16(np.float32(x0) * c - np.float32(x1) * s)
        x1p_re = np.float16(np.float32(x0) * s + np.float32(x1) * c)

        # Allow 1 ULP tolerance in fp16
        if abs(np.float32(x0p_re) - np.float32(x0p)) <= 1e-3 and abs(np.float32(x1p_re) - np.float32(x1p)) <= 1e-3:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"MISMATCH vec={i}: x0={x0} x1={x1} theta_q13={theta_q13} -> got ({x0p}, {x1p}) recomputed ({x0p_re}, {x1p_re})")

    print(f"rope vectors: ok={ok} bad={bad} (out of {n})")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
