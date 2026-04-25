"""Generate test vectors for rope_engine.v.

Each tuple: (x0_fp16, x1_fp16, theta_q13) -> (x0'_fp16, x1'_fp16)
Computed as:
    cos = cos(theta_radians); sin = sin(theta_radians)
    x0' = x0 * cos - x1 * sin
    x1' = x0 * sin + x1 * cos

theta_q13: signed 16-bit, theta_radians = theta_q13 / 2**13.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]


def fp32_to_fp16_uint(x):
    return np.float16(x).view(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sim/verilator/vectors/rope.bin")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        for _ in range(args.n):
            x0 = np.float32(rng.normal(0, 1))
            x1 = np.float32(rng.normal(0, 1))
            theta_q13 = int(rng.integers(-32768, 32768))
            theta_rad = np.float32(theta_q13 / 8192.0)  # 2^13
            c = np.float32(np.cos(theta_rad))
            s = np.float32(np.sin(theta_rad))
            x0p = np.float32(x0 * c - x1 * s)
            x1p = np.float32(x0 * s + x1 * c)

            f.write(struct.pack("<HHhHH",
                                int(fp32_to_fp16_uint(x0)),
                                int(fp32_to_fp16_uint(x1)),
                                theta_q13,
                                int(fp32_to_fp16_uint(x0p)),
                                int(fp32_to_fp16_uint(x1p))))

    print(f"Wrote {args.n} rope vectors to {out_path}")


if __name__ == "__main__":
    main()
