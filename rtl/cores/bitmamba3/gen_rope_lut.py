"""Generate sin/cos LUT mem files for rope_engine.v.

LUT layout: 1024 entries, addressed by top-10 bits of theta_q13.
Each entry is FP16 (16-bit hex).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[3]


def fp32_to_fp16_hex(v):
    return f"{int(np.float16(v).view(np.uint16)):04x}"


def main():
    n = 1024
    sin_path = Path(__file__).parent / "rope_sin_lut.mem"
    cos_path = Path(__file__).parent / "rope_cos_lut.mem"

    with open(sin_path, "w") as fs, open(cos_path, "w") as fc:
        for i in range(n):
            theta = (i / n) * 2 * math.pi
            fs.write(fp32_to_fp16_hex(math.sin(theta)) + "\n")
            fc.write(fp32_to_fp16_hex(math.cos(theta)) + "\n")

    print(f"Wrote {n}-entry sin LUT -> {sin_path}")
    print(f"Wrote {n}-entry cos LUT -> {cos_path}")


if __name__ == "__main__":
    main()
