# RTX 5090 (Blackwell SM 12.0) MIMO TileLang Backward Shared Memory Budget

## Issue

Upstream state-spaces/mamba v2.3.1 `mamba3_mimo_combined` backward kernel
allocates dynamic shared memory proportional to `chunk_size * mimo_rank`. On
NVIDIA Blackwell consumer GPUs (RTX 5090, SM 12.0), the hardware's allowed
dynamic shmem per block is lower than on H100/H200 (SM 9.0/10.0), causing:

```
tvm.error.InternalError: Failed to set the allowed dynamic shared memory
size to <N> bytes
```

observed at N ≥ 143,616 bytes on our setup.

## Empirical Compatibility Sweep (RTX 5090, CUDA 12.8, TileLang 0.1.8)

Test config: d_model=384, d_state=64, headdim=64, batch=2, seqlen=128.

| mimo_rank | chunk_size | chunk×rank | Shmem requested | Backward |
|---|---|---|---|---|
| 4 | 16 | 64 | 146,080 B | **FAIL** |
| 4 | 32 | 128 | 143,616 B | FAIL (warn: "chunk×rank > 64") |
| 4 | **8** | **32** | (fits) | ✅ **OK** |
| 2 | 16 | 32 | (fits) | ✅ OK |
| 2 | 32 | 64 | 150,080 B | FAIL |

Pattern: **`chunk_size × mimo_rank ≤ 32` works on RTX 5090**, anything larger
exceeds Blackwell's dynamic shmem limit.

## Workaround Adopted

All BitMamba-3 MIMO presets use `chunk_size=8` (with `mimo_rank=4`), giving
`chunk × rank = 32`. This is **half** the upstream recommendation (`64 / mimo_rank = 16`).

Upstream comment in `mamba3.py` line 46:
```python
chunk_size=64,  # Recommended: 64 for SISO, 64/mimo_rank for MIMO
```

## Paper Defense

This is a **hardware-specific tuning** not an algorithmic modification:

- Mamba-3 MIMO math is unchanged.
- TileLang kernel is unchanged.
- Only the `chunk_size` kernel-launch parameter is halved for Blackwell
  consumer-grade GPUs with reduced dynamic shmem budget.
- Forward + backward produce identical results to upstream at smaller chunks.
- The reduction would be reverted on data-center GPUs (H100/H200, B100+).

Numerical correctness is preserved; only compilation / launch feasibility
changes.

## References

- Mamba-3 paper arXiv:2603.15569 (Section 3, MIMO formulation)
- state-spaces/mamba commit v2.3.1, `mamba_ssm/modules/mamba3.py`
- TileLang 0.1.8 kernel: `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`
