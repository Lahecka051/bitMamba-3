# Mamba-3 vs Mamba-2 Architecture Diff (from state-spaces/mamba v2.3.1)

Source files reviewed:
- `third_party/state-spaces-mamba/mamba_ssm/modules/mamba3.py`
- `third_party/state-spaces-mamba/mamba_ssm/modules/mamba2.py`
- `third_party/bitmamba-2/src/model.py` (BitMamba-2 ternarization reference)

## Module-level Structural Differences

### 1. Input Projection Output (`in_proj`)

Mamba-2:
```
[z, x, B, C, dt]
sizes: [d_inner, d_inner, ngroups*d_state, ngroups*d_state, nheads]
```

Mamba-3:
```
[z, x, B, C, dd_dt, dd_A, trap, angles]
sizes: [d_inner, d_inner,
        d_state*num_bc_heads*mimo_rank,
        d_state*num_bc_heads*mimo_rank,
        nheads, nheads, nheads, num_rope_angles]
```

Mamba-3 `d_in_proj` is **larger** — more signals to produce per token.

### 2. Causal conv1d
- Mamba-2: has `self.conv1d` (`nn.Conv1d`, depthwise, kernel_size=d_conv=4)
- Mamba-3: **removed entirely** — one fewer projection path

### 3. A parameter
- Mamba-2: static `self.A_log` parameter (nheads,)
- Mamba-3: dynamic `dd_A` projected per token from hidden state

### 4. RMSNorm on B and C (NEW in Mamba-3)
- `self.B_norm = RMSNormGated(d_state)`
- `self.C_norm = RMSNormGated(d_state)`

### 5. RoPE (NEW in Mamba-3 — critical novelty)
- `self.num_rope_angles = split_tensor_size // 2`
- `self.rotary_dim_divisor = int(2 / rope_fraction)` (default rope_fraction=0.5)
- Data-dependent `angles` tensor produced by `in_proj`
- Applied inside SSM kernel (fused with chunk scan)

### 6. MIMO (optional, `is_mimo=True`)
- Three new parameters: `mimo_x`, `mimo_z`, `mimo_o` (shape: `nheads × mimo_rank × headdim`)
- B and C become rank-augmented: `(B, L, R, G, N)` instead of `(B, L, G, N)`
- Uses `mamba3_mimo_combined` (TileLang kernel) instead of `mamba3_siso_combined` (Triton)

### 7. Trap gate (NEW)
- `trap = sigmoid(trap_proj)` — per-token per-head gate
- Shape: `(batch, nheads, seqlen)`

### 8. Biases
- `B_bias`, `C_bias`: `(nheads, mimo_rank, d_state)` — learned initialization biases
- Added to B and C before rotation/norm

### 9. `is_outproj_norm` option
- Optional: move RMSNormGated to output projection
- Default: `False`

## Fused Kernels Used

| Kernel | Location | Purpose |
|---|---|---|
| `mamba3_siso_combined` | triton/mamba3/ | Full forward SISO (training + prefill) |
| `mamba3_mimo_combined` | tilelang/mamba3/ | Full forward MIMO |
| `apply_rotary_qk_inference_fwd` | triton/mamba3/ | RoPE on Q (=C) and K (=B) |
| `mamba3_step_fn` | cute/mamba3/ | Single-token decode (H100 tested) |
| `RMSNormGated` | triton/layernorm_gated | Gated RMSNorm for B, C, output |

## Ternarization Targets for BitMamba-3

### Linear layers to replace with `BitLinear` (per BitMamba-2 strategy):

| Layer | Input dim | Output dim | Priority |
|---|---|---|---|
| `in_proj` | `d_model` | `2*d_inner + 2*d_state*num_bc_heads*mimo_rank + 3*nheads + num_rope_angles` | **Critical** (largest) |
| `out_proj` | `d_inner` | `d_model` | **Critical** (second largest) |

### Parameters to keep in FP (small, critical for stability):
- `dt_bias`, `B_bias`, `C_bias`, `D` — bias/scale vectors, `nheads`-sized
- `B_norm.weight`, `C_norm.weight` — RMSNorm scales, `d_state`-sized

### Parameters to ternarize (optional, MIMO mode only):
- `mimo_x`, `mimo_z`, `mimo_o` — extended ternarization experiment

### Activation flow (per BitMamba-2 BitLinear):
- Pre-quantization: RMSNorm on activation
- Quantization: INT8 per-token (absmax / 127)
- Matmul: effective FP (fake-quant, with STE during training)

## Complex SSM ↔ Real+RoPE Equivalence (in Code)

The Mamba-3 paper proves discrete complex SSM equals real SSM with data-dependent RoPE on B and C projections.

In code, this is **already realized** as real-valued operations:
- `apply_rotary_qk_inference_fwd(q=C, k=B, angle_state, angle_proj, dt, ...)`
- `rotate_pairwise=True` for SISO, permuted (i, i+N/2) pairs for MIMO

This means BitMamba-3 operates entirely in real domain — no complex number arithmetic needed for ternary compatibility. RoPE is done in FP (sin/cos of angles) but weights that project to `angles` can be ternarized.

## Implementation Minimal-Diff for BitMamba-3

```python
# bitmamba3/bitlinear.py
class BitLinear(nn.Linear):
    def forward(self, x):
        x_n = F.rms_norm(x, [x.size(-1)])
        scale_x = 127.0 / x_n.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        x_q = (x_n * scale_x).round().clamp(-128, 127) / scale_x
        x_eff = x_n + (x_q - x_n).detach()  # STE
        
        scale_w = 1.0 / self.weight.abs().mean().clamp(min=1e-5)
        w_q = (self.weight * scale_w).round().clamp(-1, 1) / scale_w
        w_eff = self.weight + (w_q - self.weight).detach()  # STE
        
        return F.linear(x_eff, w_eff, self.bias)


# bitmamba3/bitmamba3.py
from mamba_ssm.modules.mamba3 import Mamba3
from .bitlinear import BitLinear

class BitMamba3(Mamba3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the two Linear layers with BitLinear (same shapes)
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype
        in_in, in_out = self.in_proj.in_features, self.in_proj.out_features
        out_in, out_out = self.out_proj.in_features, self.out_proj.out_features
        self.in_proj = BitLinear(in_in, in_out, bias=False, device=device, dtype=dtype)
        self.out_proj = BitLinear(out_in, out_out, bias=False, device=device, dtype=dtype)
```

This is the **minimum viable BitMamba-3**. Extensions:
- Ternarize `mimo_x/z/o` (if MIMO mode)
- Ternarize embedding (BitMamba-2 does this for 90% bit reduction)
- Ternarize LM head output projection

## Remaining Ternarization Opportunities

| Target | Param count (370M reference) | Current plan |
|---|---|---|
| `in_proj` weight | Largest | ✅ Ternarize |
| `out_proj` weight | 2nd largest | ✅ Ternarize |
| Embedding `nn.Embedding` | `vocab_size × d_model` | Extension (consider) |
| LM head `lm_head` | `d_model × vocab_size` | Extension (tie with embedding?) |
| `mimo_x/z/o` | `nheads × mimo_rank × headdim` (small) | Optional ablation |
| RMSNorm weights | `d_state`, `d_model` (tiny) | Keep FP |
| `dt_bias`, `A_floor`, `D`, `B_bias`, `C_bias` | `nheads`-sized | Keep FP |
