"""
Phase 1 Pilot v2: Polar vs Cartesian Quantization for Mamba-3 SSM Activation
=============================================================================
Key insight from v1: Complex state is NOT stored as weight.
It's constructed during forward as:
  - ADT = softplus(A) * softplus(dt)  → magnitude (decay rate)
  - Angles → cos/sin → rotary embedding  → phase (rotation)

These are ACTIVATIONS computed inside forward, then fed to Triton kernel.
We monkey-patch Mamba3.forward() to intercept ADT and Angles,
apply quantization, then call the original kernel.

Comparison:
  Cartesian: quantize ADT and cos(Θ)/sin(Θ) independently as real values
  Polar: quantize |ADT| with log-scale grid + quantize Θ with uniform angular grid
"""
import sys, os, json, math, time, copy, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from pathlib import Path
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DEVICE = 'cuda'
WORKDIR = Path(__file__).resolve().parent.parent


# ============================================================
# Quantization Functions
# ============================================================

def quantize_uniform(t, bits):
    """Standard uniform quantization."""
    if bits >= 16: return t
    mx = t.abs().max()
    if mx == 0: return t
    s = mx / (2 ** (bits - 1) - 1)
    return (t / s).round().clamp(-(2**(bits-1)), 2**(bits-1)-1) * s


def quantize_log_magnitude(mag, bits):
    """Log-scale magnitude quantization for decay rates.
    ADT values are negative (log-space decay), typically in [-8, 0].
    Higher resolution near 0 (slow decay = long memory).
    """
    if bits >= 16: return mag
    # ADT is negative: clamp and quantize
    n_levels = 2 ** bits
    mx = mag.abs().max()
    if mx == 0: return mag
    # Log-scale grid: more resolution near 0
    grid = torch.linspace(0, 1, n_levels, device=mag.device) ** 2 * mx
    grid = torch.cat([-grid.flip(0)[:-1], grid])  # symmetric

    flat = mag.flatten()
    diffs = (flat.unsqueeze(-1) - grid.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)
    return grid[indices].reshape(mag.shape)


def quantize_phase_uniform(angles, bits):
    """Uniform phase quantization in [-π, π) or arbitrary range."""
    if bits >= 16: return angles
    n_levels = 2 ** bits
    # Determine range from data
    mn, mx = angles.min(), angles.max()
    if mn == mx: return angles
    step = (mx - mn) / n_levels
    return ((angles - mn) / step).round() * step + mn


def quantize_polar_adt_angles(ADT, Angles, bits_mag, bits_phase):
    """Polar quantization: ADT (magnitude) with log-scale, Angles (phase) with uniform.
    Stability: ensure ADT stays negative (decay) after quantization.
    """
    ADT_q = quantize_log_magnitude(ADT, bits_mag)
    # Ensure stability: ADT must remain negative (or <= 0)
    ADT_q = ADT_q.clamp(max=0)
    Angles_q = quantize_phase_uniform(Angles, bits_phase)
    return ADT_q, Angles_q


def quantize_cartesian_adt_angles(ADT, Angles, bits):
    """Cartesian quantization: treat ADT and Angles as independent real tensors."""
    ADT_q = quantize_uniform(ADT, bits)
    Angles_q = quantize_uniform(Angles, bits)
    return ADT_q, Angles_q


# ============================================================
# Monkey-patching Mamba3 forward
# ============================================================

def make_patched_forward(original_forward, quant_method, bits, bits_phase=None):
    """Create a patched forward that quantizes ADT and Angles before kernel call."""

    def patched_forward(self_module, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        batch, seqlen, dim = u.shape

        angle_dt_state, ssm_state, k_state, v_state = None, None, None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            angle_dt_state, ssm_state, k_state, v_state = self_module._get_states_from_cache(
                inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _, _, _ = self_module.step(u, angle_dt_state, ssm_state, k_state, v_state)
                return out

        # in_proj
        zxBCdtAtrap = self_module.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [self_module.d_inner, self_module.d_inner,
             self_module.d_state * self_module.num_bc_heads * self_module.mimo_rank,
             self_module.d_state * self_module.num_bc_heads * self_module.mimo_rank,
             self_module.nheads, self_module.nheads, self_module.nheads,
             self_module.num_rope_angles],
            dim=-1)

        z = rearrange(z, "b l (h p) -> b l h p", p=self_module.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self_module.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n",
                      r=self_module.mimo_rank, g=self_module.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n",
                      r=self_module.mimo_rank, g=self_module.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT, DT — THIS IS WHERE THE COMPLEX STATE IS CONSTRUCTED
        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self_module.A_floor)
        DT = F.softplus(dd_dt + self_module.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Angles
        angles = angles.unsqueeze(-2).expand(-1, -1, self_module.nheads, -1)

        # ======= QUANTIZATION INJECTION POINT =======
        if quant_method == "cartesian":
            ADT, angles = quantize_cartesian_adt_angles(ADT, angles, bits)
            DT = quantize_uniform(DT, bits)
        elif quant_method == "polar":
            bp = bits_phase if bits_phase is not None else bits
            ADT, angles = quantize_polar_adt_angles(ADT, angles, bits, bp)
            DT = quantize_uniform(DT, bits)
        elif quant_method == "polar_stable":
            bp = bits_phase if bits_phase is not None else bits
            ADT_q = quantize_log_magnitude(ADT, bits)
            ADT_q = ADT_q.clamp(max=-self_module.A_floor)  # Ensure stability
            angles_q = quantize_phase_uniform(angles, bp)
            ADT = ADT_q
            angles = angles_q
            DT = quantize_uniform(DT, bits)
        # =============================================

        # Apply B/C norm
        B = self_module.B_norm(B)
        C = self_module.C_norm(C)

        # Call kernel
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
        y = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x,
            ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self_module.C_bias.squeeze(1),
            K_bias=self_module.B_bias.squeeze(1),
            Angles=angles,
            D=self_module.D,
            Z=z if not self_module.is_outproj_norm else None,
            chunk_size=self_module.chunk_size,
            Input_States=None,
            return_final_states=ssm_state is not None,
            cu_seqlens=cu_seqlens,
        )
        if ssm_state is not None:
            y, last_angle, last_state, last_k, last_v, *rest = y
            angle_dt_state.copy_(last_angle)
            ssm_state.copy_(last_state)
            k_state.copy_(last_k.unsqueeze(1))
            v_state.copy_(last_v)
        y = rearrange(y, "b l h p -> b l (h p)")
        if self_module.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self_module.norm(y, z)

        out = self_module.out_proj(y.to(x.dtype))
        return out

    return patched_forward


# ============================================================
# Evaluation
# ============================================================

def get_calib_data(n_samples=64, seq_len=2048):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    all_tokens = []
    for sample in ds:
        all_tokens.extend(tokenizer.encode(sample["text"]))
        if len(all_tokens) >= n_samples * (seq_len + 1):
            break
    batches = []
    for i in range(n_samples):
        start = i * seq_len
        chunk = all_tokens[start:start + seq_len]
        if len(chunk) == seq_len:
            batches.append(torch.tensor(chunk, dtype=torch.long))
    print(f"Calibration: {len(batches)} sequences × {seq_len} tokens")
    return batches


@torch.no_grad()
def eval_perplexity(model, calib_data, max_batches=32):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for i, tokens in enumerate(calib_data):
        if i >= max_batches: break
        input_ids = tokens[:-1].unsqueeze(0).to(DEVICE)  # input: [0..N-2]
        targets = tokens[1:].unsqueeze(0).to(DEVICE)      # target: [1..N-1] (shifted)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if not torch.isnan(loss):
            total_loss += loss.item() * (tokens.size(0) - 1)
            total_tokens += tokens.size(0) - 1
    avg_loss = total_loss / max(total_tokens, 1)
    return round(math.exp(min(avg_loss, 20)), 4)


def apply_ssm_quantization(model, quant_method, bits, bits_phase=None):
    """Monkey-patch all Mamba3 blocks to quantize ADT and Angles."""
    from mamba_ssm import Mamba3
    patched = make_patched_forward(None, quant_method, bits, bits_phase)

    for name, module in model.named_modules():
        if isinstance(module, Mamba3):
            # Bind patched forward as a method
            import types
            module.forward = types.MethodType(patched, module)

    return model


# ============================================================
# Statistics Collection
# ============================================================

def collect_adt_angle_stats(model, calib_data):
    """Collect ADT and Angle distributions from forward passes."""
    from mamba_ssm import Mamba3
    stats = {"ADT": [], "Angles": [], "DT": []}
    hooks = []

    def make_hook(name):
        def hook_fn(module, args, output):
            # Intercept in_proj output to extract ADT and Angles
            pass  # We'll collect in a different way
        return hook_fn

    # Temporarily patch to collect stats
    collected = {"ADT": [], "Angles": []}

    original_forwards = {}
    for name, module in model.named_modules():
        if isinstance(module, Mamba3):
            original_forwards[name] = module.forward

            def make_collector(mod):
                orig = mod.forward
                def collecting_forward(u, **kwargs):
                    batch, seqlen, dim = u.shape
                    zxBCdtAtrap = mod.in_proj(u)
                    _, _, _, _, dd_dt, dd_A, _, angles = torch.split(
                        zxBCdtAtrap,
                        [mod.d_inner, mod.d_inner,
                         mod.d_state * mod.num_bc_heads * mod.mimo_rank,
                         mod.d_state * mod.num_bc_heads * mod.mimo_rank,
                         mod.nheads, mod.nheads, mod.nheads,
                         mod.num_rope_angles], dim=-1)
                    _A = -F.softplus(dd_A.to(torch.float32))
                    _A = torch.clamp(_A, max=-mod.A_floor)
                    DT = F.softplus(dd_dt + mod.dt_bias)
                    ADT = _A * DT
                    collected["ADT"].append(ADT.detach().cpu())
                    collected["Angles"].append(angles.detach().cpu())
                    return orig(u, **kwargs)
                return collecting_forward

            module.forward = make_collector(module)

    with torch.no_grad():
        for i, tokens in enumerate(calib_data[:16]):
            input_ids = tokens.unsqueeze(0).to(DEVICE)
            model(input_ids, input_ids.clone())

    # Restore
    for name, module in model.named_modules():
        if name in original_forwards:
            module.forward = original_forwards[name]

    # Analyze
    if collected["ADT"]:
        all_adt = torch.cat(collected["ADT"]).numpy().flatten()
        all_angles = torch.cat(collected["Angles"]).numpy().flatten()
        stats = {
            "ADT_mean": round(float(all_adt.mean()), 6),
            "ADT_std": round(float(all_adt.std()), 6),
            "ADT_min": round(float(all_adt.min()), 6),
            "ADT_max": round(float(all_adt.max()), 6),
            "Angles_mean": round(float(all_angles.mean()), 6),
            "Angles_std": round(float(all_angles.std()), 6),
            "Angles_min": round(float(all_angles.min()), 6),
            "Angles_max": round(float(all_angles.max()), 6),
        }
        print(f"  ADT: mean={stats['ADT_mean']}, std={stats['ADT_std']}, "
              f"range=[{stats['ADT_min']}, {stats['ADT_max']}]")
        print(f"  Angles: mean={stats['Angles_mean']}, std={stats['Angles_std']}, "
              f"range=[{stats['Angles_min']}, {stats['Angles_max']}]")
        return stats
    return {}


# ============================================================
# Main Experiments
# ============================================================

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Phase 1 v2: Activation-level Polar vs Cartesian\n")

    # Load model
    from ppq.train_mamba3_180m import Mamba3LM, MODEL_CONFIGS
    config = MODEL_CONFIGS["180M"]
    config["vocab_size"] = 50257
    model = Mamba3LM(config).to(DEVICE)
    ckpt_path = WORKDIR / "ppq" / "checkpoints" / "mamba3_180M_final.pt"
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {ckpt_path.name}\n")

    # Calibration data
    calib_data = get_calib_data(n_samples=64, seq_len=config["seq_len"])

    results = {}

    # ============================================================
    # Step 0: Collect ADT and Angle distributions
    # ============================================================
    print(f"{'='*60}")
    print(f"Step 0: ADT & Angle Distribution Analysis")
    print(f"{'='*60}")
    dist_stats = collect_adt_angle_stats(model, calib_data)
    results["distributions"] = dist_stats

    # ============================================================
    # Step 1: Baseline perplexity
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Baseline Perplexity")
    print(f"{'='*60}")
    baseline_ppl = eval_perplexity(model, calib_data)
    print(f"  Baseline PPL: {baseline_ppl}")
    results["baseline_ppl"] = baseline_ppl

    # ============================================================
    # Step 2: Cartesian vs Polar at different bit widths
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Cartesian vs Polar — Activation Quantization")
    print(f"{'='*60}")

    for bits in [8, 6, 4, 3]:
        print(f"\n  --- {bits}-bit ---")

        for method in ["cartesian", "polar", "polar_stable"]:
            m = copy.deepcopy(model).to(DEVICE).eval()
            m = apply_ssm_quantization(m, method, bits, bits_phase=bits)

            ppl = eval_perplexity(m, calib_data)
            drop = ppl - baseline_ppl
            key = f"W{bits}_{method}"
            results[key] = {"ppl": ppl, "drop": round(drop, 4)}
            print(f"    {method:<15} PPL={ppl:.4f} (Δ={drop:+.4f})")
            del m; gc.collect(); torch.cuda.empty_cache()

    # ============================================================
    # Step 3: Go/No-Go Decision
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Go/No-Go Decision")
    print(f"{'='*60}")

    for bits in [8, 4]:
        cart = results.get(f"W{bits}_cartesian", {})
        polar = results.get(f"W{bits}_polar_stable", {})
        if cart and polar:
            advantage = cart.get("drop", 0) - polar.get("drop", 0)
            threshold = 0.5 if bits == 8 else 2.0
            # For PPL, lower is better, so positive advantage means polar is better
            decision = "GO ✓" if advantage > 0 else "INVESTIGATE"
            print(f"  W{bits}: Cart PPL={cart['ppl']}, Polar PPL={polar['ppl']}, "
                  f"Advantage={advantage:.4f} → {decision}")

    # Save
    out = WORKDIR / "ppq" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phase1_pilot_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out / 'phase1_pilot_v2.json'}")


if __name__ == "__main__":
    main()
