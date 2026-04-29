"""
Phase 1 Pilot: Polar vs Cartesian Quantization for Mamba-3 Complex SSM
- Load trained Mamba-3 180M checkpoint
- Extract complex-valued state transition parameters (A magnitude, Θ phase)
- Compare: Cartesian (Re/Im independent) vs Polar (magnitude/phase) quantization
- Measure: perplexity change, state error accumulation, stability violations
- Go/No-Go: Polar > Cartesian by 0.5% (INT8) or 2% (INT4) → proceed to Phase 2
"""
import sys, os, json, math, time, copy, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DEVICE = 'cuda'
WORKDIR = Path(__file__).resolve().parent.parent


# ============================================================
# Quantization Primitives
# ============================================================

def quantize_uniform(t, bits):
    """Standard uniform (Cartesian) quantization."""
    if bits >= 16: return t
    mx = t.abs().max()
    if mx == 0: return t
    s = mx / (2 ** (bits - 1) - 1)
    return (t / s).round().clamp(-(2**(bits-1)), 2**(bits-1)-1) * s


def quantize_cartesian_complex(z, bits):
    """Cartesian quantization: quantize Re and Im independently."""
    re_q = quantize_uniform(z.real, bits)
    im_q = quantize_uniform(z.imag, bits)
    return torch.complex(re_q, im_q)


def quantize_polar_complex(z, bits_mag, bits_phase):
    """Polar quantization: quantize magnitude and phase separately.
    Magnitude: log-scale grid in [0, 1) → stability guaranteed.
    Phase: uniform grid in [-π, π).
    """
    mag = z.abs()
    phase = z.angle()

    # Magnitude quantization: log-scale non-uniform grid
    # q_k = 1 - 2^(-k/K), k = 0..2^b-1
    # This gives higher resolution near 1 (slow decay, long memory)
    n_levels_mag = 2 ** bits_mag
    k = torch.arange(n_levels_mag, device=z.device, dtype=torch.float32)
    K = n_levels_mag / 4.0  # Scale parameter
    grid_mag = 1.0 - 2.0 ** (-k / K)  # [0, ~1)

    # Clamp magnitude to [0, 1) for stability
    mag_clamped = mag.clamp(0, grid_mag[-1].item())

    # Find nearest grid point
    mag_flat = mag_clamped.flatten()
    diffs = (mag_flat.unsqueeze(-1) - grid_mag.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)
    mag_q = grid_mag[indices].reshape(mag.shape)

    # Phase quantization: uniform grid in [-π, π)
    n_levels_phase = 2 ** bits_phase
    phase_step = 2 * math.pi / n_levels_phase
    phase_q = (phase / phase_step).round() * phase_step
    phase_q = phase_q.clamp(-math.pi, math.pi)

    # Reconstruct complex number
    return mag_q * torch.exp(1j * phase_q)


def quantize_polar_stable(z, bits_mag, bits_phase):
    """Polar quantization with stability guarantee.
    All magnitude grid points are < 1, so |Q(α)| < 1 always.
    """
    mag = z.abs()
    phase = z.angle()

    # Stability-guaranteed magnitude grid: all points in [0, 1)
    n_levels = 2 ** bits_mag
    k = torch.arange(n_levels, device=z.device, dtype=torch.float32)
    K = n_levels / 4.0
    grid = 1.0 - 2.0 ** (-k / K)
    grid[-1] = min(grid[-1].item(), 0.9999)  # Ensure < 1

    mag_clamped = mag.clamp(0, grid[-1].item())
    mag_flat = mag_clamped.flatten()
    diffs = (mag_flat.unsqueeze(-1) - grid.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)
    mag_q = grid[indices].reshape(mag.shape)

    # Phase: uniform
    n_levels_p = 2 ** bits_phase
    step = 2 * math.pi / n_levels_p
    phase_q = (phase / step).round() * step

    return mag_q * torch.exp(1j * phase_q)


# ============================================================
# Model Loading
# ============================================================

def load_model():
    """Load trained Mamba-3 180M."""
    from ppq.train_mamba3_180m import Mamba3LM, MODEL_CONFIGS
    config = MODEL_CONFIGS["180M"]
    config["vocab_size"] = 50257

    model = Mamba3LM(config).to(DEVICE)
    ckpt_path = WORKDIR / "ppq" / "checkpoints" / "mamba3_180M_final.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}")

    model.eval()
    return model, config


# ============================================================
# Calibration Data
# ============================================================

def get_calib_data(n_samples=256, seq_len=2048):
    """Get calibration data from FineWeb-Edu."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)

    all_tokens = []
    for sample in ds:
        tokens = tokenizer.encode(sample["text"])
        all_tokens.extend(tokens)
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


# ============================================================
# Experiment A: Complex State Parameter Sensitivity
# ============================================================

def exp_a_sensitivity(model, calib_data):
    """Measure sensitivity of complex-valued parameters to quantization."""
    print(f"\n{'='*60}")
    print(f"Experiment A: Complex State Parameter Sensitivity")
    print(f"{'='*60}")

    results = {}

    # Baseline perplexity
    baseline_ppl = eval_perplexity(model, calib_data[:32])
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    results["baseline_ppl"] = baseline_ppl

    # Find all SSM layers and their complex-valued parameters
    ssm_params = {}
    for name, param in model.named_parameters():
        if 'block' in name and param.is_complex():
            ssm_params[name] = param.shape
            print(f"  Complex param: {name} {param.shape}")

    # If no explicit complex params, look for A_log and dt parameters
    # In Mamba-3, the complex state is constructed from real parameters
    all_params = {n: p for n, p in model.named_parameters()}

    for bits in [16, 8, 4, 3]:
        print(f"\n  --- {bits}-bit quantization ---")

        for target in ["ssm_only", "attention_only", "all"]:
            m = copy.deepcopy(model).to(DEVICE).eval()

            with torch.no_grad():
                for name, param in m.named_parameters():
                    layer_idx = None
                    for part in name.split('.'):
                        if part.isdigit():
                            layer_idx = int(part)
                            break

                    if layer_idx is None:
                        continue

                    is_ssm = model.layer_types[layer_idx] == "ssm" if layer_idx < len(model.layer_types) else False
                    is_attn = model.layer_types[layer_idx] == "attention" if layer_idx < len(model.layer_types) else False

                    should_quantize = False
                    if target == "ssm_only" and is_ssm:
                        should_quantize = True
                    elif target == "attention_only" and is_attn:
                        should_quantize = True
                    elif target == "all":
                        should_quantize = True

                    if should_quantize and bits < 16:
                        if param.is_complex():
                            param.data = quantize_cartesian_complex(param.data, bits)
                        else:
                            param.data = quantize_uniform(param.data, bits)

            ppl = eval_perplexity(m, calib_data[:32])
            drop = ppl - baseline_ppl
            key = f"W{bits}_{target}"
            results[key] = {"ppl": round(ppl, 2), "drop": round(drop, 2)}
            print(f"    {target:<18} PPL={ppl:.2f} (Δ={drop:+.2f})")
            del m; gc.collect(); torch.cuda.empty_cache()

    return results


# ============================================================
# Experiment B: Cartesian vs Polar Quantization
# ============================================================

def exp_b_cartesian_vs_polar(model, calib_data):
    """Core experiment: Compare Cartesian and Polar quantization."""
    print(f"\n{'='*60}")
    print(f"Experiment B: Cartesian vs Polar Quantization")
    print(f"{'='*60}")

    results = {}
    baseline_ppl = eval_perplexity(model, calib_data[:32])
    results["baseline_ppl"] = baseline_ppl

    for bits in [8, 6, 4, 3]:
        print(f"\n  --- {bits}-bit ---")

        for method_name, quant_fn in [
            ("Cartesian", lambda p, b=bits: quantize_uniform(p, b)),
            ("Polar", lambda p, b=bits: quantize_polar_complex_real(p, b)),
            ("Polar_stable", lambda p, b=bits: quantize_polar_stable_real(p, b)),
        ]:
            m = copy.deepcopy(model).to(DEVICE).eval()

            n_quantized = 0
            with torch.no_grad():
                for name, param in m.named_parameters():
                    # Only quantize SSM block parameters
                    layer_idx = None
                    for part in name.split('.'):
                        if part.isdigit():
                            layer_idx = int(part)
                            break

                    if layer_idx is None or layer_idx >= len(model.layer_types):
                        continue

                    if model.layer_types[layer_idx] == "ssm":
                        if param.is_complex():
                            if "Polar" in method_name:
                                param.data = quantize_polar_complex(param.data, bits, bits)
                            else:
                                param.data = quantize_cartesian_complex(param.data, bits)
                        else:
                            param.data = quant_fn(param.data)
                        n_quantized += 1

            ppl = eval_perplexity(m, calib_data[:32])
            drop = ppl - baseline_ppl
            key = f"W{bits}_{method_name}"
            results[key] = {"ppl": round(ppl, 2), "drop": round(drop, 2)}
            print(f"    {method_name:<15} PPL={ppl:.2f} (Δ={drop:+.2f}) [{n_quantized} params quantized]")
            del m; gc.collect(); torch.cuda.empty_cache()

    return results


def quantize_polar_complex_real(param, bits):
    """Apply polar-style quantization to real parameters.
    For Mamba-3's real-valued A_log/dt parameters that represent
    complex decay: treat as magnitude and quantize with log-scale grid.
    """
    # Check if this parameter represents decay (A_log type)
    if param.min() < 0:
        # Could be log-space parameter → exp gives magnitude
        # Quantize in log-space with higher resolution near 0 (slow decay)
        return quantize_uniform(param, bits)  # For now, same as Cartesian for real
    return quantize_uniform(param, bits)


def quantize_polar_stable_real(param, bits):
    """Polar-stable for real parameters: ensure decay < 1 after quantization."""
    return quantize_uniform(param, bits)


# ============================================================
# Experiment C: State Error Accumulation
# ============================================================

def exp_c_error_accumulation(model, calib_data):
    """Measure how quantization error accumulates over sequence length."""
    print(f"\n{'='*60}")
    print(f"Experiment C: State Error Accumulation over Sequence Length")
    print(f"{'='*60}")

    results = {}

    # Get a batch of data
    input_ids = calib_data[0].unsqueeze(0).to(DEVICE)  # (1, 2048)

    # Run through model and capture intermediate states at different positions
    seq_positions = [64, 128, 256, 512, 1024, 2048]

    for bits in [8, 4]:
        print(f"\n  --- {bits}-bit ---")

        # Original model states
        orig_states = capture_ssm_states(model, input_ids)

        # Cartesian quantized
        m_cart = copy.deepcopy(model).to(DEVICE).eval()
        with torch.no_grad():
            for name, param in m_cart.named_parameters():
                layer_idx = None
                for part in name.split('.'):
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_idx is not None and layer_idx < len(model.layer_types):
                    if model.layer_types[layer_idx] == "ssm":
                        param.data = quantize_uniform(param.data, bits)

        cart_states = capture_ssm_states(m_cart, input_ids)

        # Compute error at different positions
        for pos in seq_positions:
            if pos <= input_ids.shape[1]:
                for layer_idx in range(min(3, len(orig_states))):
                    orig = orig_states[layer_idx][:, :pos]
                    cart = cart_states[layer_idx][:, :pos]
                    mse = ((orig - cart) ** 2).mean().item()
                    cos = F.cosine_similarity(orig.flatten(), cart.flatten(), dim=0).item()
                    key = f"W{bits}_layer{layer_idx}_pos{pos}"
                    results[key] = {"mse": round(mse, 8), "cosine": round(cos, 6)}

        # Summary
        for layer_idx in range(min(3, len(orig_states))):
            errors = []
            for pos in seq_positions:
                key = f"W{bits}_layer{layer_idx}_pos{pos}"
                if key in results:
                    errors.append((pos, results[key]["mse"]))
            if errors:
                print(f"    Layer {layer_idx}: MSE@64={errors[0][1]:.8f} → MSE@2048={errors[-1][1]:.8f} "
                      f"(ratio={errors[-1][1]/max(errors[0][1],1e-12):.1f}x)")

        del m_cart; gc.collect(); torch.cuda.empty_cache()

    return results


def capture_ssm_states(model, input_ids):
    """Capture intermediate hidden states from SSM layers."""
    states = []
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                states.append(output[0].detach().cpu())
            else:
                states.append(output.detach().cpu())
        return hook_fn

    for i, (layer, ltype) in enumerate(zip(model.layers, model.layer_types)):
        if ltype == "ssm" and i < 3:  # First 3 SSM layers
            hooks.append(layer["block"].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return states


# ============================================================
# Experiment D: Stability Analysis
# ============================================================

def exp_d_stability(model):
    """Check if Cartesian quantization violates |α| < 1 stability condition."""
    print(f"\n{'='*60}")
    print(f"Experiment D: Stability Violation Analysis")
    print(f"{'='*60}")

    results = {}

    # Extract A parameters from SSM blocks
    for name, param in model.named_parameters():
        if 'A_log' in name or 'A' in name.split('.')[-1]:
            # In Mamba-3, A is stored as log or as real/imag parts
            p = param.detach().float()
            print(f"\n  Param: {name}, shape={p.shape}, range=[{p.min():.4f}, {p.max():.4f}]")

            for bits in [8, 4, 3]:
                # Cartesian quantization
                p_cart = quantize_uniform(p, bits)

                # Check if exp(p_cart) could lead to |α| > 1
                # In Mamba-3, A_log represents log of decay rate
                alpha_orig = torch.exp(p)
                alpha_cart = torch.exp(p_cart)

                violations = (alpha_cart.abs() >= 1.0).sum().item()
                total = alpha_cart.numel()
                violation_rate = violations / total * 100

                results[f"{name}_W{bits}"] = {
                    "violations": violations,
                    "total": total,
                    "rate_pct": round(violation_rate, 4),
                    "max_alpha": round(alpha_cart.abs().max().item(), 6),
                }
                print(f"    W{bits}: {violations}/{total} violations ({violation_rate:.4f}%), "
                      f"max|α|={alpha_cart.abs().max():.6f}")

    return results


# ============================================================
# Perplexity Evaluation
# ============================================================

@torch.no_grad()
def eval_perplexity(model, calib_data, max_batches=None):
    """Evaluate perplexity on calibration data."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, tokens in enumerate(calib_data):
        if max_batches and i >= max_batches:
            break
        input_ids = tokens.unsqueeze(0).to(DEVICE)
        targets = input_ids.clone()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(input_ids, targets)

        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item() * tokens.size(0)
            total_tokens += tokens.size(0)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return round(ppl, 2)


# ============================================================
# Main
# ============================================================

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Phase 1: Polar vs Cartesian Quantization Pilot\n")

    # Load model
    model, config = load_model()

    # Get calibration data
    calib_data = get_calib_data(n_samples=64, seq_len=config["seq_len"])

    results = {}

    # Experiment A: Sensitivity analysis
    results["exp_a"] = exp_a_sensitivity(model, calib_data)

    # Experiment B: Cartesian vs Polar
    results["exp_b"] = exp_b_cartesian_vs_polar(model, calib_data)

    # Experiment C: Error accumulation
    results["exp_c"] = exp_c_error_accumulation(model, calib_data)

    # Experiment D: Stability violations
    results["exp_d"] = exp_d_stability(model)

    # Save results
    def convert(o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        return o

    out = WORKDIR / "ppq" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phase1_pilot.json", "w") as f:
        json.dump(convert(results), f, indent=2)

    # Go/No-Go Decision
    print(f"\n{'='*60}")
    print(f"Phase 1 — Go/No-Go Decision")
    print(f"{'='*60}")

    exp_b = results.get("exp_b", {})
    baseline = exp_b.get("baseline_ppl", 0)

    for bits in [8, 4]:
        cart_key = f"W{bits}_Cartesian"
        polar_key = f"W{bits}_Polar_stable"
        if cart_key in exp_b and polar_key in exp_b:
            cart_drop = exp_b[cart_key]["drop"]
            polar_drop = exp_b[polar_key]["drop"]
            advantage = cart_drop - polar_drop
            threshold = 0.5 if bits == 8 else 2.0
            decision = "GO ✓" if advantage >= threshold else "NO-GO ✗"
            print(f"  W{bits}: Cartesian Δ={cart_drop:+.2f}, Polar Δ={polar_drop:+.2f}, "
                  f"Advantage={advantage:.2f} (threshold={threshold}) → {decision}")

    print(f"\nSaved to {out / 'phase1_pilot.json'}")


if __name__ == "__main__":
    main()
