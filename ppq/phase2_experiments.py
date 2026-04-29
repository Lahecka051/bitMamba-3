"""
Phase 2: Core Experiments — Scale Comparison + Component Ablation + SSM Sensitivity
====================================================================================
Run after both 180M and 440M models are trained.

Experiments:
  2-1. Scale comparison: 180M vs 440M × {Cartesian, Polar, Polar_stable} × {8,6,4,3 bit}
  2-2. Component ablation: SSM-only / Attention-only / All quantization
  2-3. SSM internal sensitivity: A_log/dt vs B/C vs in/out_proj vs FFN
  2-4. Stability violation analysis: |α| > 1 check per method
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

# Import from phase1_pilot_v2
from ppq.phase1_pilot_v2 import (
    quantize_uniform, quantize_log_magnitude, quantize_phase_uniform,
    quantize_polar_adt_angles, quantize_cartesian_adt_angles,
    apply_ssm_quantization, get_calib_data, eval_perplexity,
)


def load_model(size="180M"):
    from ppq.train_mamba3_180m import Mamba3LM, MODEL_CONFIGS
    config = MODEL_CONFIGS[size]
    config["vocab_size"] = 50257
    model = Mamba3LM(config).to(DEVICE)
    ckpt_path = WORKDIR / "ppq" / "checkpoints" / f"mamba3_{size}_final.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {size}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    else:
        print(f"WARNING: checkpoint not found: {ckpt_path}")
    model.eval()
    return model, config


# ============================================================
# Experiment 2-1: Scale Comparison
# ============================================================
def exp_2_1_scale_comparison(calib_data):
    """Compare 180M vs 440M across quantization methods and bit widths."""
    print(f"\n{'#'*60}")
    print(f"# Exp 2-1: Scale Comparison (180M vs 440M)")
    print(f"{'#'*60}")

    results = {}

    for size in ["180M", "440M"]:
        ckpt = WORKDIR / "ppq" / "checkpoints" / f"mamba3_{size}_final.pt"
        if not ckpt.exists():
            print(f"  {size}: checkpoint not found, skipping")
            continue

        model, config = load_model(size)
        baseline = eval_perplexity(model, calib_data)
        print(f"\n  {size} Baseline PPL: {baseline}")
        results[size] = {"baseline": baseline}

        for bits in [8, 6, 4, 3]:
            for method in ["cartesian", "polar", "polar_stable"]:
                m = copy.deepcopy(model).to(DEVICE).eval()
                m = apply_ssm_quantization(m, method, bits)
                ppl = eval_perplexity(m, calib_data)
                drop = ppl - baseline
                key = f"W{bits}_{method}"
                results[size][key] = {"ppl": ppl, "drop": round(drop, 4)}
                print(f"    {size} {key}: PPL={ppl:.4f} (Δ={drop:+.4f})")
                del m; gc.collect(); torch.cuda.empty_cache()

        del model; gc.collect(); torch.cuda.empty_cache()

    return results


# ============================================================
# Experiment 2-2: Component Ablation (SSM vs Attention)
# ============================================================
def exp_2_2_component_ablation(calib_data):
    """SSM-only vs Attention-only vs All quantization."""
    print(f"\n{'#'*60}")
    print(f"# Exp 2-2: Component Ablation")
    print(f"{'#'*60}")

    results = {}

    for size in ["180M"]:
        model, config = load_model(size)
        baseline = eval_perplexity(model, calib_data)
        print(f"\n  {size} Baseline: {baseline}")
        results[size] = {"baseline": baseline}

        for bits in [8, 4]:
            for target in ["ssm_only", "attention_only", "all"]:
                m = copy.deepcopy(model).to(DEVICE).eval()

                with torch.no_grad():
                    for name, param in m.named_parameters():
                        layer_idx = None
                        for part in name.split('.'):
                            if part.isdigit():
                                layer_idx = int(part)
                                break
                        if layer_idx is None or layer_idx >= len(model.layer_types):
                            continue

                        is_ssm = model.layer_types[layer_idx] == "ssm"
                        is_attn = model.layer_types[layer_idx] == "attention"

                        should_q = False
                        if target == "ssm_only" and is_ssm: should_q = True
                        elif target == "attention_only" and is_attn: should_q = True
                        elif target == "all": should_q = True

                        if should_q:
                            param.data = quantize_uniform(param.data, bits)

                # Also apply activation quantization for SSM if target includes SSM
                if target in ["ssm_only", "all"]:
                    m = apply_ssm_quantization(m, "polar_stable", bits)

                ppl = eval_perplexity(m, calib_data)
                key = f"W{bits}_{target}"
                results[size][key] = {"ppl": ppl, "drop": round(ppl - baseline, 4)}
                print(f"    {key}: PPL={ppl:.4f}")
                del m; gc.collect(); torch.cuda.empty_cache()

        del model; gc.collect(); torch.cuda.empty_cache()

    return results


# ============================================================
# Experiment 2-3: SSM Internal Component Sensitivity
# ============================================================
def exp_2_3_ssm_sensitivity(calib_data):
    """Measure sensitivity of each SSM sub-component."""
    print(f"\n{'#'*60}")
    print(f"# Exp 2-3: SSM Internal Component Sensitivity")
    print(f"{'#'*60}")

    results = {}
    model, config = load_model("180M")
    baseline = eval_perplexity(model, calib_data)
    print(f"  Baseline: {baseline}")
    results["baseline"] = baseline

    # Categorize parameters
    groups = {
        "state_transition": ["dt_bias", "A_floor"],  # dt_bias, A-related
        "BC_proj": ["B_bias", "C_bias", "B_norm", "C_norm"],
        "io_proj": ["in_proj", "out_proj"],
        "ffn": ["ffn"],
    }

    for bits in [8, 4]:
        print(f"\n  --- {bits}-bit ---")
        for group_name, keywords in groups.items():
            m = copy.deepcopy(model).to(DEVICE).eval()
            n_quantized = 0

            with torch.no_grad():
                for name, param in m.named_parameters():
                    # Check if parameter belongs to SSM layer
                    layer_idx = None
                    for part in name.split('.'):
                        if part.isdigit():
                            layer_idx = int(part)
                            break
                    if layer_idx is None or layer_idx >= len(model.layer_types):
                        continue
                    if model.layer_types[layer_idx] != "ssm":
                        continue

                    # Check if matches group keywords
                    if any(kw in name for kw in keywords):
                        param.data = quantize_uniform(param.data, bits)
                        n_quantized += 1

            ppl = eval_perplexity(m, calib_data)
            key = f"W{bits}_{group_name}"
            results[key] = {"ppl": ppl, "drop": round(ppl - baseline, 4), "n_params": n_quantized}
            print(f"    {group_name:<20} PPL={ppl:.4f} (Δ={ppl-baseline:+.4f}) [{n_quantized} params]")
            del m; gc.collect(); torch.cuda.empty_cache()

        # Also test activation-only (ADT + Angles) without weight quantization
        for method in ["cartesian", "polar_stable"]:
            m = copy.deepcopy(model).to(DEVICE).eval()
            m = apply_ssm_quantization(m, method, bits)
            ppl = eval_perplexity(m, calib_data)
            key = f"A{bits}_{method}"
            results[key] = {"ppl": ppl, "drop": round(ppl - baseline, 4)}
            print(f"    activation_{method:<12} PPL={ppl:.4f} (Δ={ppl-baseline:+.4f})")
            del m; gc.collect(); torch.cuda.empty_cache()

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Experiment 2-4: Stability Violation Analysis
# ============================================================
def exp_2_4_stability(calib_data):
    """Check stability violations: does |quantized α| exceed 1?"""
    print(f"\n{'#'*60}")
    print(f"# Exp 2-4: Stability Violation Analysis")
    print(f"{'#'*60}")

    from mamba_ssm import Mamba3
    results = {}
    model, config = load_model("180M")

    # Collect ADT values during forward
    adt_values = []

    original_forwards = {}
    for name, module in model.named_modules():
        if isinstance(module, Mamba3):
            orig = module.forward
            original_forwards[name] = orig

            def make_collector(mod, orig_fn):
                def collecting_forward(u, **kwargs):
                    batch, seqlen, dim = u.shape
                    zxBCdtAtrap = mod.in_proj(u)
                    splits = torch.split(zxBCdtAtrap,
                        [mod.d_inner, mod.d_inner,
                         mod.d_state * mod.num_bc_heads * mod.mimo_rank,
                         mod.d_state * mod.num_bc_heads * mod.mimo_rank,
                         mod.nheads, mod.nheads, mod.nheads,
                         mod.num_rope_angles], dim=-1)
                    dd_dt, dd_A = splits[4], splits[5]
                    _A = -F.softplus(dd_A.to(torch.float32))
                    _A = torch.clamp(_A, max=-mod.A_floor)
                    DT = F.softplus(dd_dt + mod.dt_bias)
                    ADT = _A * DT
                    adt_values.append(ADT.detach().cpu())
                    return orig_fn(u, **kwargs)
                return collecting_forward

            module.forward = make_collector(module, orig)

    with torch.no_grad():
        for i, tokens in enumerate(calib_data[:8]):
            model(tokens.unsqueeze(0).to(DEVICE))

    # Restore
    for name, module in model.named_modules():
        if name in original_forwards:
            module.forward = original_forwards[name]

    all_adt = torch.cat(adt_values)
    print(f"  ADT statistics:")
    print(f"    Shape: {all_adt.shape}")
    print(f"    Range: [{all_adt.min():.6f}, {all_adt.max():.6f}]")
    print(f"    Mean: {all_adt.mean():.6f}")

    # exp(ADT) = |α| — check if > 1
    alpha_mag = torch.exp(all_adt)
    violations = (alpha_mag >= 1.0).sum().item()
    total = alpha_mag.numel()
    print(f"    |α| >= 1 violations: {violations}/{total} ({violations/total*100:.4f}%)")
    print(f"    max|α|: {alpha_mag.max():.6f}")

    results["original"] = {
        "violations": violations,
        "total": total,
        "rate_pct": round(violations / total * 100, 4),
        "max_alpha": round(alpha_mag.max().item(), 6),
        "adt_range": [round(all_adt.min().item(), 6), round(all_adt.max().item(), 6)],
    }

    # Check after quantization
    for bits in [8, 4, 3]:
        for method in ["cartesian", "polar_stable"]:
            adt_flat = all_adt.flatten()
            if method == "cartesian":
                adt_q = quantize_uniform(adt_flat, bits)
            else:
                adt_q = quantize_log_magnitude(adt_flat, bits)
                adt_q = adt_q.clamp(max=0)  # Stability guarantee

            alpha_q = torch.exp(adt_q)
            v = (alpha_q >= 1.0).sum().item()
            key = f"W{bits}_{method}"
            results[key] = {
                "violations": v,
                "rate_pct": round(v / total * 100, 4),
                "max_alpha": round(alpha_q.max().item(), 6),
            }
            status = "STABLE ✓" if v == 0 else f"UNSTABLE ({v} violations)"
            print(f"    {key}: {status}, max|α|={alpha_q.max():.6f}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================
def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Phase 2: Core Experiments\n")

    calib_data = get_calib_data(n_samples=64, seq_len=2048)

    results = {}

    # Run experiments
    results["exp_2_1"] = exp_2_1_scale_comparison(calib_data)
    results["exp_2_2"] = exp_2_2_component_ablation(calib_data)
    results["exp_2_3"] = exp_2_3_ssm_sensitivity(calib_data)
    results["exp_2_4"] = exp_2_4_stability(calib_data)

    # Save
    def convert(o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list): return [convert(v) for v in o]
        return o

    out = WORKDIR / "ppq" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phase2_experiments.json", "w") as f:
        json.dump(convert(results), f, indent=2)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"Phase 2 Summary")
    print(f"{'='*60}")

    # Scale comparison table
    if "exp_2_1" in results:
        print(f"\nScale Comparison:")
        print(f"  {'Config':<25} {'180M PPL':>12} {'440M PPL':>12}")
        print(f"  {'-'*50}")
        for key in ["baseline", "W8_polar_stable", "W6_polar_stable",
                     "W4_polar_stable", "W3_polar_stable",
                     "W8_cartesian", "W4_cartesian"]:
            r180 = results["exp_2_1"].get("180M", {}).get(key, {})
            r440 = results["exp_2_1"].get("440M", {}).get(key, {})
            v180 = r180 if isinstance(r180, (int, float)) else r180.get("ppl", "-")
            v440 = r440 if isinstance(r440, (int, float)) else r440.get("ppl", "-")
            print(f"  {key:<25} {str(v180):>12} {str(v440):>12}")

    print(f"\nSaved to {out / 'phase2_experiments.json'}")


if __name__ == "__main__":
    main()
