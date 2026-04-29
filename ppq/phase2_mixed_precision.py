"""
Phase 2: Mixed-Precision & Asymmetric Bit Allocation for Mamba-3 SSM
=====================================================================
Phase 1 v2 showed polar quantization is near-lossless at W8/W6 but
W4 PPL degrades to 61.9 (baseline 1.0036). This phase closes the gap.

Three experiments:
  A) Asymmetric bits: ADT and Angles may need different precision
  B) Per-layer sensitivity: Which layers are most sensitive to quantization?
  C) Mixed-precision: Assign bits per-layer based on sensitivity ranking
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
# Quantization Functions (from phase1_pilot_v2)
# ============================================================

def quantize_uniform(t, bits):
    if bits >= 16: return t
    mx = t.abs().max()
    if mx == 0: return t
    s = mx / (2 ** (bits - 1) - 1)
    return (t / s).round().clamp(-(2**(bits-1)), 2**(bits-1)-1) * s


def quantize_log_magnitude(mag, bits):
    if bits >= 16: return mag
    n_levels = 2 ** bits
    mx = mag.abs().max()
    if mx == 0: return mag
    grid = torch.linspace(0, 1, n_levels, device=mag.device) ** 2 * mx
    grid = torch.cat([-grid.flip(0)[:-1], grid])
    flat = mag.flatten()
    diffs = (flat.unsqueeze(-1) - grid.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)
    return grid[indices].reshape(mag.shape)


def quantize_phase_uniform(angles, bits):
    if bits >= 16: return angles
    n_levels = 2 ** bits
    mn, mx = angles.min(), angles.max()
    if mn == mx: return angles
    step = (mx - mn) / n_levels
    return ((angles - mn) / step).round() * step + mn


# ============================================================
# Per-Layer Quantization Patching
# ============================================================

def make_layer_patched_forward(quant_config):
    """Quantize ADT/Angles with per-layer config.
    quant_config: dict with 'adt_bits' and 'angle_bits' for this layer.
    """
    adt_bits = quant_config.get("adt_bits", 16)
    angle_bits = quant_config.get("angle_bits", 16)
    dt_bits = quant_config.get("dt_bits", 16)

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

        zxBCdtAtrap = self_module.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [self_module.d_inner, self_module.d_inner,
             self_module.d_state * self_module.num_bc_heads * self_module.mimo_rank,
             self_module.d_state * self_module.num_bc_heads * self_module.mimo_rank,
             self_module.nheads, self_module.nheads, self_module.nheads,
             self_module.num_rope_angles], dim=-1)

        z = rearrange(z, "b l (h p) -> b l h p", p=self_module.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self_module.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n",
                      r=self_module.mimo_rank, g=self_module.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n",
                      r=self_module.mimo_rank, g=self_module.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self_module.A_floor)
        DT = F.softplus(dd_dt + self_module.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        angles = angles.unsqueeze(-2).expand(-1, -1, self_module.nheads, -1)

        # === QUANTIZE with per-layer config ===
        ADT_q = quantize_log_magnitude(ADT, adt_bits)
        ADT_q = ADT_q.clamp(max=-self_module.A_floor)
        angles_q = quantize_phase_uniform(angles, angle_bits)
        DT_q = quantize_uniform(DT, dt_bits)
        ADT, angles, DT = ADT_q, angles_q, DT_q

        B = self_module.B_norm(B)
        C = self_module.C_norm(C)

        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
        y = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x,
            ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self_module.C_bias.squeeze(1),
            K_bias=self_module.B_bias.squeeze(1),
            Angles=angles, D=self_module.D,
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


def apply_per_layer_quantization(model, layer_configs):
    """Apply different quantization configs per SSM layer.
    layer_configs: list of dicts, one per SSM layer, with keys:
      'adt_bits', 'angle_bits', 'dt_bits'
    """
    from mamba_ssm import Mamba3
    import types

    ssm_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, Mamba3):
            if ssm_idx < len(layer_configs):
                config = layer_configs[ssm_idx]
            else:
                config = {"adt_bits": 16, "angle_bits": 16, "dt_bits": 16}
            patched = make_layer_patched_forward(config)
            module.forward = types.MethodType(patched, module)
            ssm_idx += 1

    return model


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
        input_ids = tokens[:-1].unsqueeze(0).to(DEVICE)
        targets = tokens[1:].unsqueeze(0).to(DEVICE)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if not torch.isnan(loss):
            total_loss += loss.item() * (tokens.size(0) - 1)
            total_tokens += tokens.size(0) - 1
    avg_loss = total_loss / max(total_tokens, 1)
    return round(math.exp(min(avg_loss, 20)), 4)


# ============================================================
# Experiment A: Asymmetric Bit Allocation
# ============================================================

def exp_a_asymmetric(model, calib_data, baseline_ppl):
    """Test different bit widths for ADT vs Angles independently."""
    print(f"\n{'='*60}")
    print(f"Exp A: Asymmetric Bit Allocation (ADT vs Angles)")
    print(f"{'='*60}")

    from mamba_ssm import Mamba3
    n_ssm = sum(1 for _, m in model.named_modules() if isinstance(m, Mamba3))

    results = {}
    # Test grid: ADT bits × Angle bits
    combos = [
        # (adt_bits, angle_bits, label)
        (8, 4, "ADT8_Ang4"),
        (6, 4, "ADT6_Ang4"),
        (4, 8, "ADT4_Ang8"),
        (4, 6, "ADT4_Ang6"),
        (6, 6, "ADT6_Ang6"),
        (8, 3, "ADT8_Ang3"),
        (6, 3, "ADT6_Ang3"),
        (4, 4, "ADT4_Ang4"),  # symmetric baseline
        (5, 5, "ADT5_Ang5"),
        (5, 4, "ADT5_Ang4"),
        (4, 5, "ADT4_Ang5"),
    ]

    for adt_b, ang_b, label in combos:
        avg_bits = (adt_b + ang_b) / 2
        config_list = [{"adt_bits": adt_b, "angle_bits": ang_b, "dt_bits": min(adt_b, ang_b)}
                       for _ in range(n_ssm)]
        m = copy.deepcopy(model).to(DEVICE).eval()
        m = apply_per_layer_quantization(m, config_list)
        ppl = eval_perplexity(m, calib_data)
        drop = ppl - baseline_ppl
        results[label] = {
            "adt_bits": adt_b, "angle_bits": ang_b,
            "avg_bits": avg_bits, "ppl": ppl, "drop": round(drop, 4)
        }
        print(f"  {label:<15} avg={avg_bits:.1f}b | PPL={ppl:.4f} (Δ={drop:+.4f})")
        del m; gc.collect(); torch.cuda.empty_cache()

    return results


# ============================================================
# Experiment B: Per-Layer Sensitivity
# ============================================================

def exp_b_sensitivity(model, calib_data, baseline_ppl):
    """Quantize one layer at a time to W4, measure PPL impact."""
    print(f"\n{'='*60}")
    print(f"Exp B: Per-Layer Sensitivity (W4 polar, one layer at a time)")
    print(f"{'='*60}")

    from mamba_ssm import Mamba3
    n_ssm = sum(1 for _, m in model.named_modules() if isinstance(m, Mamba3))
    print(f"  Total SSM layers: {n_ssm}")

    results = {}
    for target_layer in range(n_ssm):
        # All layers at W16 except target at W4
        config_list = []
        for i in range(n_ssm):
            if i == target_layer:
                config_list.append({"adt_bits": 4, "angle_bits": 4, "dt_bits": 4})
            else:
                config_list.append({"adt_bits": 16, "angle_bits": 16, "dt_bits": 16})

        m = copy.deepcopy(model).to(DEVICE).eval()
        m = apply_per_layer_quantization(m, config_list)
        ppl = eval_perplexity(m, calib_data)
        drop = ppl - baseline_ppl
        results[f"layer_{target_layer}"] = {"ppl": ppl, "drop": round(drop, 4)}
        print(f"  Layer {target_layer:>2d} → PPL={ppl:.4f} (Δ={drop:+.4f})")
        del m; gc.collect(); torch.cuda.empty_cache()

    # Rank by sensitivity
    ranked = sorted(results.items(), key=lambda x: x[1]["drop"], reverse=True)
    print(f"\n  Sensitivity ranking (most sensitive first):")
    for name, v in ranked[:5]:
        print(f"    {name}: Δ={v['drop']:+.4f}")

    results["ranking"] = [name for name, _ in ranked]
    return results


# ============================================================
# Experiment C: Mixed-Precision Assignment
# ============================================================

def exp_c_mixed_precision(model, calib_data, baseline_ppl, sensitivity_results):
    """Assign bits per-layer based on sensitivity: sensitive layers get more bits."""
    print(f"\n{'='*60}")
    print(f"Exp C: Mixed-Precision (sensitivity-guided)")
    print(f"{'='*60}")

    from mamba_ssm import Mamba3
    n_ssm = sum(1 for _, m in model.named_modules() if isinstance(m, Mamba3))

    # Get sensitivity ranking
    ranking = sensitivity_results.get("ranking", [])
    if not ranking:
        print("  ERROR: No sensitivity ranking available")
        return {}

    # Extract layer indices and their drops
    layer_drops = {}
    for name, v in sensitivity_results.items():
        if name.startswith("layer_"):
            idx = int(name.split("_")[1])
            layer_drops[idx] = v["drop"]

    results = {}

    # Strategy 1: Top-K sensitive at W8, rest at W4
    configs_to_test = [
        ("top2_W8_rest_W4", 2),
        ("top4_W8_rest_W4", 4),
        ("top6_W8_rest_W4", 6),
        ("top8_W8_rest_W4", 8),
        ("top10_W8_rest_W4", 10),
    ]

    sensitive_indices = [int(name.split("_")[1]) for name in ranking if name.startswith("layer_")]

    for label, top_k in configs_to_test:
        if top_k > len(sensitive_indices):
            continue
        sensitive_set = set(sensitive_indices[:top_k])
        config_list = []
        total_bits = 0
        for i in range(n_ssm):
            if i in sensitive_set:
                config_list.append({"adt_bits": 8, "angle_bits": 8, "dt_bits": 8})
                total_bits += 8
            else:
                config_list.append({"adt_bits": 4, "angle_bits": 4, "dt_bits": 4})
                total_bits += 4
        avg_bits = total_bits / n_ssm

        m = copy.deepcopy(model).to(DEVICE).eval()
        m = apply_per_layer_quantization(m, config_list)
        ppl = eval_perplexity(m, calib_data)
        drop = ppl - baseline_ppl
        results[label] = {"avg_bits": round(avg_bits, 2), "ppl": ppl, "drop": round(drop, 4)}
        print(f"  {label:<25} avg={avg_bits:.1f}b | PPL={ppl:.4f} (Δ={drop:+.4f})")
        del m; gc.collect(); torch.cuda.empty_cache()

    # Strategy 2: Gradual — sensitive at W6, rest at W4
    for label_prefix, high_bits in [("W6", 6), ("W5", 5)]:
        for top_k in [4, 8, 12]:
            label = f"top{top_k}_{label_prefix}_rest_W4"
            if top_k > len(sensitive_indices):
                continue
            sensitive_set = set(sensitive_indices[:top_k])
            config_list = []
            total_bits = 0
            for i in range(n_ssm):
                if i in sensitive_set:
                    config_list.append({"adt_bits": high_bits, "angle_bits": high_bits, "dt_bits": high_bits})
                    total_bits += high_bits
                else:
                    config_list.append({"adt_bits": 4, "angle_bits": 4, "dt_bits": 4})
                    total_bits += 4
            avg_bits = total_bits / n_ssm

            m = copy.deepcopy(model).to(DEVICE).eval()
            m = apply_per_layer_quantization(m, config_list)
            ppl = eval_perplexity(m, calib_data)
            drop = ppl - baseline_ppl
            results[label] = {"avg_bits": round(avg_bits, 2), "ppl": ppl, "drop": round(drop, 4)}
            print(f"  {label:<25} avg={avg_bits:.1f}b | PPL={ppl:.4f} (Δ={drop:+.4f})")
            del m; gc.collect(); torch.cuda.empty_cache()

    # Strategy 3: Asymmetric mixed — ADT gets more bits on sensitive layers
    print(f"\n  --- Asymmetric mixed-precision ---")
    for top_k in [4, 8]:
        label = f"top{top_k}_ADT8_Ang4_rest_W4"
        if top_k > len(sensitive_indices):
            continue
        sensitive_set = set(sensitive_indices[:top_k])
        config_list = []
        total_adt = 0
        total_ang = 0
        for i in range(n_ssm):
            if i in sensitive_set:
                config_list.append({"adt_bits": 8, "angle_bits": 4, "dt_bits": 6})
                total_adt += 8; total_ang += 4
            else:
                config_list.append({"adt_bits": 4, "angle_bits": 4, "dt_bits": 4})
                total_adt += 4; total_ang += 4
        avg_adt = total_adt / n_ssm
        avg_ang = total_ang / n_ssm

        m = copy.deepcopy(model).to(DEVICE).eval()
        m = apply_per_layer_quantization(m, config_list)
        ppl = eval_perplexity(m, calib_data)
        drop = ppl - baseline_ppl
        results[label] = {
            "avg_adt_bits": round(avg_adt, 2), "avg_ang_bits": round(avg_ang, 2),
            "ppl": ppl, "drop": round(drop, 4)
        }
        print(f"  {label:<35} ADT={avg_adt:.1f}b Ang={avg_ang:.1f}b | PPL={ppl:.4f} (Δ={drop:+.4f})")
        del m; gc.collect(); torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Phase 2: Mixed-Precision & Asymmetric Bit Allocation\n")

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

    # Baseline
    print(f"{'='*60}")
    print(f"Baseline Perplexity")
    print(f"{'='*60}")
    baseline_ppl = eval_perplexity(model, calib_data)
    print(f"  Baseline PPL: {baseline_ppl}")

    all_results = {"baseline_ppl": baseline_ppl}

    # Exp A: Asymmetric
    all_results["exp_a_asymmetric"] = exp_a_asymmetric(model, calib_data, baseline_ppl)

    # Exp B: Per-layer sensitivity
    all_results["exp_b_sensitivity"] = exp_b_sensitivity(model, calib_data, baseline_ppl)

    # Exp C: Mixed-precision
    all_results["exp_c_mixed_precision"] = exp_c_mixed_precision(
        model, calib_data, baseline_ppl, all_results["exp_b_sensitivity"])

    # ============================================================
    # Summary & Decision
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 2 Summary")
    print(f"{'='*60}")

    # Find best config at ~4-bit average budget
    best_config = None
    best_ppl = float('inf')
    for exp_name in ["exp_a_asymmetric", "exp_c_mixed_precision"]:
        for key, val in all_results.get(exp_name, {}).items():
            if isinstance(val, dict) and "ppl" in val:
                avg = val.get("avg_bits", val.get("avg_adt_bits", 0))
                if avg <= 5.5 and val["ppl"] < best_ppl:
                    best_ppl = val["ppl"]
                    best_config = f"{exp_name}/{key}"

    print(f"  Baseline PPL: {baseline_ppl}")
    print(f"  Phase 1 Uniform W4 Polar: 61.93")
    if best_config:
        print(f"  Best ≤5.5b config: {best_config} → PPL={best_ppl:.4f}")
    else:
        print(f"  No config found under 5.5 avg bits")

    # Determine go/no-go for Phase 3
    if best_ppl < 5.0:
        print(f"\n  → GO for Phase 3: Mixed-precision achieves near-lossless at ~4-5 bits")
    elif best_ppl < 20.0:
        print(f"\n  → CONDITIONAL GO: Significant improvement but still degraded")
    else:
        print(f"\n  → INVESTIGATE: Need better strategies (learned grids, GPTQ-style)")

    # Save
    out = WORKDIR / "ppq" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phase2_mixed_precision.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out / 'phase2_mixed_precision.json'}")


if __name__ == "__main__":
    main()
