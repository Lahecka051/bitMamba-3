"""Larger-scale parity ablation: d_model=256, depth=2, 5 seeds, 5000 steps with cosine LR.

Hypothesis: The high seed variance at d=128/depth=1/3K-steps may stabilize
at this larger configuration, giving a clearer ternary-vs-FP comparison.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered  # noqa: E402
ensure_mamba3_registered()

from evaluation.parity_task import ParityModel, make_parity_batch, eval_acc  # type: ignore  # noqa: E402


def cosine_lr(step, max_steps, base_lr, warmup_steps, min_lr=1e-5):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_one(arch, bitize, seed, n_steps=5000, d_model=256, seqlen=128, batch=32, base_lr=1e-3, depth=2):
    torch.manual_seed(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

    tag = f"{arch}{'_bit' if bitize else ''}_seed{seed}_d{d_model}_depth{depth}"
    print(f"\n=== {tag} (steps={n_steps}, base_lr={base_lr}) ===")
    model = ParityModel(arch, d_model=d_model, bitize=bitize, depth=depth, device="cuda", dtype=torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

    history = []
    peak_acc = 0.0
    t0 = time.time()
    for step in range(n_steps):
        bits, labels = make_parity_batch(batch, seqlen, "cuda")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(bits)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, n_steps, base_lr, warmup_steps=200, min_lr=1e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()

        if step % 250 == 0 or step == n_steps - 1:
            model.eval()
            acc_tot, acc_last = eval_acc(model, batch=64, seqlen=seqlen, device="cuda")
            model.train()
            peak_acc = max(peak_acc, acc_tot)
            elapsed = time.time() - t0
            print(f"  step={step:>5} lr={lr:.1e} loss={loss.item():.4f} acc={acc_tot:.4f} peak={peak_acc:.4f} ({elapsed:.1f}s)")
            history.append({"step": step, "lr": lr, "loss": loss.item(), "acc_all": acc_tot, "acc_last": acc_last})

    model.eval()
    try:
        acc_2x_tot, acc_2x_last = eval_acc(model, batch=64, seqlen=seqlen * 2, device="cuda")
    except Exception:
        acc_2x_tot, acc_2x_last = None, None

    summary = {
        "tag": tag, "arch": arch, "bitize": bitize, "seed": seed,
        "d_model": d_model, "depth": depth, "seqlen": seqlen, "n_steps": n_steps,
        "history": history, "peak_acc_all": peak_acc,
        "final_acc_all": history[-1]["acc_all"] if history else None,
        "acc_2x_seqlen_all": acc_2x_tot, "acc_2x_seqlen_last": acc_2x_last,
        "time_sec": time.time() - t0,
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    out_path = _root / "results/tables/parity_larger.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Focus on the most informative configurations: skip Mamba-2 (already
    # confirmed at chance) and SISO/MIMO comparisons across ternary on/off.
    configs = [
        ("mamba3_siso", False),
        ("mamba3_siso", True),
        ("mamba3_mimo", False),
        ("mamba3_mimo", True),
    ]
    seeds = [0, 1, 2, 3, 4]  # 5 seeds

    results = []
    for arch, bitize in configs:
        for seed in seeds:
            summary = run_one(arch, bitize, seed)
            results.append(summary)
            out_path.write_text(json.dumps(results, indent=2))

    print(f"\nAll {len(results)} larger-scale runs complete -> {out_path}")


if __name__ == "__main__":
    main()
