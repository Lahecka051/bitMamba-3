"""Quick parity check: does INT4 also enable state tracking, or only ternary?

If INT4 stays at chance like FP, then the inductive-bias finding is specific to
sub-2-bit quantization (ternary), not general low-bit. If INT4 also reaches
0.98, the finding generalizes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "scripts"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered  # noqa: E402
ensure_mamba3_registered()

from evaluation.parity_task import ParityModel, make_parity_batch, eval_acc  # noqa: E402

import math


def cosine_lr(step, max_steps, base_lr, warmup_steps, min_lr=1e-5):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_one(arch, bitize_mode, seed, n_steps=5000, d_model=512, depth=4):
    import random, numpy as np
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tag = f"{arch}_{bitize_mode}_seed{seed}"
    print(f"\n=== {tag} (d={d_model}, depth={depth}) ===")

    model = ParityModel(arch, d_model=d_model, bitize=bitize_mode, depth=depth,
                        device="cuda", dtype=torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = []
    peak = 0.0
    for step in range(n_steps):
        bits, labels = make_parity_batch(32, 128, "cuda")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(bits)
            loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, n_steps, 1e-3, 200, 1e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()

        if step % 250 == 0 or step == n_steps - 1:
            model.eval()
            acc, acc_last = eval_acc(model, batch=64, seqlen=128, device="cuda")
            model.train()
            peak = max(peak, acc)
            print(f"  step={step:>5} loss={loss.item():.4f} acc={acc:.4f} peak={peak:.4f}")
            history.append({"step": step, "loss": loss.item(), "acc_all": acc, "acc_last": acc_last})

    model.eval()
    acc_2x, _ = eval_acc(model, batch=64, seqlen=256, device="cuda")
    out = {"tag": tag, "arch": arch, "bitize_mode": bitize_mode, "seed": seed,
           "peak_acc_all": peak, "final_acc_all": history[-1]["acc_all"],
           "acc_2x_seqlen_all": acc_2x, "history": history,
           "d_model": d_model, "depth": depth, "n_steps": n_steps}
    del model, opt
    torch.cuda.empty_cache()
    return out


def main():
    out_path = _root / "results/tables/parity_int4_d512.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Run INT4 + ternary + FP for SISO and MIMO at 3 seeds
    for arch in ["mamba3_mimo", "mamba3_siso"]:
        for mode in ["int4", "ternary", False]:  # INT4, ternary, FP
            for seed in [0, 1, 2]:
                r = run_one(arch, mode, seed)
                results.append(r)
                out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} runs to {out_path}")


if __name__ == "__main__":
    main()
