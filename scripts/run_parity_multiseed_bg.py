"""Multi-seed + multi-config parity sweep, all in-process so we don't pay
TileLang autotune cost per subprocess. Total run: 18 configurations.

Each config: same as parity_task.py but parameterized in code.
Saves results to results/tables/parity_multiseed.json.
"""

from __future__ import annotations

import json
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


def run_one(arch, bitize, seed, n_steps=3000, d_model=128, seqlen=128, batch=32, lr=1e-3):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    tag = f"{arch}{'_bit' if bitize else ''}_seed{seed}"
    print(f"\n=== {tag} (d={d_model}, seqlen={seqlen}, steps={n_steps}) ===")
    model = ParityModel(arch, d_model=d_model, bitize=bitize, device="cuda", dtype=torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

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
        opt.step()

        if step % 200 == 0 or step == n_steps - 1:
            model.eval()
            acc_tot, acc_last = eval_acc(model, batch=64, seqlen=seqlen, device="cuda")
            model.train()
            peak_acc = max(peak_acc, acc_tot)
            elapsed = time.time() - t0
            print(f"  step={step:>5} loss={loss.item():.4f} acc={acc_tot:.4f} acc_last={acc_last:.4f} peak={peak_acc:.4f} ({elapsed:.1f}s)")
            history.append({"step": step, "loss": loss.item(), "acc_all": acc_tot, "acc_last": acc_last})

    # 2x seqlen
    model.eval()
    try:
        acc_2x_tot, acc_2x_last = eval_acc(model, batch=64, seqlen=seqlen * 2, device="cuda")
    except Exception:
        acc_2x_tot, acc_2x_last = None, None

    summary = {
        "tag": tag,
        "arch": arch,
        "bitize": bitize,
        "seed": seed,
        "d_model": d_model,
        "seqlen": seqlen,
        "n_steps": n_steps,
        "history": history,
        "peak_acc_all": peak_acc,
        "final_acc_all": history[-1]["acc_all"] if history else None,
        "acc_2x_seqlen_all": acc_2x_tot,
        "acc_2x_seqlen_last": acc_2x_last,
        "time_sec": time.time() - t0,
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    out_path = _root / "results/tables/parity_multiseed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("mamba2", False),
        ("mamba2", True),
        ("mamba3_siso", False),
        ("mamba3_siso", True),
        ("mamba3_mimo", False),
        ("mamba3_mimo", True),
    ]
    seeds = [0, 1, 2]

    results = []
    for arch, bitize in configs:
        for seed in seeds:
            summary = run_one(arch, bitize, seed)
            results.append(summary)
            # Save incrementally so a crash doesn't lose work
            out_path.write_text(json.dumps(results, indent=2))

    print(f"\nAll {len(results)} runs complete -> {out_path}")


if __name__ == "__main__":
    main()
