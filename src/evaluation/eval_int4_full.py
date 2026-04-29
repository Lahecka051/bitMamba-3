"""Full eval suite (lm-eval + needle + long-context) on INT4 PTQ Mamba-3.

Runs the same eval suite we ran on BitMamba-2 / BitMamba-3 / Mamba-3 FP, but
on the INT4-PTQ-applied variant. Lets us complete the 4-config matrix:
  FP / INT4 / ternary at 130M
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered
ensure_mamba3_registered()

from transformers import AutoTokenizer
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from training.train import PRESETS
from bitmamba3.int4_linear import int4_quantize_model


def load_int4(ckpt, preset, device="cuda"):
    cfg = MambaConfig(**PRESETS[preset])
    model = MambaLMHeadModel(cfg, device=device, dtype=torch.bfloat16)
    state = torch.load(ckpt, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    int4_quantize_model(model)
    model.eval()
    return model


@torch.no_grad()
def long_context_ppl(model, tokenizer, n_books=5, seqlens=(1024, 2048, 4096)):
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="test", streaming=False)
    texts = [ex["text"] for ex in ds.select(range(n_books))]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.cuda()
    results = {}
    for L in seqlens:
        nlls = []
        prev = 0
        for begin in range(0, enc.size(1), L // 2):
            end = min(begin + L, enc.size(1))
            trg = end - prev
            ids = enc[:, begin:end]
            labels = ids.clone()
            labels[:, :-trg] = -100
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids).logits
                loss = F.cross_entropy(
                    logits[..., :-1, :].reshape(-1, logits.size(-1)),
                    labels[..., 1:].reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            nlls.append(loss.item())
            prev = end
            if end == enc.size(1):
                break
        results[L] = {"ppl": math.exp(sum(nlls) / enc.size(1)), "tokens": int(enc.size(1))}
    return results


@torch.no_grad()
def needle(model, tokenizer, contexts=(512, 2048, 4096), depths=(0, 50, 100), n_trials=3):
    import random
    FILLER = [
        "The sky was clear and blue without a single cloud in sight.",
        "Researchers have found that green spaces improve cognitive function.",
        "Economic reports suggest a gradual recovery in coastal regions.",
        "Ancient manuscripts preserved for centuries reveal forgotten histories.",
        "Volunteers organized a community effort to clean up the riverbank.",
    ]
    results = {L: {} for L in contexts}
    for L in contexts:
        for d in depths:
            scores = []
            for _ in range(n_trials):
                magic = f"{random.randint(10000, 99999)}"
                needle_s = f"The magic number is {magic}."
                target_before = int(L * d / 100)
                target_after = L - target_before
                def make_filler(n):
                    s = ""; total = 0
                    while total < n:
                        s += random.choice(FILLER) + " "
                        total += len(tokenizer.encode(s, add_special_tokens=False))
                    return s
                passage = f"{make_filler(target_before)} {needle_s} {make_filler(target_after)}"
                question = " What is the magic number mentioned in the passage? The magic number is"
                prompt = passage + question
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                magic_ids = tokenizer.encode(" " + magic, add_special_tokens=False)
                full = torch.tensor([prompt_ids + magic_ids], device="cuda", dtype=torch.long)
                logits = model(full).logits[0]
                logp = torch.log_softmax(logits, dim=-1)
                pl = len(prompt_ids)
                total = sum(logp[pl + i - 1, t].item() for i, t in enumerate(magic_ids))
                scores.append(total / max(len(magic_ids), 1))
            results[L][d] = {"mean_log_prob": sum(scores) / len(scores), "trials": scores}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="130M")
    ap.add_argument("--mode", choices=["needle", "long_context", "both"], default="both")
    args = ap.parse_args()

    print(f"Loading INT4 PTQ from {args.ckpt}")
    model = load_int4(args.ckpt, args.preset)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if args.mode in ("long_context", "both"):
        print("\nRunning PG19 long-context PPL...")
        lc = long_context_ppl(model, tok)
        out_lc = _root / "results/tables/long_context_int4_130M.json"
        out_lc.parent.mkdir(parents=True, exist_ok=True)
        out_lc.write_text(json.dumps({"ckpt": args.ckpt, "preset": args.preset, "results": lc}, indent=2))
        for L, r in lc.items():
            print(f"  L={L}: PPL={r['ppl']:.2f}")
        print(f"Saved {out_lc}")

    if args.mode in ("needle", "both"):
        print("\nRunning needle-in-haystack...")
        nh = needle(model, tok)
        out_nh = _root / "results/tables/needle_int4_130M.json"
        out_nh.write_text(json.dumps({"ckpt": args.ckpt, "preset": args.preset, "results": nh}, indent=2))
        for L, depths in nh.items():
            for d, c in depths.items():
                print(f"  L={L} depth={d}%: {c['mean_log_prob']:+.2f}")
        print(f"Saved {out_nh}")


if __name__ == "__main__":
    main()
