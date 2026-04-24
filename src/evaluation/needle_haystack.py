"""Needle-in-Haystack: retrieval evaluation across long contexts.

Classic long-context retrieval benchmark. We construct filler text with a
randomly-placed "needle" sentence containing a unique magic number, then ask
the model to recall it.

For autoregressive evaluation we measure the model's probability of generating
the correct magic number given a "What is the magic number?" query appended
after the context.

Usage:
    python src/evaluation/needle_haystack.py \
        --ckpt /path/ckpt.pt --preset 30M \
        --contexts 512 2048 8192 32768 \
        --depths 0 25 50 75 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

from bitmamba3 import ensure_mamba3_registered  # noqa: E402
ensure_mamba3_registered()

from transformers import AutoTokenizer  # type: ignore  # noqa: E402
from mamba_ssm.models.config_mamba import MambaConfig  # type: ignore  # noqa: E402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore  # noqa: E402
from training.train import PRESETS, bitify_model  # type: ignore  # noqa: E402


FILLER_SENTENCES = [
    "The sky was clear and blue without a single cloud in sight.",
    "Researchers have found that green spaces improve cognitive function.",
    "Economic reports suggest a gradual recovery in coastal regions.",
    "Ancient manuscripts preserved for centuries reveal forgotten histories.",
    "Volunteers organized a community effort to clean up the riverbank.",
    "Innovations in battery technology continue to advance each year.",
    "Archeologists discovered pottery fragments buried near the old temple.",
    "Nutritionists recommend varying your diet to include many vegetables.",
    "The orchestra performed a rousing symphony that moved the audience.",
    "New transit lines will connect the suburbs to the downtown core.",
]


def build_haystack(num_tokens: int, needle_sentence: str, depth_pct: int, tokenizer) -> tuple[str, int]:
    """Produce a haystack passage of ~num_tokens with needle at depth_pct% through."""
    target_tokens_before = int(num_tokens * depth_pct / 100.0)
    target_tokens_after = num_tokens - target_tokens_before

    def make_filler(n_toks):
        chunks = []
        total = 0
        while total < n_toks:
            s = random.choice(FILLER_SENTENCES) + " "
            chunks.append(s)
            total += len(tokenizer.encode(s, add_special_tokens=False))
        return " ".join(chunks)

    before = make_filler(target_tokens_before)
    after = make_filler(target_tokens_after)
    passage = f"{before} {needle_sentence} {after}"
    # Tokenize final passage length
    actual_len = len(tokenizer.encode(passage, add_special_tokens=False))
    return passage, actual_len


@torch.no_grad()
def score_needle(model, tokenizer, passage: str, magic_number: str, device="cuda") -> float:
    """Return average log-probability of the magic number tokens given the passage + question."""
    question = " What is the magic number mentioned in the passage? The magic number is"
    prompt = passage + question
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    magic_ids = tokenizer.encode(" " + magic_number, add_special_tokens=False)

    full_ids = torch.tensor([prompt_ids + magic_ids], device=device, dtype=torch.long)
    logits = model(full_ids).logits  # (1, L, V)

    # Score the magic number tokens by their log-prob given prior context
    log_probs = torch.log_softmax(logits[0], dim=-1)
    prompt_len = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(magic_ids):
        total += log_probs[prompt_len + i - 1, tok].item()
    return total / max(len(magic_ids), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="30M")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 25, 50, 75, 100])
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--out_json", default="results/tables/needle_haystack.json")
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    cfg = MambaConfig(**PRESETS[args.preset])
    model = MambaLMHeadModel(cfg, device=device, dtype=torch.bfloat16)
    bitify_model(model)
    state = torch.load(args.ckpt, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    results = {}
    for L in args.contexts:
        results[L] = {}
        for d in args.depths:
            scores = []
            for t in range(args.n_trials):
                magic = f"{random.randint(10000, 99999)}"
                needle = f"The magic number is {magic}."
                passage, actual_L = build_haystack(L, needle, d, tokenizer)
                score = score_needle(model, tokenizer, passage, magic, device=device)
                scores.append(score)
            avg = sum(scores) / len(scores)
            results[L][d] = {"mean_log_prob": avg, "trials": scores}
            print(f"L={L:>6}  depth={d:>3}%  avg_log_prob={avg:+.3f}")

    out_path = _root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"ckpt": args.ckpt, "preset": args.preset, "results": results}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
