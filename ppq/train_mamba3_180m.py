"""
Train Mamba-3 180M from scratch on FineWeb-Edu.
Following the paper: 100B tokens, Llama-3.1 tokenizer, 2K context.

Model config (180M):
  d_model=768, n_layer=24, d_state=128, headdim=64
  5:1 SSM:Attention ratio → 20 Mamba3 + 4 Attention = 24 layers
"""
import os, sys, json, math, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path

DEVICE = 'cuda'
WORKDIR = Path(__file__).resolve().parent.parent


# ============================================================
# Model Configuration
# ============================================================
MODEL_CONFIGS = {
    "180M": {
        "d_model": 768,
        "n_layer": 24,
        "d_state": 128,
        "headdim": 64,
        "vocab_size": 50257,  # GPT-2 tokenizer (Llama-3.1 gated)
        "seq_len": 2048,
    },
    "440M": {
        "d_model": 1024,
        "n_layer": 48,
        "d_state": 128,
        "headdim": 64,
        "vocab_size": 50257,
        "seq_len": 2048,
    },
}


class Mamba3LM(nn.Module):
    """Mamba-3 Language Model with 5:1 SSM:Attention hybrid."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config["d_model"]
        n = config["n_layer"]
        V = config["vocab_size"]

        self.embedding = nn.Embedding(V, d)
        self.norm_f = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        # Weight tying (standard for LMs)
        self.lm_head.weight = self.embedding.weight

        # 5:1 ratio: every 6th layer is attention
        self.layers = nn.ModuleList()
        self.layer_types = []

        try:
            from mamba_ssm import Mamba3
            has_mamba3 = True
        except ImportError:
            has_mamba3 = False
            print("WARNING: Mamba3 not available, using placeholder")

        for i in range(n):
            if (i + 1) % 6 == 0:  # Every 6th layer = attention
                self.layers.append(self._make_attention(d))
                self.layer_types.append("attention")
            else:
                if has_mamba3:
                    self.layers.append(self._make_mamba3(d, config["d_state"], config["headdim"]))
                else:
                    self.layers.append(self._make_mamba3_placeholder(d))
                self.layer_types.append("ssm")

        n_ssm = sum(1 for t in self.layer_types if t == "ssm")
        n_attn = sum(1 for t in self.layer_types if t == "attention")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Mamba3LM: {total_params/1e6:.1f}M params, "
              f"{n_ssm}×SSM + {n_attn}×Attention ({n_ssm}:{n_attn})")

    def _make_mamba3(self, d_model, d_state, headdim):
        from mamba_ssm import Mamba3
        return nn.ModuleDict({
            "norm": nn.RMSNorm(d_model),
            "block": Mamba3(
                d_model=d_model,
                d_state=d_state,
                headdim=headdim,
                is_mimo=False,  # SISO for simplicity first
            ),
        })

    def _make_mamba3_placeholder(self, d_model):
        """Placeholder if Mamba3 not built yet."""
        return nn.ModuleDict({
            "norm": nn.RMSNorm(d_model),
            "block": nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            ),
        })

    def _make_attention(self, d_model, nheads=None):
        if nheads is None:
            nheads = d_model // 64
        return nn.ModuleDict({
            "norm": nn.RMSNorm(d_model),
            "block": nn.MultiheadAttention(
                d_model, nheads, batch_first=True, dropout=0.0
            ),
            "ffn_norm": nn.RMSNorm(d_model),
            "ffn": nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model),
            ),
        })

    def forward(self, input_ids, targets=None):
        h = self.embedding(input_ids)  # (B, L, D)

        for i, (layer, ltype) in enumerate(zip(self.layers, self.layer_types)):
            if ltype == "ssm":
                residual = h
                h_norm = layer["norm"](h)
                h = residual + layer["block"](h_norm)
            else:  # attention
                residual = h
                h_norm = layer["norm"](h)
                h_attn, _ = layer["block"](h_norm, h_norm, h_norm, need_weights=False)
                h = residual + h_attn
                # FFN
                residual = h
                h = residual + layer["ffn"](layer["ffn_norm"](h))

        h = self.norm_f(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ============================================================
# Data Loading (FineWeb-Edu streaming)
# ============================================================
class FineWebEduDataset:
    """Streaming dataset from HuggingFace FineWeb-Edu."""
    def __init__(self, tokenizer, seq_len=2048, buffer_size=10000):
        from datasets import load_dataset
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",  # 10B token sample for faster start
            split="train",
            streaming=True,
        )
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []
        self.iter = iter(self.dataset)

    def get_batch(self, batch_size):
        """Get a batch of tokenized sequences."""
        while len(self.buffer) < batch_size * (self.seq_len + 1):
            try:
                sample = next(self.iter)
                tokens = self.tokenizer.encode(sample["text"])
                self.buffer.extend(tokens)
            except StopIteration:
                # Restart iterator
                self.iter = iter(self.dataset)
                sample = next(self.iter)
                tokens = self.tokenizer.encode(sample["text"])
                self.buffer.extend(tokens)

        # Create batch
        input_ids = []
        targets = []
        for _ in range(batch_size):
            chunk = self.buffer[:self.seq_len + 1]
            self.buffer = self.buffer[self.seq_len + 1:]
            input_ids.append(torch.tensor(chunk[:self.seq_len], dtype=torch.long))
            targets.append(torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long))

        return torch.stack(input_ids), torch.stack(targets)


# ============================================================
# Training Loop
# ============================================================
def validate_model(model, tokenizer, tag=""):
    """Run sanity checks: unseen PPL, random PPL, copy test."""
    import torch.nn.functional as F
    from torch.amp import autocast

    model.eval()
    print(f"\n  {'='*50}")
    print(f"  Validation Checkpoint {tag}")
    print(f"  {'='*50}")

    # 1. Unseen text
    texts = [
        "The solar system consists of the Sun and eight planets orbiting around it.",
        "Machine learning algorithms identify patterns in data without explicit programming.",
        "The French Revolution began in 1789 and changed the course of modern history.",
        "Quantum computing uses superposition and entanglement to process information.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    ]
    total_loss = 0
    total_tokens = 0
    for text in texts:
        ids = tokenizer.encode(text)
        x = torch.tensor([ids[:-1]]).to(DEVICE)
        y = torch.tensor([ids[1:]]).to(DEVICE)
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * len(ids)
        total_tokens += len(ids)
    unseen_ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))

    # 2. Random tokens
    import random
    rand_ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(128)]
    x = torch.tensor([rand_ids[:-1]]).to(DEVICE)
    y = torch.tensor([rand_ids[1:]]).to(DEVICE)
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    random_ppl = math.exp(min(loss.item(), 20))

    # 3. Copy test
    x = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]]).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
    copy_matches = sum(1 for i in range(7) if preds[0, i].item() == x[0, i + 1].item())

    print(f"  Unseen PPL:  {unseen_ppl:.2f}  (expect 20-80)")
    print(f"  Random PPL:  {random_ppl:.2f}  (expect >1000)")
    print(f"  Copy ratio:  {copy_matches}/7  (expect 0-1)")

    status = "PASS" if unseen_ppl > 10 and random_ppl > 100 and copy_matches <= 2 else "FAIL"
    print(f"  Status: {status}")
    print(f"  {'='*50}\n")

    model.train()
    return {"unseen_ppl": unseen_ppl, "random_ppl": random_ppl, "copy_ratio": copy_matches, "status": status}


def train(model_size="180M", total_tokens=100_000_000_000, batch_size=8, grad_accum=4,
          resume_from=None, stop_at_tokens=None):
    config = MODEL_CONFIGS[model_size]
    seq_len = config["seq_len"]
    effective_batch = batch_size * grad_accum
    tokens_per_step = effective_batch * seq_len

    print(f"{'='*60}")
    print(f"Training Mamba-3 {model_size}")
    print(f"{'='*60}")
    print(f"  Effective batch: {batch_size} × {grad_accum} = {effective_batch}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {total_tokens/1e9:.0f}B (LR schedule basis)")
    if stop_at_tokens:
        print(f"  Stop at: {stop_at_tokens/1e9:.0f}B (checkpoint + validate)")
    print(f"  Total steps: {total_tokens // tokens_per_step:,}")
    print()

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config["vocab_size"] = len(tokenizer)

    # Model
    model = Mamba3LM(config).to(DEVICE)

    # Dataset
    dataset = FineWebEduDataset(tokenizer, seq_len)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # LR Schedule: warmup + cosine decay (always based on total_tokens)
    total_steps = total_tokens // tokens_per_step
    warmup_steps = min(2000, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler(device='cuda')

    # Resume from checkpoint
    start_step = 0
    total_tokens_seen = 0
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        total_tokens_seen = ckpt["tokens_seen"]
        # Advance scheduler to correct position
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resumed from step {start_step}, {total_tokens_seen/1e9:.2f}B tokens")

    # Training
    ckpt_dir = WORKDIR / "ppq" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_interval = 100
    save_interval = 10000
    running_loss = 0
    t0 = time.time()
    actual_stop = stop_at_tokens if stop_at_tokens else total_tokens

    model.train()
    optimizer.zero_grad()

    step = start_step
    while total_tokens_seen < actual_stop:
        for micro_step in range(grad_accum):
            input_ids, targets = dataset.get_batch(batch_size)
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(input_ids, targets)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item()

        # Gradient step
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        step += 1
        total_tokens_seen += tokens_per_step

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - t0
            tokens_since_start = total_tokens_seen - (start_step * tokens_per_step if resume_from else 0)
            tokens_per_sec = tokens_since_start / max(elapsed, 1)
            loss_avg = running_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            ppl = math.exp(min(loss_avg * grad_accum, 20))
            progress = total_tokens_seen / total_tokens * 100

            print(f"  Step {step:>6d} | Loss {loss_avg*grad_accum:.4f} | PPL {ppl:.1f} | "
                  f"LR {lr:.2e} | {tokens_per_sec/1e3:.1f}K tok/s | "
                  f"{total_tokens_seen/1e9:.2f}B/{total_tokens/1e9:.0f}B ({progress:.1f}%)",
                  flush=True)
            running_loss = 0

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = ckpt_dir / f"mamba3_{model_size}_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "tokens_seen": total_tokens_seen,
                "config": config,
            }, str(ckpt_path))
            print(f"  Saved: {ckpt_path.name}")

    # Save at stop point
    stop_path = ckpt_dir / f"mamba3_{model_size}_{total_tokens_seen/1e9:.0f}B.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "tokens_seen": total_tokens_seen,
        "config": config,
    }, str(stop_path))

    elapsed = time.time() - t0
    print(f"\nTraining stopped at {total_tokens_seen/1e9:.1f}B tokens in {elapsed/3600:.1f}h")
    print(f"Saved: {stop_path}")

    # Validation
    val_results = validate_model(model, tokenizer, tag=f"{total_tokens_seen/1e9:.0f}B")

    return model, val_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="180M", choices=["180M", "440M"])
    parser.add_argument("--tokens", type=float, default=100e9, help="Total tokens for LR schedule (default: 100B)")
    parser.add_argument("--stop", type=float, default=None, help="Stop at N tokens for checkpoint (e.g. 10e9)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    args = parser.parse_args()

    train(model_size=args.size, total_tokens=int(args.tokens),
          batch_size=args.batch, grad_accum=args.accum,
          resume_from=args.resume, stop_at_tokens=int(args.stop) if args.stop else None)
