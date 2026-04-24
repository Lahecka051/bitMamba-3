"""End-to-end smoke test: load real tokenized data + 100 training steps on 30M MIMO model."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "third_party" / "state-spaces-mamba"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bitmamba3 import ensure_mamba3_registered
ensure_mamba3_registered()

from mamba_ssm.models.config_mamba import MambaConfig  # type: ignore
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # type: ignore
from training.train import PRESETS, bitify_model  # type: ignore
from training.dataset import TokenShardDataset  # type: ignore


def main():
    data_dir = _root / "data" / "fineweb_10M"
    assert data_dir.exists(), f"Data dir missing: {data_dir}"

    print(f"Loading dataset from {data_dir}")
    ds = TokenShardDataset(str(data_dir), seqlen=512, samples_per_epoch=200)
    loader = DataLoader(ds, batch_size=4, num_workers=0, pin_memory=True)
    print(f"  total_tokens={ds.total_tokens:,}, shards={len(ds.shard_memmaps)}")

    # Verify a batch
    x, y = next(iter(loader))
    print(f"  batch shapes: x={tuple(x.shape)}, y={tuple(y.shape)}")
    print(f"  token range: min={x.min().item()}, max={x.max().item()}")

    print("\nBuilding 30M MIMO model")
    cfg = MambaConfig(**PRESETS["30M"])
    model = MambaLMHeadModel(cfg, device="cuda", dtype=torch.bfloat16)
    bitify_model(model)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    print("\n100-step training run")
    t_start = time.time()
    step = 0
    for x, y in loader:
        if step >= 100:
            break
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x).logits
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            elapsed = time.time() - t_start
            tok_s = (step + 1) * x.shape[0] * x.shape[1] / max(elapsed, 0.001)
            print(f"  step={step:>3}  loss={loss.item():.4f}  tok/s={tok_s:.0f}  elapsed={elapsed:.1f}s")
        step += 1

    print("\n100-step training OK")


if __name__ == "__main__":
    main()
