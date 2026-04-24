"""Memmap-backed dataset for tokenized shards produced by prepare_data.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenShardDataset(Dataset):
    """Random windows of `seqlen+1` tokens drawn from sharded memmap files.

    The extra +1 yields (input_ids, labels) as (tokens[:-1], tokens[1:]).
    Effective epoch size is bounded by `samples_per_epoch` to decouple from
    the total number of tokens.
    """

    def __init__(self, data_dir: str | Path, seqlen: int = 2048, samples_per_epoch: int = 10_000):
        self.data_dir = Path(data_dir)
        self.seqlen = seqlen
        self.samples_per_epoch = samples_per_epoch

        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)
        dtype = np.uint32 if self.meta["dtype"] == "<class 'numpy.uint32'>" or "uint32" in self.meta["dtype"] else np.uint16

        shards = sorted(self.data_dir.glob("shard_*.bin"))
        assert shards, f"No shards found in {self.data_dir}"
        self.shard_memmaps = [np.memmap(s, dtype=dtype, mode="r") for s in shards]
        self.shard_sizes = np.array([len(m) for m in self.shard_memmaps])
        self.shard_offsets = np.concatenate([[0], np.cumsum(self.shard_sizes)[:-1]])
        self.total_tokens = int(self.shard_sizes.sum())
        assert self.total_tokens > seqlen + 1, f"Not enough tokens: {self.total_tokens} <= {seqlen+1}"

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        max_start = self.total_tokens - self.seqlen - 1
        g_start = np.random.randint(0, max_start + 1)
        g_end = g_start + self.seqlen + 1

        # Locate shard(s) containing this slice
        shard_idx = int(np.searchsorted(self.shard_offsets, g_start, side="right") - 1)
        local_start = g_start - int(self.shard_offsets[shard_idx])
        local_end = local_start + self.seqlen + 1
        shard = self.shard_memmaps[shard_idx]

        if local_end <= len(shard):
            toks = np.asarray(shard[local_start:local_end], dtype=np.int64)
        else:
            # Crosses shard boundary
            part1 = np.asarray(shard[local_start:], dtype=np.int64)
            need = (self.seqlen + 1) - len(part1)
            next_shard = self.shard_memmaps[shard_idx + 1]
            part2 = np.asarray(next_shard[:need], dtype=np.int64)
            toks = np.concatenate([part1, part2])

        toks_t = torch.from_numpy(toks)
        return toks_t[:-1], toks_t[1:]
