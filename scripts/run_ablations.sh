#!/bin/bash
# Dispatch ablation matrix for BitMamba-3.
#
# Requires a trained 30M proxy checkpoint as warm start (or trains each from scratch).

set -e

cd "$(dirname "$0")/.."

WSL_PREFIX='wsl -d Ubuntu-24.04 --exec bash -lc "env -i HOME=\$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c \"source /home/qerti/miniconda3/etc/profile.d/conda.sh && conda activate taq-vidssm && cd /mnt/g/Github\\\\ Desktop/bitMamba-3 && '

# Placeholder — to be fleshed out once 30M baseline is trained.
echo "Ablations (all run 30M preset from scratch on 500M tokens):"
echo "  1. mimo vs siso"
echo "  2. rope_fraction=0.5 vs 1.0"
echo "  3. mimo_rank=2 vs 4"
echo "  4. ternarize_embedding on/off"
echo ""
echo "Implementation deferred until baseline 30M training completes."
