#!/bin/bash
# Multi-seed parity task confirmation: run 6 configs x 3 seeds at d_model=128, seqlen=128, n_steps=3000.
# Total: 18 runs, each ~6-8 min on RTX 5090 = ~2-2.5 hours total.
#
# Run with: bash scripts/run_parity_multiseed.sh

set -e
cd "$(dirname "$0")/.."

CONFIGS=(
  "mamba2"
  "mamba2 --bitize"
  "mamba3_siso"
  "mamba3_siso --bitize"
  "mamba3_mimo"
  "mamba3_mimo --bitize"
)

SEEDS=(0 1 2)

for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "=== Running: $CONFIG (seed=$SEED) ==="
    wsl -d Ubuntu-24.04 --exec bash -lc "
      env -i HOME=\$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c '
        source /home/qerti/miniconda3/etc/profile.d/conda.sh &&
        conda activate taq-vidssm &&
        cd /mnt/g/Github\\ Desktop/bitMamba-3 &&
        python src/evaluation/parity_task.py \
            --arch ${CONFIG} \
            --seed ${SEED} \
            --n_steps 3000 \
            --d_model 128 \
            --seqlen 128 \
            --batch 32 \
            --eval_interval 200 \
            2>&1 | tee -a results/logs/parity_multiseed.log
      '
    "
  done
done

echo "All multiseed runs complete."
