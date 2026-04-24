#!/bin/bash
# Launch 30M BitMamba-3 MIMO proxy training on 1B fineweb-edu tokens.
# Runs under WSL taq-vidssm env with clean PATH.

set -e

cd "$(dirname "$0")/.."

wsl -d Ubuntu-24.04 --exec bash -lc '
env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c "
    source /home/qerti/miniconda3/etc/profile.d/conda.sh &&
    conda activate taq-vidssm &&
    cd /mnt/g/Github\\ Desktop/bitMamba-3 &&
    python src/training/train.py \
        --preset 30M \
        --data_dir data/fineweb_1B \
        --out_dir checkpoints/bitmamba3_30M_mimo \
        --seqlen 2048 \
        --batch_size 8 \
        --grad_accum 1 \
        --max_steps 10000 \
        --warmup_steps 500 \
        --base_lr 3e-4 \
        --log_interval 50 \
        --save_interval 1000 \
        --dtype bfloat16 \
        --wandb_run_name bitmamba3_30M_mimo_fineweb1B \
        2>&1 | tee results/logs/train_30M_mimo.log
"
'
