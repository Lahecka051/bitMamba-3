#!/bin/bash
# Launch 130M BitMamba-3 MIMO training on 1B fineweb-edu tokens.
# Single epoch. ~3 hours estimated at 195K tok/s for 30M, scales down for 130M.

set -e
cd "$(dirname "$0")/.."

wsl -d Ubuntu-24.04 --exec bash -lc '
env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c "
    source /home/qerti/miniconda3/etc/profile.d/conda.sh &&
    conda activate taq-vidssm &&
    cd /mnt/g/Github\\ Desktop/bitMamba-3 &&
    python -u src/training/train.py \
        --preset 130M \
        --data_dir data/fineweb_1B \
        --out_dir checkpoints/bitmamba3_130M_mimo \
        --seqlen 2048 \
        --batch_size 4 \
        --grad_accum 4 \
        --max_steps 30000 \
        --warmup_steps 2000 \
        --base_lr 3e-4 \
        --log_interval 50 \
        --save_interval 2500 \
        --dtype bfloat16 \
        --grad_ckpt \
        --wandb_run_name bitmamba3_130M_mimo_fineweb1B \
        2>&1 | tee results/logs/train_130M_mimo.log
"
'
