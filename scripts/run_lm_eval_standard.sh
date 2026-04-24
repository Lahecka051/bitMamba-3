#!/bin/bash
# Run lm-evaluation-harness standard zero-shot suite against a BitMamba-3 checkpoint.
#
# Usage:
#   ./scripts/run_lm_eval_standard.sh <ckpt.pt> [preset]
#
# Output: results/tables/lm_eval_<ckpt_name>.json

set -e
cd "$(dirname "$0")/.."

CKPT="${1:?usage: $0 <ckpt.pt> [preset=30M]}"
PRESET="${2:-30M}"
NAME=$(basename "$CKPT" .pt)
OUT="results/tables/lm_eval_${NAME}.json"

TASKS="lambada_openai,hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq,openbookqa"

wsl -d Ubuntu-24.04 --exec bash -lc "
    env -i HOME=\$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c '
        source /home/qerti/miniconda3/etc/profile.d/conda.sh &&
        conda activate taq-vidssm &&
        cd /mnt/g/Github\\ Desktop/bitMamba-3 &&
        export PYTHONPATH=/mnt/g/Github\\ Desktop/bitMamba-3/src:/mnt/g/Github\\ Desktop/bitMamba-3/third_party/state-spaces-mamba &&
        python -m lm_eval \
            --model bitmamba3 \
            --model_args preset=${PRESET},ckpt=${CKPT},tokenizer=EleutherAI/gpt-neox-20b,dtype=bfloat16,batch_size=8 \
            --tasks ${TASKS} \
            --output_path ${OUT} \
            --log_samples
    '
"
