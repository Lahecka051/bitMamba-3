#!/bin/bash
# Helper: invoke command in WSL taq-vidssm env with clean Windows PATH contamination removed.
# Usage: ./scripts/wsl-run.sh "python -c 'import torch; print(torch.__version__)'"

CMD="$*"
wsl -d Ubuntu-24.04 --exec bash -lc "env -i HOME=\$HOME PATH=/usr/local/bin:/usr/bin:/bin bash -c 'source /home/qerti/miniconda3/etc/profile.d/conda.sh && conda activate taq-vidssm && cd /mnt/g/Github\\ Desktop/bitMamba-3 && $CMD'"
