#!/bin/bash
#SBATCH -J aimo-selflabel
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/selflabel_%j.log
# Pilot on 40 of the 164 problems that already carry official labels, about 1.9 GPU hours at the
# measured 22.5 s per sequence. The run is resumable, so it extends rather than restarts if the
# self-derived label turns out to correlate with the official one.
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$PY - <<"PYEOF" || { echo "ABORT: GPU smoke test failed"; exit 1; }
import torch
assert torch.cuda.is_available(), "no CUDA"
x = torch.randn(2048, 2048, device="cuda"); y = x @ x.T; torch.cuda.synchronize()
assert torch.isfinite(y).all().item(), "non-finite matmul"
print("cuda smoke ok:", torch.__version__, torch.cuda.get_device_name(0))
PYEOF
cd $BASE/code && $PY selflabel_generate.py \
  --data-dir $BASE/data --answers $BASE/data_derived/answers_math.json \
  --out-dir $BASE/results/selflabel_pilot --limit-problems 40 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
