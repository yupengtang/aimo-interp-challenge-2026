#!/bin/bash
#SBATCH -J aimo-tv2gen
#SBATCH -p ice-gpu
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=128G
#SBATCH -t 03:30:00
#SBATCH -o logs/tv2gen_%x_%j.log
# Usage: sbatch --gres=gpu:h200:1 -J tv2-<tag> aimo_trainv2_gen.sbatch <model_id> [batch_size]
set -uo pipefail
MODEL=$1 BS=${2:-32}
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
cd $BASE/code && $PY trainv2_generate.py --parquet $BASE/data_trainv2/train-main-v2.parquet \
  --out-dir $BASE/results/trainv2_gen --model $MODEL --batch-size $BS 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
