#!/bin/bash
#SBATCH -J aimo-ownunc
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH -t 01:30:00
#SBATCH -o logs/ownunc_%x_%j.log
# Usage: sbatch -p <partition> --gres=<gpu> --mem=<mem> -J ownunc-<tag> aimo_ownunc.sbatch <model_id> [effort ...]
set -uo pipefail
MODEL=$1; shift
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache TRITON_CACHE_DIR=$BASE/triton_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('cuda smoke ok', torch.cuda.get_device_name(0))" || exit 1
cd $BASE/code
if [ $# -eq 0 ]; then
  $PY own_uncertainty.py --parquet $BASE/data_trainv2/train-main-v2.parquet --out-dir $BASE/results/ownunc --model $MODEL --batch-size ${BS:-36} 2>&1 | grep -v "Loading weights"
else
  for E in "$@"; do
    $PY own_uncertainty.py --parquet $BASE/data_trainv2/train-main-v2.parquet --out-dir $BASE/results/ownunc --model $MODEL --batch-size ${BS:-36} --effort $E 2>&1 | grep -v "Loading weights"
  done
fi
echo "=== DONE $(date -Is)"
