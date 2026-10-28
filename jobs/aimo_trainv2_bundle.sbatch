#!/bin/bash
#SBATCH -J aimo-tv2bundle
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=160G
#SBATCH -t 01:30:00
#SBATCH -o logs/tv2bundle_%j.log
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('cuda smoke ok', torch.cuda.get_device_name(0))" || exit 1
mkdir -p $BASE/results/trainv2_bundle
cd $BASE/code && $PY trainv2_bundle_eval.py --parquet $BASE/data_trainv2/train-main-v2.parquet \
  --bundle $BASE/bundle --out $BASE/results/trainv2_bundle/scores.jsonl 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
