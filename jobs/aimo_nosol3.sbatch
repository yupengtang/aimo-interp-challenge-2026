#!/bin/bash
#SBATCH -J aimo-nosol
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o logs/nosol_%j.log
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('cuda smoke ok', torch.cuda.get_device_name(0))" || exit 1
cd $BASE/code && $PY nosol_experiment.py --parquet $BASE/data_trainv2/train-main-v2.parquet --out-dir $BASE/results/nosol3 --seed 2 --variants 6 --rewrite subtle 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
