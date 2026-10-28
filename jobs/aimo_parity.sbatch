#!/bin/bash
#SBATCH -J aimo-parity
#SBATCH -p ice-bw-gpu
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o logs/parity_%j.log
# One model where batching clearly helped (Qwen3.5) and one where it seemingly did not (DeepSeek).
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$PY $BASE/code/test_batch_parity.py --base $BASE --n-problems 6 \
  --models Qwen/Qwen3.5-4B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
