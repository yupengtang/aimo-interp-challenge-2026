#!/bin/bash
#SBATCH -J tv2-gptoss
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -o logs/tv2gen_gptoss_%j.log
# gpt-oss-120b at each reasoning effort on the train-main-v2 problems. Resumable per effort.
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('cuda smoke ok', torch.cuda.get_device_name(0))" || exit 1
cd $BASE/code
for E in low medium high; do
  echo "=== effort $E $(date -Is)"
  $PY trainv2_generate.py --parquet $BASE/data_trainv2/train-main-v2.parquet --out-dir $BASE/results/trainv2_gen \
    --model openai/gpt-oss-120b --effort $E --samples 4 --batch-size 24 --cache-implementation dynamic 2>&1 | grep -v "Loading weights"
done
echo "=== DONE $(date -Is)"
