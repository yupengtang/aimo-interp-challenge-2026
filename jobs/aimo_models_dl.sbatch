#!/bin/bash
#SBATCH -J aimo-models-dl
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/models_dl_%j.log
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home
export UV_CACHE_DIR=$BASE/uv_cache
export XDG_CACHE_HOME=$BASE/xdg_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p $HF_HOME $UV_CACHE_DIR $XDG_CACHE_HOME
UV=uv
$UV venv -q --python 3.11 $BASE/dl_venv 2>/dev/null || true
$UV pip install -q --python $BASE/dl_venv/bin/python "huggingface_hub>=1.0" 
for m in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B openai/gpt-oss-120b; do
  echo "=== $(date -Is) downloading $m"
  for attempt in 1 2 3; do
    $BASE/dl_venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py','tokenizer*'], max_workers=8)
print('ok', p)
" && break
    echo "retry $attempt for $m"; sleep 30
  done
done
echo "=== $(date -Is) sizes"; du -sh $HF_HOME/hub/models--* 2>/dev/null
echo "=== DONE"
