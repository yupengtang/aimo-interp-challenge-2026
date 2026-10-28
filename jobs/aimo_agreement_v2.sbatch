#!/bin/bash
#SBATCH -J aimo-agreement-v2
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/agreement_v2_%j.log
# Redo of the agreement test without the truncation confound: 8192 tokens instead of 4096, static
# cache, DeepSeek only, and only the MATH sets, which hold 154 labelled cases over 87 problems.
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
$PY $BASE/code/aimo_selfperturb.py --data-dir $BASE/data --out-dir $BASE/results/agreement_v2 \
  --models deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --datasets hendrycks-math-train-reference-500 hendrycks-math-test-reference-100 \
  --max-new-tokens 8192 --batch-size 32 --cache-implementation static 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
