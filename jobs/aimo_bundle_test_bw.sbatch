#!/bin/bash
#SBATCH -J aimo-bundle-bw
#SBATCH -p ice-bw-gpu
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o logs/bundle_bw_%j.log
# Four small checkpoints only: each fits the 48 GB MIG slice this partition hands out, and the
# partition queues separately from ice-gpu. gpt-oss needs about 70 GB and is tested on H200.
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
$PY $BASE/code/test_bundle_end_to_end.py --base $BASE --n-problems 30 \
  --models Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
