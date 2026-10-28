#!/bin/bash
#SBATCH -J aimo-selfperturb
#SBATCH -p ice-bw-gpu
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -N 1 --ntasks-per-node=1 -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o logs/selfperturb_%j.log
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$PY - <<'PYEOF' || { echo "ABORT: GPU smoke test failed"; exit 1; }
import torch
assert torch.cuda.is_available(), "no CUDA"
x = torch.randn(2048, 2048, device="cuda"); y = x @ x.T; torch.cuda.synchronize()
assert torch.isfinite(y).all().item(), "non-finite matmul"
print("cuda smoke ok:", torch.__version__, torch.cuda.get_device_name(0))
PYEOF
# gpt-oss first: its 20 labeled olympiad cases are the closest match to the private test problems.
$PY $BASE/code/aimo_selfperturb.py --data-dir $BASE/data --out-dir $BASE/results/selfperturb \
  --models openai/gpt-oss-120b deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think 2>&1
echo "=== DONE $(date -Is)"
