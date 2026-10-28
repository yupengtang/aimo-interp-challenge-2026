#!/bin/bash
#SBATCH -J aimo-genbench-h200
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 16
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -o logs/genbench_h200_%j.log
# Blackwell on purpose: step overhead and compile behaviour are what the eval worker will see.
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
$PY $BASE/code/aimo_gen_bench.py --data-dir $BASE/data --out $BASE/results/genbench_h200.jsonl \
  --models openai/gpt-oss-120b allenai/Olmo-3-7B-Think Skywork/Skywork-OR1-Math-7B 2>&1
echo "=== DONE $(date -Is)"
