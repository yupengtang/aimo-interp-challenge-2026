#!/bin/bash
#SBATCH -J aimo-bundle-test
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o logs/bundle_test_%j.log
# H200 on purpose: gpt-oss-120b needs about 70 GB and the ice-bw-gpu Blackwell cards are handed out
# as 48 GB MIG slices, where it cannot load at all.
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
$PY $BASE/code/test_bundle_end_to_end.py --base $BASE --n-problems 30 2>&1 | grep -v "Loading weights"
echo "=== DONE $(date -Is)"
