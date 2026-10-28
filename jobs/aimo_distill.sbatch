#!/bin/bash
#SBATCH -J aimo-distill
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=160G
#SBATCH -t 10:00:00
#SBATCH -o logs/distill_%j.log
# Judge targets + gpt-oss encodings: train-v2's 36 problems first (evaluation), then the Omni-MATH pool.
# Resumable: rerun the same command and finished problems are skipped.
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$BASE/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$BASE/xdg_cache TRITON_CACHE_DIR=$BASE/triton_cache
PY=$BASE/runtime_venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('cuda smoke ok', torch.cuda.get_device_name(0))" || exit 1
D=$BASE/distill
cd $D
$PY - <<'EOF'
import json, os, sys, time
from pathlib import Path
sys.argv = ["judge_pool.py"]
import judge_pool as jp
judge = jp.judge
assert judge._load_judge(), "judge failed to load"
D = Path(os.environ["AIMO_BASE"]) / "distill"
for name in ["trainv2_36", "pool_omni720"]:
    out = D / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    problems = [json.loads(l) for l in open(D / "data" / f"{name}.jsonl")]
    if not (out / "hidden.npz").exists():
        jp.encode(problems, out / "hidden.npz")
for name in ["trainv2_36", "pool_omni720"]:
    out = D / "results" / name
    problems = [json.loads(l) for l in open(D / "data" / f"{name}.jsonl")]
    jp.generate(problems, out / "judge.jsonl")
print("ALL DONE", flush=True)
EOF
echo "=== DONE $(date -Is)"
