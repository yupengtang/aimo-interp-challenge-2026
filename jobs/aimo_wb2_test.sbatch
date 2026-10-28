#!/bin/bash
#SBATCH -J aimo-wb2-test
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=160G
#SBATCH -t 01:30:00
#SBATCH -o logs/wb2_test_%j.log
# Export the probe ensemble from the finished pool, then run bundle_wb2 the way the harness does on train-v2.
set -uo pipefail
A=${AIMO_BASE:?set AIMO_BASE to your working directory}
export HF_HOME=$A/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$A/xdg_cache TRITON_CACHE_DIR=$A/triton_cache OMP_NUM_THREADS=8
PY=$A/runtime_venv/bin/python
$PY -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print('cuda smoke ok', torch.cuda.get_device_name(0))" || exit 1
wc -l $A/distill/results/pool_omni720/judge.jsonl
$PY $A/distill/export_probe.py $A/distill/results 0.5 $A/bundle_wb2/probe.npz || exit 1
mkdir -p $A/results/wb2
$PY $A/code/trainv2_bundle_test.py $A/bundle_wb2 $A/data_trainv2/train-main-v2.parquet $A/results/wb2/run_$SLURM_JOB_ID.jsonl 2>&1 | grep -v "Loading weights"
# The verdicts from the bundle must equal the offline ensemble scores against the same threshold.
$PY - <<'EOF'
import json, os, numpy as np, pandas as pd
A = os.environ["AIMO_BASE"]
run = sorted(f for f in os.listdir(f"{A}/results/wb2") if f.startswith("run_"))[-1]
pred = pd.read_json(f"{A}/results/wb2/{run}", lines=True)
offline = json.load(open(f"{A}/bundle_wb2/probe.trainv2_scores.json"))
thr = float(np.load(f"{A}/bundle_wb2/probe.npz")["threshold"])
probs = [json.loads(l)["problem"] for l in open(f"{A}/distill/data/trainv2_36.jsonl")]
pid = {p: f"tv2_{j}" for j, p in enumerate(probs)}
pred["offline"] = pred.problem.map(lambda p: offline[pid[p]] >= thr)
print("bundle verdicts equal offline ensemble:", int((pred.pred == pred.offline).sum()), "of", len(pred))
EOF
echo "=== DONE $(date -Is)"
