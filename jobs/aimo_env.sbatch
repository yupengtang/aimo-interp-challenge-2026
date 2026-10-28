#!/bin/bash
#SBATCH -J aimo-env
#SBATCH -N 1 --ntasks-per-node=1 -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/env_%j.log
set -uo pipefail
BASE=${AIMO_BASE:?set AIMO_BASE to your working directory}
export UV_CACHE_DIR=$BASE/uv_cache XDG_CACHE_HOME=$BASE/xdg_cache
UV=uv
V=$BASE/runtime_venv
$UV venv -q --python 3.11 $V
ok=0
for idx in https://download.pytorch.org/whl/cu132 https://download.pytorch.org/whl/cu130; do
  echo "=== trying torch 2.12.1 from $idx"
  if $UV pip install -q --python $V/bin/python "torch==2.12.1" --index-url $idx; then ok=1; echo "torch from $idx"; break; fi
done
[ $ok = 1 ] || { echo "ABORT: torch 2.12.1 not installable"; exit 1; }
$UV pip install -q --python $V/bin/python \
  "accelerate==1.13.0" "huggingface-hub==1.22.0" "joblib==1.5.3" "pandas==3.0.3" "safetensors==0.8.0" \
  "scikit-learn==1.8.0" "tokenizers==0.22.2" "transformers==5.13.0" "kernels==0.15.2" pyarrow
$V/bin/python -c "import sklearn, torch, transformers, accelerate, pandas; from transformers import Qwen3ForCausalLM; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'sklearn', sklearn.__version__, 'pandas', pandas.__version__)"
$V/bin/python -m pip list 2>/dev/null | head -0
$UV pip freeze --python $V/bin/python > $BASE/runtime_venv_freeze.txt
echo "=== DONE $(date -Is)"
