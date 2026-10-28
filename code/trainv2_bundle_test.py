"""Run a submission bundle the way the harness does, on train-main-v2, and score it."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

bundle, parquet, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(bundle))
t0 = time.time()
import solution  # noqa: E402

df = pd.read_parquet(parquet)
rows = []
for (model_id, effort), g in df.groupby(["model_id", "reasoning_effort"], sort=False):
    t = time.time()
    pred = solution.are_robust(model_id, reasoning_effort=effort, problems=g.problem.tolist())
    assert type(pred) is list and len(pred) == len(g) and all(type(x) is bool for x in pred)
    print(f"{model_id} {effort}: {len(g)} problems in {time.time() - t:.0f}s, robust rate {sum(pred) / len(pred):.2f}",
          flush=True)
    for (_, r), p in zip(g.iterrows(), pred):
        rows.append({"model_id": model_id, "effort": effort, "problem": r.problem, "pred": p,
                     "is_robust": None if pd.isna(r.is_robust) else bool(r.is_robust)})
total = time.time() - t0
res = pd.DataFrame(rows)
res.to_json(out, orient="records", lines=True)
lab = res[res.is_robust.notna()]
acc = (lab.pred == lab.is_robust.astype(bool)).mean()
y = lab.is_robust.astype(bool)
bal = 0.5 * ((lab.pred[y]).mean() + (~lab.pred[~y]).mean())
print(f"TOTAL {total:.0f}s | stats {getattr(solution, 'stats', None)}")
import torch
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print(f"labelled n={len(lab)} acc={acc:.3f} balanced_acc={bal:.3f} always-true={y.mean():.3f} pred robust rate={lab.pred.mean():.2f}")
