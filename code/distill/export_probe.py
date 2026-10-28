"""Fit the probe ensemble on the whole pool and write probe.npz for bundle_wb2.

  python export_probe.py <results dir> <robust rate> <out probe.npz>

The ensemble is fixed in advance: one ridge probe per feature (last-token and mean hidden state at
layers 9, 18, 27, 36) predicting the judge's modal-answer share; each probe's score is standardised
with the mean and spread of its 5-fold pool predictions, and the verdict uses the average. The
threshold calls the given share of the pool robust, using the cross-validated ensemble scores.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

res, rate, out = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
FEATURES = [f"{pool}{layer}" for layer in (9, 18, 27, 36) for pool in ("last", "mean")]

hid = np.load(res / "pool_omni720" / "hidden.npz")
judge = {r["id"]: r for r in map(json.loads, open(res / "pool_omni720" / "judge.jsonl"))}
keep = [i for i, x in enumerate(hid["ids"]) if x in judge]
y = np.array([judge[hid["ids"][i]]["modal_share"] for i in keep], dtype=float)
tv = np.load(res / "trainv2_36" / "hidden.npz")

cv = KFold(5, shuffle=True, random_state=0)
params = {"features": np.array(FEATURES)}
cv_parts, tv_parts = [], []
for k, name in enumerate(FEATURES):
    X = hid[name][keep]
    cvp = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11))), X, y, cv=cv)
    scaler = StandardScaler().fit(X)
    ridge = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(scaler.transform(X), y)
    params[f"mean_{k}"] = scaler.mean_.astype(np.float32)
    params[f"scale_{k}"] = scaler.scale_.astype(np.float32)
    params[f"coef_{k}"] = ridge.coef_.astype(np.float32)
    params[f"intercept_{k}"] = np.float32(ridge.intercept_)
    params[f"center_{k}"] = np.float32(cvp.mean())
    params[f"spread_{k}"] = np.float32(cvp.std())
    cv_parts.append((cvp - cvp.mean()) / cvp.std())
    s = scaler.transform(tv[name]) @ ridge.coef_ + ridge.intercept_
    tv_parts.append((s - cvp.mean()) / cvp.std())
    print(f"{name}: alpha {ridge.alpha_:g}, pool CV r {np.corrcoef(cvp, y)[0, 1]:.3f}", flush=True)

cv_score = np.mean(cv_parts, axis=0)
params["threshold"] = np.float32(np.quantile(cv_score, 1.0 - rate))
np.savez(out, **params)
tv_score = np.mean(tv_parts, axis=0)
print(f"{len(y)} pool problems; ensemble pool CV r {np.corrcoef(cv_score, y)[0, 1]:.3f}; "
      f"threshold {float(params['threshold']):.4f}; train-v2 called robust {np.mean(tv_score >= params['threshold']):.2f}")
json.dump({i: float(s) for i, s in zip(tv["ids"].tolist(), tv_score)},
          open(Path(out).with_suffix(".trainv2_scores.json"), "w"))
