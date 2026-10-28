"""Fit a probe on gpt-oss prompt hidden states to the judge's agreement score; test on train-main-v2.

  python fit_probe.py <results dir> <train-main-v2.parquet>

<results dir> holds pool_omni720/ and trainv2_36/, each with hidden.npz and judge.jsonl (judge_pool.py).
Reports, for each feature set: CV fit on the pool, then AUC against the official labels on the 71
labelled train-v2 rows (each problem's score is reused for every configuration of that problem, as
the judge does), next to the black-box judge's own AUC on the same rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

res, parquet = Path(sys.argv[1]), sys.argv[2]


def load(name):
    hid = np.load(res / name / "hidden.npz")
    ids = list(hid["ids"])
    judge = {r["id"]: r for r in map(json.loads, open(res / name / "judge.jsonl"))}
    keep = [i for i, x in enumerate(ids) if x in judge]
    feats = {k: hid[k][keep] for k in hid.files if k != "ids"}
    rows = [judge[ids[i]] for i in keep]
    return [ids[i] for i in keep], feats, rows


def target(rows, kind):
    if kind == "verdict":
        return np.array([r["verdict"] for r in rows], dtype=float)
    if kind == "distinct":
        return -np.array([r["distinct"] if r["distinct"] else 9 for r in rows], dtype=float)
    return np.array([r[kind] for r in rows], dtype=float)


pool_ids, pool_X, pool_rows = load("pool_omni720")
tv_ids, tv_X, tv_rows = load("trainv2_36")
problems = [json.loads(l)["problem"] for l in open(res.parent / "data" / "trainv2_36.jsonl")]
id_of = {p: f"tv2_{j}" for j, p in enumerate(problems)}
lab = pd.read_parquet(parquet).dropna(subset=["is_robust"])
lab["pid"] = lab["problem"].map(id_of)
y_rows = lab["is_robust"].astype(int).to_numpy()
print(f"pool {len(pool_ids)} problems, judge robust rate {np.mean([r['verdict'] for r in pool_rows]):.2f}; "
      f"train-v2 {len(tv_ids)} problems, {len(lab)} labelled rows")


def row_auc(score_by_pid):
    s = lab["pid"].map(score_by_pid).to_numpy(dtype=float)
    ok = ~np.isnan(s)
    return roc_auc_score(y_rows[ok], s[ok]), int(ok.sum())


# The black-box judge itself on these rows (its own 8 samples from this run).
for kind in ["modal_share", "distinct", "verdict"]:
    auc, n = row_auc(dict(zip(tv_ids, target(tv_rows, kind))))
    print(f"judge {kind:12s} AUC {auc:.3f} on {n} rows")

# Text-only baselines: is the hidden state telling us more than "long, LaTeX-heavy problems are hard"?
pool_problems = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open(res.parent / "data" / "pool_omni720.jsonl")}
for name, fn in [("-length", lambda p: -len(p)), ("-dollars", lambda p: -p.count("$"))]:
    auc, n = row_auc({pid: fn(p) for p, pid in id_of.items()})
    pool_r = np.corrcoef([fn(pool_problems[i]) for i in pool_ids], target(pool_rows, "modal_share"))[0, 1]
    print(f"text {name:9s} AUC {auc:.3f} on {n} rows; corr with judge modal_share on the pool {pool_r:.3f}")

cv = KFold(5, shuffle=True, random_state=0)
ensemble = {}
for kind in ["modal_share", "verdict"]:
    y = target(pool_rows, kind)
    for name in sorted(pool_X):
        X, Xt = pool_X[name], tv_X[name]
        if kind == "verdict":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
            cvp = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            cv_score = roc_auc_score(y, cvp)
            model.fit(X, y)
            s = model.predict_proba(Xt)[:, 1]
        else:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
            cvp = cross_val_predict(model, X, y, cv=cv)
            cv_score = np.corrcoef(cvp, y)[0, 1]
            model.fit(X, y)
            s = model.predict(Xt)
        auc, n = row_auc(dict(zip(tv_ids, s)))
        judge_corr = np.corrcoef(s, target(tv_rows, kind))[0, 1]
        print(f"probe {kind:11s} {name:7s} pool-CV {cv_score:.3f}  train-v2: AUC vs labels {auc:.3f} ({n} rows), "
              f"corr with judge {judge_corr:.3f}")
        # Standardise each probe's train-v2 scores with its own pool CV predictions, then average.
        ensemble.setdefault(kind, []).append((s - cvp.mean()) / cvp.std())

# Fixed in advance (not chosen on train-v2): the mean of all eight probes per target, and of all sixteen.
for kind, parts in list(ensemble.items()) + [("all", sum(ensemble.values(), []))]:
    s = np.mean(parts, axis=0)
    auc, n = row_auc(dict(zip(tv_ids, s)))
    print(f"ensemble {kind:11s} ({len(parts)} probes): train-v2 AUC vs labels {auc:.3f} ({n} rows)")
