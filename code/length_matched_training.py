"""Does training on the longer MATH problems transfer better to olympiad problems?

Short MATH problems are mostly labelled non-robust (9% robust in the short tail against 24% overall),
and on them a free scalar, problem length, beats every learned feature. That suggests the label there
encodes triviality rather than a robustness mechanism, which would explain why a classifier fit on all
of MATH does not transfer to olympiad problems. This refits the same stability classifier on
increasingly length-matched subsets and reports olympiad transfer for each.
"""
import os
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.environ.get("AIMO_BASE", "/path/to/aimo") + "/code")
from train_stability_model import LABEL_CK, MODELS, safe, to_depth_features

B = os.environ.get("AIMO_BASE", "/path/to/aimo")

src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{B}/data/*.parquet")])
oly = src[src["dataset_id"] == "aimo-2025-reference"]
lab = (oly.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0)
       .rename("robust").reset_index())
lab = lab[lab["model_id"] != "gpt-5.2-2025-12-11"]
oly_ids = sorted(lab["problem_id"].unique())
y_oly = lab["robust"].astype(int).to_numpy()

cases = pd.read_csv(f"{B}/results/cheap_features_h200/labeled_cases.csv")
train_all = cases[(cases["checkpoint"] == LABEL_CK) & ~cases["problem_id"].isin(oly_ids)].copy()
train_all["chars"] = train_all["original_problem"].str.len()
oly_chars = oly.drop_duplicates("problem_id")["original_problem"].str.len()
print(f"MATH labelled: n={len(train_all)}, robust={int(train_all['derived_label'].sum())}, "
      f"median chars={train_all['chars'].median():.0f}")
print(f"olympiad problems: n={len(oly_ids)}, median chars={oly_chars.median():.0f}, min={oly_chars.min()}")

st_label = pd.read_csv(f"{B}/results/stability/stability_{safe(LABEL_CK)}.csv").drop_duplicates("id").set_index("id")
F_label = to_depth_features(st_label)
names = list(F_label.columns)

cutoffs = [("all", 0), ("chars>=median", train_all["chars"].median()),
           ("chars>=p75", train_all["chars"].quantile(.75)),
           ("chars>=olympiad min", float(oly_chars.min()))]
for label, cut in cutoffs:
    sub = train_all[train_all["chars"] >= cut]
    y = sub["derived_label"].astype(int).to_numpy()
    if len(set(y)) < 2 or len(sub) < 25:
        print(f"{label:22s} n={len(sub):3d} too small or single class")
        continue
    X = F_label.loc[sub["problem_id"]].to_numpy(dtype=float)
    groups = sub["original_problem"].to_numpy()
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.003, max_iter=5000, class_weight="balanced").fit(scaler.transform(X), y)
    aucs = []
    for seed in range(5):
        k = min(5, int(np.bincount(y).min()))
        if k < 2:
            break
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=0.003, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.decision_function(sc.transform(X[te]))
        aucs.append(roc_auc_score(y, oof))
    line = f"{label:22s} n={len(sub):3d} robust={y.sum():3d} math_cv={np.mean(aucs):.3f}" if aucs else f"{label:22s} n={len(sub)}"
    for ck in MODELS:
        st = pd.read_csv(f"{B}/results/stability/stability_{safe(ck)}.csv").drop_duplicates("id").set_index("id")
        F = to_depth_features(st)[names]
        pool = F.loc[[i for i in F.index if i not in set(cases["problem_id"]) and i not in oly_ids]]
        ms = StandardScaler().fit(pool.to_numpy(dtype=float))
        s = pd.Series(clf.decision_function(ms.transform(F.loc[oly_ids].to_numpy(dtype=float))), index=oly_ids)
        line += f" | {ck.split('/')[-1][:12]}={roc_auc_score(y_oly, s.loc[lab['problem_id']].to_numpy()):.2f}"
    print(line, flush=True)
