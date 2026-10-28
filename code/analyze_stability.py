"""Score representation-stability features against the official labels.

Two evaluation sets:
  * MATH: 154 DeepSeek cases, grouped CV by problem text, the only set large enough to fit on;
  * olympiad: 10 problems x the 7 official ":low" runs (70 cases), scored with a problem-level score,
    which is the closest available proxy for the private test set.

Reported per checkpoint: the AUC of every single stability feature, a cross-validated combination of
them, and whether adding stability to the Qwen3.5 probe score helps.
"""

from __future__ import annotations

import os
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def safe(name: str) -> str:
    return name.replace("/", "__")


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, seeds: int = 5) -> float:
    aucs = []
    for seed in range(seeds):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.decision_function(X[te])
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--stability-dir", default=None)
    args = ap.parse_args()
    sdir = args.stability_dir or f"{args.base}/results/stability"

    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    oly = src[src["dataset_id"] == "aimo-2025-reference"]
    lab_oly = (
        oly.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0)
        .rename("robust").reset_index()
    )
    lab_oly = lab_oly[lab_oly["model_id"] != "gpt-5.2-2025-12-11"]
    oly_ids = sorted(lab_oly["problem_id"].unique())
    cases = pd.read_csv(f"{args.base}/results/cheap_features_h200/labeled_cases.csv")
    math = cases[(cases["checkpoint"] == LABEL_CK) & ~cases["problem_id"].isin(oly_ids)]

    for path in sorted(glob.glob(f"{sdir}/stability_*.csv")):
        st = pd.read_csv(path)
        ck = st["checkpoint"].iloc[0]
        feats = [c for c in st.columns if c.startswith(("cos_", "rel_l2_", "kl_", "js_", "dtok_"))]
        st = st.drop_duplicates("id").set_index("id")
        print(f"\n===== {ck}: {len(st)} problems, {len(feats)} features")

        m = math.merge(st[feats], left_on="problem_id", right_index=True, how="inner")
        y_m = m["derived_label"].astype(int).to_numpy()
        g_m = m["original_problem"].to_numpy()
        singles = []
        for f in feats:
            x = m[f].to_numpy(dtype=float)
            if np.isfinite(x).all() and np.ptp(x) > 0:
                singles.append((f, roc_auc_score(y_m, x)))
        singles.sort(key=lambda kv: -abs(kv[1] - 0.5))
        print("MATH n=%d robust=%d | best single features:" % (len(y_m), y_m.sum()))
        for f, a in singles[:8]:
            print(f"   {f:34s} {a:.3f}")
        X_m = m[feats].to_numpy(dtype=float)
        for C in (0.01, 0.1, 1.0):
            print(f"   MATH combined LR C={C}: AUC {cv_auc(X_m, y_m, g_m, C):.3f}")

        have = [p for p in oly_ids if p in st.index]
        if len(have) == len(oly_ids):
            y_o = lab_oly["robust"].astype(int).to_numpy()
            print(f"olympiad n={len(y_o)} robust={y_o.sum()} | single-feature AUC (problem-level score):")
            rows = []
            for f in feats:
                s = st.loc[lab_oly["problem_id"], f].to_numpy(dtype=float)
                if np.isfinite(s).all() and np.ptp(s) > 0:
                    rows.append((f, roc_auc_score(y_o, s)))
            rows.sort(key=lambda kv: -abs(kv[1] - 0.5))
            for f, a in rows[:8]:
                print(f"   {f:34s} {a:.3f}")
            # train the combination on MATH, apply to olympiad: the honest out-of-domain check
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
            clf.fit(X_m, y_m)
            s_o = pd.Series(clf.decision_function(st.loc[oly_ids, feats].to_numpy(dtype=float)), index=oly_ids)
            print(f"   MATH-trained combination on olympiad: AUC {roc_auc_score(y_o, s_o.loc[lab_oly['problem_id']].to_numpy()):.3f}")
        else:
            print(f"olympiad: missing {len(oly_ids) - len(have)} problems in this file")


if __name__ == "__main__":
    main()
