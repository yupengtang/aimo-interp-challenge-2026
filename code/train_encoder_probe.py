"""Fit a problem-level robustness probe and export it as a submission artifact.

Training data: the official public labels for DeepSeek-R1-0528-Qwen3-8B on MATH problems (the only
labelled set large enough to fit on). Features: the last-token hidden state of the raw problem text
import os
from one encoder checkpoint, so the same score can be produced for any evaluated model.

Model selection uses grouped cross-validation on the MATH cases. The olympiad cases (10 problems x
the ":low" model runs in the official sample) are reported as an out-of-domain check only; they are
never trained on, and with 10 problems they are indicative at best.

The exported artifact holds the encoder id, layer, scaler statistics, coefficients and the decision
threshold, so inference needs one forward pass and a dot product.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def safe(name: str) -> str:
    return name.replace("/", "__")


def grouped_cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, seeds: int = 5) -> float:
    aucs = []
    for seed in range(seeds):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-positive-rate", type=float, default=0.5,
                    help="fraction of cases predicted robust; the public validation set is near balanced")
    args = ap.parse_args()

    feat_dir = f"{args.base}/results/cheap_features_h200"
    pool_dir = f"{args.base}/results/encode_olympiad_pool"
    hid = np.load(f"{feat_dir}/hidden_{safe(args.encoder)}.npz")
    pool = np.load(f"{pool_dir}/hidden_{safe(args.encoder)}.npz")

    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    oly_ids = sorted(src.loc[src["dataset_id"] == "aimo-2025-reference", "problem_id"].unique())
    cases = pd.read_csv(f"{feat_dir}/labeled_cases.csv")
    train = cases[(cases["checkpoint"] == LABEL_CK) & ~cases["problem_id"].isin(oly_ids)]
    y = train["derived_label"].astype(int).to_numpy()
    groups = train["original_problem"].to_numpy()

    oly = src[src["dataset_id"] == "aimo-2025-reference"]
    lab = oly.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0).rename("robust").reset_index()
    lab = lab[lab["model_id"] != "gpt-5.2-2025-12-11"]

    layers = sorted({k.split("|")[2] for k in hid.files})
    results = []
    for layer in layers:
        X = np.stack([hid[f"{p}|raw|{layer}"] for p in train["problem_id"]]).astype(np.float64)
        for C in (0.003, 0.01, 0.03, 0.1, 0.3):
            cv_auc = grouped_cv_auc(X, y, groups, C)
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X), y)
            X_oly = np.stack([hid[f"{p}|raw|{layer}"] for p in oly_ids]).astype(np.float64)
            s_oly = pd.Series(clf.decision_function(sc.transform(X_oly)), index=oly_ids)
            oly_auc = roc_auc_score(lab["robust"].astype(int), s_oly.loc[lab["problem_id"]].to_numpy())
            results.append({"layer": layer, "C": C, "cv_auc_math": round(cv_auc, 3), "oly_auc": round(oly_auc, 3)})
            print(json.dumps(results[-1]), flush=True)

    best = max(results, key=lambda r: r["cv_auc_math"])
    print("selected by MATH CV:", best)
    layer, C = best["layer"], best["C"]
    X = np.stack([hid[f"{p}|raw|{layer}"] for p in train["problem_id"]]).astype(np.float64)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(scaler.transform(X), y)

    # Threshold: hit the target positive rate on the unlabeled olympiad pool, which matches the test
    # domain and needs no labels. Fitting a threshold on the 10 official olympiad problems would
    # overfit them.
    X_pool = np.stack([pool[k] for k in pool.files if k.endswith(f"|raw|{layer}")]).astype(np.float64)
    pool_scores = clf.decision_function(scaler.transform(X_pool))
    threshold = float(np.quantile(pool_scores, 1.0 - args.target_positive_rate))

    X_oly = np.stack([hid[f"{p}|raw|{layer}"] for p in oly_ids]).astype(np.float64)
    s_oly = pd.Series(clf.decision_function(scaler.transform(X_oly)), index=oly_ids)
    y_oly = lab["robust"].astype(int).to_numpy()
    pred = s_oly.loc[lab["problem_id"]].to_numpy() > threshold
    print(f"olympiad check: n={len(y_oly)} robust={y_oly.sum()} auc={roc_auc_score(y_oly, s_oly.loc[lab['problem_id']].to_numpy()):.3f} "
          f"acc={accuracy_score(y_oly, pred):.3f} predicted_robust_rate={pred.mean():.2f} "
          f"(always-true acc={y_oly.mean():.3f}, always-false acc={1 - y_oly.mean():.3f})")

    artifact = {
        "encoder_model_id": args.encoder,
        "layer_index": int(layer.split("_")[1]),
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coef": clf.coef_[0],
        "intercept": float(clf.intercept_[0]),
        "threshold": threshold,
        "C": C,
        "cv_auc_math": best["cv_auc_math"],
        "olympiad_auc": float(roc_auc_score(y_oly, s_oly.loc[lab["problem_id"]].to_numpy())),
        "n_train": int(len(y)),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
