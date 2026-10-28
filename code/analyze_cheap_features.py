"""Grouped-CV check of the cheap forward-pass signals against official robustness labels.

Only checkpoints with official public labels are scored (DeepSeek-R1-0528-Qwen3-8B via the
qwen3-8b:low alias and the DeepSeek files, gpt-oss-120b via gpt-oss-120b:low). Folds are grouped by
problem text so that the same problem never sits on both sides of a split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def safe(name: str) -> str:
    return name.replace("/", "__")


def univariate(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = []
    y = df["derived_label"].astype(int).to_numpy()
    for f in feats:
        x = df[f].to_numpy(dtype=float)
        if np.isnan(x).any() or len(set(y)) < 2:
            continue
        auc = roc_auc_score(y, x)
        rows.append({"feature": f, "auc": auc, "auc_abs": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc_abs", ascending=False)


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, seeds=(0, 1, 2, 3, 4)) -> dict:
    aucs, accs = [], []
    for seed in seeds:
        n_splits = min(5, int(np.bincount(y).min()))
        if n_splits < 2:
            return {"auc": np.nan, "acc": np.nan}
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        accs.append(accuracy_score(y, oof >= 0.5))
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)), "acc": float(np.mean(accs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    args = ap.parse_args()
    cases = pd.read_csv(args.results_dir / "labeled_cases.csv")
    for checkpoint, sub in cases.groupby("checkpoint"):
        fpath = args.results_dir / f"features_{safe(checkpoint)}.csv"
        if not fpath.exists():
            print(f"skip {checkpoint}: no features")
            continue
        feats = pd.read_csv(fpath)
        hid = np.load(args.results_dir / f"hidden_{safe(checkpoint)}.npz")
        print(f"\n===== {checkpoint}: {len(sub)} labeled cases, robust={int(sub['derived_label'].sum())}")
        for view in ("raw", "chat"):
            fv = feats[feats["view"] == view].drop(columns=["view", "checkpoint"])
            df = sub.merge(fv, on="problem_id", how="inner")
            df["problem_chars"] = df["original_problem"].str.len()
            scalar = [c for c in fv.columns if c != "problem_id"] + ["problem_chars"]
            print(f"--- view={view} univariate AUC (robust=1)")
            print(univariate(df, scalar).head(12).to_string(index=False))
            y = df["derived_label"].astype(int).to_numpy()
            groups = df["original_problem"].to_numpy()
            print("scalar LR", grouped_cv(df[scalar].to_numpy(dtype=float), y, groups, C=0.3))
            layer_keys = sorted({k.split("|")[2] for k in hid.files if k.split("|")[1] == view})
            for lk in layer_keys:
                X = np.stack([hid[f"{pid}|{view}|{lk}"] for pid in df["problem_id"]])
                for C in (0.001, 0.01, 0.1):
                    print(f"hidden {lk} C={C}", grouped_cv(X, y, groups, C=C))
    tp = args.results_dir / "throughput.csv"
    if tp.exists():
        print("\n", pd.read_csv(tp).to_string(index=False))


if __name__ == "__main__":
    main()
