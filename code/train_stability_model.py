"""Fit the cross-model stability classifier and export a submission artifact.

Why this can transfer across checkpoints: the features are dimensionless comparisons of a model
against itself (cosine similarity and KL divergence between the original problem and a rewrite that
cannot change the answer), so a coefficient vector learned on one model is meaningful for another.
Raw hidden states are not comparable that way, which is why the earlier probes failed to transfer.

Training labels: the official public labels for DeepSeek-R1-0528-Qwen3-8B on MATH problems, which is
the only labelled set large enough to fit on. No labels are generated here.

Per model, the artifact also stores standardisation statistics and a decision threshold computed from
unlabeled public problems, so each checkpoint's scores are centred on its own distribution.
"""

from __future__ import annotations

import os
import argparse
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b",
]
# Depth is expressed as a fraction of the model's layers so one coefficient vector fits every model.
DEPTHS = ["q1", "q2", "q3", "q4"]
FAMILIES = ["typos", "distract", "wrap"]


def safe(name: str) -> str:
    return name.replace("/", "__")


def to_depth_features(st: pd.DataFrame) -> pd.DataFrame:
    """Rename this model's absolute layer indices to quartile names, and keep family aggregates."""
    layers = sorted({int(c.split("_")[1][1:]) for c in st.columns if c.startswith("cos_l")})
    mapping = {l: d for l, d in zip(layers, DEPTHS)}
    out = pd.DataFrame(index=st.index)
    for fam in FAMILIES:
        for l, depth in mapping.items():
            for stat in ("mean", "min"):
                col = f"cos_l{l}_{fam}_{stat}"
                if col in st.columns:
                    out[f"cos_{depth}_{fam}_{stat}"] = st[col].astype(float)
        for stat in ("mean", "min"):
            col = f"kl_{fam}_{stat}"
            if col in st.columns:
                out[f"kl_{fam}_{stat}"] = st[col].astype(float)
    out["prompt_tokens"] = st["prompt_tokens"].astype(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-positive-rate", type=float, default=0.5)
    ap.add_argument("--drop-prefix", action="append", default=[],
                    help="leave out features whose name starts with this prefix (repeatable); "
                         "--drop-prefix cos_ fits the ablation without the hidden-state features")
    args = ap.parse_args()

    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    oly = src[src["dataset_id"] == "aimo-2025-reference"]
    lab = (
        oly.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0)
        .rename("robust").reset_index()
    )
    lab = lab[lab["model_id"] != "gpt-5.2-2025-12-11"]
    oly_ids = sorted(lab["problem_id"].unique())
    cases = pd.read_csv(f"{args.base}/results/cheap_features_h200/labeled_cases.csv")
    train_cases = cases[(cases["checkpoint"] == LABEL_CK) & ~cases["problem_id"].isin(oly_ids)]

    st_label = pd.read_csv(f"{args.base}/results/stability/stability_{safe(LABEL_CK)}.csv").drop_duplicates("id").set_index("id")
    F_label = to_depth_features(st_label)
    F_label = F_label[[c for c in F_label.columns if not c.startswith(tuple(args.drop_prefix))]] if args.drop_prefix else F_label
    feature_names = list(F_label.columns)
    print(f"{len(feature_names)} features: {feature_names}")

    X = F_label.loc[train_cases["problem_id"]].to_numpy(dtype=float)
    y = train_cases["derived_label"].astype(int).to_numpy()
    groups = train_cases["original_problem"].to_numpy()
    scaler = StandardScaler().fit(X)

    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        aucs = []
        for seed in range(5):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
                oof[te] = clf.decision_function(sc.transform(X[te]))
            aucs.append(roc_auc_score(y, oof))
        auc = float(np.mean(aucs))
        print(json.dumps({"C": C, "math_cv_auc": round(auc, 3)}), flush=True)
        if best is None or auc > best[1]:
            best = (C, auc)
    C, cv_auc = best
    print(f"selected C={C} with MATH CV AUC {cv_auc:.3f}")
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(scaler.transform(X), y)

    per_model = {}
    for ck in MODELS:
        st = pd.read_csv(f"{args.base}/results/stability/stability_{safe(ck)}.csv").drop_duplicates("id").set_index("id")
        F = to_depth_features(st)[feature_names]
        pool = F.loc[[i for i in F.index if i not in set(cases["problem_id"]) and i not in oly_ids]]
        # Centre each checkpoint on its own unlabeled distribution, then fix its threshold there.
        model_scaler = StandardScaler().fit(pool.to_numpy(dtype=float))
        pool_scores = clf.decision_function(model_scaler.transform(pool.to_numpy(dtype=float)))
        threshold = float(np.quantile(pool_scores, 1.0 - args.target_positive_rate))
        per_model[ck] = {
            "mean": model_scaler.mean_, "scale": model_scaler.scale_,
            "threshold": threshold, "n_pool": int(len(pool)),
        }
        s_oly = pd.Series(
            clf.decision_function(model_scaler.transform(F.loc[oly_ids].to_numpy(dtype=float))), index=oly_ids
        )
        auc_all = roc_auc_score(lab["robust"].astype(int), s_oly.loc[lab["problem_id"]].to_numpy())
        pred_rate = float((s_oly.loc[lab["problem_id"]].to_numpy() > threshold).mean())
        acc = float(((s_oly.loc[lab["problem_id"]].to_numpy() > threshold) == lab["robust"].to_numpy()).mean())
        print(f"{ck:40s} pool={len(pool):4d} olympiad_auc={auc_all:.3f} acc={acc:.3f} predicted_robust={pred_rate:.2f}")

    artifact = {
        "feature_names": feature_names,
        "depths": DEPTHS,
        "families": FAMILIES,
        "train_mean": scaler.mean_, "train_scale": scaler.scale_,
        "coef": clf.coef_[0], "intercept": float(clf.intercept_[0]),
        "C": C, "math_cv_auc": cv_auc,
        "per_model": per_model,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "n_train": int(len(y)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
