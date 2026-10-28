"""Train a DeepSeek hidden-state probe on MATH cases only, score the olympiad problems.

The same problem-level score is compared against the olympiad labels of DeepSeek (qwen3-8b:low alias)
and of gpt-oss-120b, to see whether a probe learned on easy problems and one model says anything about
hard problems and another model. Olympiad cases are never in the training set.
"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B = os.environ.get("AIMO_BASE", "/path/to/aimo")
R = f"{B}/results/cheap_features_h200"
CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
hid = np.load(f"{R}/hidden_{CK.replace('/', '__')}.npz")
cases = pd.read_csv(f"{R}/labeled_cases.csv")
src = pd.concat([pd.read_parquet(p) for p in __import__("glob").glob(f"{B}/data/*.parquet")])
olympiad_ids = set(src.loc[src["dataset_id"] == "aimo-2025-reference", "problem_id"])

ds = cases[cases["checkpoint"] == CK]
train = ds[~ds["problem_id"].isin(olympiad_ids)]
print("train MATH cases", len(train), "robust", int(train["derived_label"].sum()))
for layer in ("layer_27", "layer_36", "layer_18"):
    X = np.stack([hid[f"{p}|raw|{layer}"] for p in train["problem_id"]])
    y = train["derived_label"].astype(int).to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")).fit(X, y)
    for ck in (CK, "openai/gpt-oss-120b"):
        te = cases[(cases["checkpoint"] == ck) & cases["problem_id"].isin(olympiad_ids)]
        Xt = np.stack([hid[f"{p}|raw|{layer}"] for p in te["problem_id"]])
        s = clf.predict_proba(Xt)[:, 1]
        yt = te["derived_label"].astype(int).to_numpy()
        auc = roc_auc_score(yt, s) if len(set(yt)) == 2 else float("nan")
        print(f"{layer} -> {ck.split('/')[-1]} olympiad n={len(yt)} robust={yt.sum()} AUC={auc:.3f}",
              "scores:", dict(zip(te["problem_id"], np.round(s, 2))), "labels:", dict(zip(te["problem_id"], yt)))
