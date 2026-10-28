"""Is the hidden-state probe more than a problem-length proxy? DeepSeek labeled cases only."""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = os.environ.get("AIMO_BASE", "/path/to/aimo") + "/results/cheap_features_h200"
CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
cases = pd.read_csv(f"{R}/labeled_cases.csv")
cases = cases[cases["checkpoint"] == CK]
feats = pd.read_csv(f"{R}/features_{CK.replace('/', '__')}.csv")
raw = feats[feats["view"] == "raw"][["problem_id", "n_tokens"]]
df = cases.merge(raw, on="problem_id")
df["log_chars"] = np.log(df["original_problem"].str.len())
df["log_tokens"] = np.log(df["n_tokens"])
hid = np.load(f"{R}/hidden_{CK.replace('/', '__')}.npz")
y = df["derived_label"].astype(int).to_numpy()
groups = df["original_problem"].to_numpy()


def oof_scores(X, C, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


L = df[["log_chars", "log_tokens"]].to_numpy()
tert = pd.qcut(df["log_chars"], 3, labels=False).to_numpy()
print("n", len(y), "robust", int(y.sum()), "by dataset:", df.groupby("problem_id").size().shape[0])
print("robust rate by length tertile:", [round(float(y[tert == t].mean()), 2) for t in range(3)])
for name, layer in [("layer_27", "layer_27"), ("layer_36", "layer_36")]:
    H = np.stack([hid[f"{p}|raw|{layer}"] for p in df["problem_id"]])
    res = {"len_only": [], "hidden": [], "hidden+len": [], "hidden_within_tertile": []}
    for seed in range(5):
        s_len = oof_scores(L, 1.0, seed)
        s_h = oof_scores(H, 0.1, seed)
        s_hl = oof_scores(np.hstack([H, L * 10]), 0.1, seed)
        res["len_only"].append(roc_auc_score(y, s_len))
        res["hidden"].append(roc_auc_score(y, s_h))
        res["hidden+len"].append(roc_auc_score(y, s_hl))
        within = [roc_auc_score(y[tert == t], s_h[tert == t]) for t in range(3) if len(set(y[tert == t])) == 2]
        res["hidden_within_tertile"].append(float(np.mean(within)))
    print(name, {k: round(float(np.mean(v)), 3) for k, v in res.items()})
