"""Interim read of the 8192-token agreement rerun, using whatever sequences have finished.

Compares against the earlier 4096-token run on the same problems, so the truncation confound is
quantified rather than asserted, then scores the agreement features against the official DeepSeek
MATH labels with cross-validation grouped by problem text.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

B = os.environ.get("AIMO_BASE", "/path/to/aimo")
NEW = f"{B}/results/agreement_v2/generations.jsonl"
OLD = f"{B}/results/selfperturb_h200/generations.jsonl"
CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

cases = pd.read_csv(f"{B}/results/agreement_v2/cases.csv")
cases = cases[cases["checkpoint"] == CK].drop_duplicates("problem_id").set_index("problem_id")

new = pd.read_json(NEW, lines=True)
old = pd.read_json(OLD, lines=True)
old = old[old["checkpoint"] == CK]
shared = sorted(set(new["problem_id"]) & set(old["problem_id"]))
print(f"sequences so far: {len(new)}; problems touched: {new['problem_id'].nunique()}; shared with 4096 run: {len(shared)}")

a = old[old["problem_id"].isin(shared)]
b = new[new["problem_id"].isin(shared)]
print("\ntruncation, same problems, 4096 versus 8192 tokens:")
print(pd.DataFrame({
    "4096": {"no_answer": a["answer"].isna().mean(), "hit_limit": a["hit_limit"].mean(), "mean_tokens": a["n_new_tokens"].mean()},
    "8192": {"no_answer": b["answer"].isna().mean(), "hit_limit": b["hit_limit"].mean(), "mean_tokens": b["n_new_tokens"].mean()},
}).round(3).to_string())

rows = []
for pid, g in new.groupby("problem_id"):
    orig = g[g["variant"] == "orig"]
    var = g[g["variant"] != "orig"]
    oa = orig["answer"].dropna().tolist()
    if len(orig) < 4 or len(oa) == 0 or len(var) == 0:
        continue
    mode, mode_n = Counter(oa).most_common(1)[0]
    va = var["answer"]
    rows.append({
        "problem_id": pid,
        "orig_consistency": mode_n / len(orig),
        "orig_answered": len(oa) / len(orig),
        "n_distinct_orig": len(set(oa)),
        "variant_agree": float((va == mode).mean()),
        "variant_agree_answered": float((va.dropna() == mode).mean()) if va.notna().any() else np.nan,
        "mean_tok": float(g["n_new_tokens"].mean()),
        "hit_limit": float(g["hit_limit"].mean()),
    })
s = pd.DataFrame(rows).set_index("problem_id").join(cases[["label_robust", "original_problem"]], how="inner")
s = s[s["label_robust"].notna()]
y = s["label_robust"].astype(bool).astype(int).to_numpy()
print(f"\nproblems with complete originals and a label: {len(s)}, robust {y.sum()}")
if len(set(y)) == 2 and len(s) >= 20:
    feats = ["orig_consistency", "orig_answered", "n_distinct_orig", "variant_agree",
             "variant_agree_answered", "mean_tok", "hit_limit"]
    for f in feats:
        col = s[f]
        m = col.notna()
        if m.sum() > 10 and len(set(y[m.to_numpy()])) == 2 and col[m].nunique() > 1:
            print(f"  {f:24s} AUC {roc_auc_score(y[m.to_numpy()], col[m]):.3f}  (n={int(m.sum())})")
    use = ["orig_consistency", "variant_agree", "mean_tok", "hit_limit"]
    X = s[use].to_numpy(dtype=float)
    groups = s["original_problem"].to_numpy()
    for C in (0.03, 0.1, 1.0):
        aucs = []
        for seed in range(5):
            cv = StratifiedGroupKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
                clf.fit(X[tr], y[tr])
                oof[te] = clf.decision_function(X[te])
            aucs.append(roc_auc_score(y, oof))
        print(f"  combined LR C={C}: grouped CV AUC {np.mean(aucs):.3f}")
else:
    print("  not enough finished problems yet for a stable estimate")
