"""Does generation length predict robustness beyond problem length?

mean_tok looked strong on the finished slice of the 8192-token rerun, but that slice is the shortest
problems (the job sorts by prompt length), and problem length is already a known correlate that costs
nothing to compute. This separates the two: length alone, generation length within length strata, and
a grouped-CV comparison of a length-only model against length plus generation length.
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
CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
cases = pd.read_csv(f"{B}/results/agreement_v2/cases.csv")
cases = cases[cases["checkpoint"] == CK].drop_duplicates("problem_id").set_index("problem_id")
new = pd.read_json(f"{B}/results/agreement_v2/generations.jsonl", lines=True)

rows = []
for pid, g in new.groupby("problem_id"):
    orig = g[g["variant"] == "orig"]
    if len(orig) < 4:
        continue
    oa = orig["answer"].dropna().tolist()
    mode = Counter(oa).most_common(1)[0][0] if oa else None
    rows.append({"problem_id": pid, "mean_tok": float(g["n_new_tokens"].mean()),
                 "hit_limit": float(g["hit_limit"].mean()),
                 "variant_agree": float((g[g["variant"] != "orig"]["answer"] == mode).mean()) if mode else 0.0})
s = pd.DataFrame(rows).set_index("problem_id").join(cases[["label_robust", "original_problem"]], how="inner")
s = s[s["label_robust"].notna()]
s["prompt_chars"] = s["original_problem"].str.len().astype(float)
y = s["label_robust"].astype(bool).astype(int).to_numpy()

allc = cases[cases["label_robust"].notna()]["original_problem"].str.len()
print(f"finished slice: n={len(s)}, robust={y.sum()} ({y.mean():.0%})")
print(f"  prompt chars, this slice : median {s['prompt_chars'].median():.0f}, "
      f"p10 {s['prompt_chars'].quantile(.1):.0f}, p90 {s['prompt_chars'].quantile(.9):.0f}")
print(f"  prompt chars, all labelled: median {allc.median():.0f}, p10 {allc.quantile(.1):.0f}, p90 {allc.quantile(.9):.0f}")
print(f"  robust rate, all labelled : {cases['label_robust'].dropna().astype(bool).mean():.0%}")

print("\nsingle-feature AUC on this slice:")
for f in ("prompt_chars", "mean_tok", "hit_limit", "variant_agree"):
    print(f"  {f:16s} {roc_auc_score(y, s[f]):.3f}")

print("\ncorrelation of mean_tok with prompt_chars:", round(float(s['mean_tok'].corr(s['prompt_chars'])), 3))
tert = pd.qcut(s["prompt_chars"], 3, labels=False, duplicates="drop").to_numpy()
inner = [roc_auc_score(y[tert == t], s["mean_tok"].to_numpy()[tert == t])
         for t in sorted(set(tert)) if len(set(y[tert == t])) == 2]
print("mean_tok AUC within prompt-length tertiles:", [round(v, 3) for v in inner],
      "mean", round(float(np.mean(inner)), 3) if inner else "n/a")

groups = s["original_problem"].to_numpy()
def cv(cols, C=0.1):
    X = s[cols].to_numpy(dtype=float)
    aucs = []
    for seed in range(5):
        k = min(5, int(np.bincount(y).min()))
        if k < 2:
            return float("nan")
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            m.fit(X[tr], y[tr]); oof[te] = m.decision_function(X[te])
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))

print("\ngrouped CV AUC:")
print(f"  prompt_chars only              {cv(['prompt_chars']):.3f}")
print(f"  prompt_chars + mean_tok        {cv(['prompt_chars', 'mean_tok']):.3f}")
print(f"  prompt_chars + mean_tok + hit  {cv(['prompt_chars', 'mean_tok', 'hit_limit']):.3f}")
print(f"  mean_tok only                  {cv(['mean_tok']):.3f}")
