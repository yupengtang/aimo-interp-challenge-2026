"""Does label-free domain standardisation rescue the MATH-trained probe on olympiad problems?

Train: DeepSeek robustness labels on the 154 MATH cases, features = last-token hidden state of the raw
problem from one encoder checkpoint (problem-level probe). Test: the 10 official olympiad problems,
scored against the labels of the 7 ":low" model runs in the official sample (70 cases) and against
the two open-weight runs that match competition checkpoints (20 cases).

Two ways to standardise the test encodings:
  * "math": the scaler fitted on the MATH training encodings (the plain transfer that collapsed);
  * "pool": a scaler fitted on 216 unlabeled public olympiad problems (no labels involved), which
    removes the domain mean/scale shift before the MATH-trained linear map is applied.
"""

from __future__ import annotations

import os
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

B = os.environ.get("AIMO_BASE", "/path/to/aimo")
FEAT = f"{B}/results/cheap_features_h200"
POOL = f"{B}/results/encode_olympiad_pool"
LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
ENCODERS = [
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "openai/gpt-oss-120b",
]


def safe(n: str) -> str:
    return n.replace("/", "__")


def main() -> None:
    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{B}/data/*.parquet")])
    olympiad = src[src["dataset_id"] == "aimo-2025-reference"]
    oly_ids = sorted(olympiad["problem_id"].unique())
    lab = (
        olympiad.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0)
        .rename("robust").reset_index()
    )
    lab = lab[lab["model_id"] != "gpt-5.2-2025-12-11"]  # keep only the ":low" runs, one row per model family
    chars = olympiad.drop_duplicates("problem_id").set_index("problem_id")["original_problem"].str.len()

    cases = pd.read_csv(f"{FEAT}/labeled_cases.csv")
    train = cases[(cases["checkpoint"] == LABEL_CK) & ~cases["problem_id"].isin(oly_ids)]
    y_tr = train["derived_label"].astype(int).to_numpy()

    y_all = lab["robust"].astype(int).to_numpy()
    print(f"train MATH n={len(y_tr)} robust={y_tr.sum()} | test olympiad cases n={len(lab)} robust={y_all.sum()}")
    print(f"problem length AUC on the same test cases: {roc_auc_score(y_all, chars.loc[lab['problem_id']].to_numpy()):.3f}")

    for enc in ENCODERS:
        hid = np.load(f"{FEAT}/hidden_{safe(enc)}.npz")
        pool = np.load(f"{POOL}/hidden_{safe(enc)}.npz")
        layers = sorted({k.split("|")[2] for k in hid.files})
        for layer in layers:
            X_tr = np.stack([hid[f"{p}|raw|{layer}"] for p in train["problem_id"]]).astype(np.float64)
            X_te = np.stack([hid[f"{p}|raw|{layer}"] for p in oly_ids]).astype(np.float64)
            X_pool = np.stack([pool[k] for k in pool.files if k.endswith(f"|raw|{layer}")]).astype(np.float64)
            sc_math = StandardScaler().fit(X_tr)
            clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(sc_math.transform(X_tr), y_tr)
            scores = {
                "math": clf.decision_function(sc_math.transform(X_te)),
                "pool": clf.decision_function(StandardScaler().fit(X_pool).transform(X_te)),
            }
            line = f"{enc.split('/')[-1]:28s} {layer:9s}"
            for name, s in scores.items():
                per_problem = pd.Series(s, index=oly_ids)
                s_cases = per_problem.loc[lab["problem_id"]].to_numpy()
                auc_all = roc_auc_score(y_all, s_cases)
                own = lab[lab["model_id"].isin(["qwen3-8b:low", "gpt-oss-120b:low"])]
                auc_open = roc_auc_score(own["robust"].astype(int), per_problem.loc[own["problem_id"]].to_numpy())
                line += (
                    f" | {name}: all{len(lab)}={auc_all:.2f} open{len(own)}={auc_open:.2f}"
                    f" frac_pos={float((s > 0).mean()):.1f}"
                )
            print(line)


if __name__ == "__main__":
    main()
