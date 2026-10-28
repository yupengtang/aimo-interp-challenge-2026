"""How much can the public evaluation sets actually tell us apart?

The olympiad check set has ten distinct problems, and every problem-level score repeats across that
problem's model rows, so the effective sample size is ten, not seventy. This resamples problems with
replacement (a cluster bootstrap) to put a confidence interval on the AUCs we have been comparing,
and reports how often a score of a given true quality would look better than another by chance.

Run this before reading any more meaning into differences on that set.
"""

from __future__ import annotations

import os
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def safe(name: str) -> str:
    return name.replace("/", "__")


def cluster_bootstrap_auc(scores: pd.Series, lab: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    problems = sorted(scores.index)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(problems, size=len(problems), replace=True)
        rows = pd.concat([lab[lab["problem_id"] == p] for p in pick])
        y = rows["robust"].astype(int).to_numpy()
        if len(set(y)) < 2:
            continue
        out.append(roc_auc_score(y, scores.loc[rows["problem_id"]].to_numpy()))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    oly = src[src["dataset_id"] == "aimo-2025-reference"]
    lab = (
        oly.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0)
        .rename("robust").reset_index()
    )
    lab = lab[lab["model_id"] != "gpt-5.2-2025-12-11"]
    oly_ids = sorted(lab["problem_id"].unique())
    print(f"olympiad check set: {len(lab)} cases over {len(oly_ids)} distinct problems, {int(lab['robust'].sum())} robust")

    # A few of the scores we have been comparing, all problem-level.
    candidates: dict[str, pd.Series] = {}
    for ck in ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        st = pd.read_csv(f"{args.base}/results/stability/stability_{safe(ck)}.csv").drop_duplicates("id").set_index("id")
        for col in st.columns:
            if col.startswith("cos_") and col.endswith(("_distract_mean", "_typos_mean")):
                candidates[f"{ck.split('/')[-1]}:{col}"] = st.loc[oly_ids, col].astype(float)
    chars = oly.drop_duplicates("problem_id").set_index("problem_id")["original_problem"].str.len().astype(float)
    candidates["problem_length"] = chars.loc[oly_ids]

    y_all = lab["robust"].astype(int).to_numpy()
    rows = []
    for name, s in candidates.items():
        point = roc_auc_score(y_all, s.loc[lab["problem_id"]].to_numpy())
        boot = cluster_bootstrap_auc(s, lab, args.n_boot, rng)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"score": name, "auc": round(point, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "width": round(hi - lo, 3), "p_above_0.5": round(float((boot > 0.5).mean()), 3)})
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(df.to_string(index=False))
    print(f"\nmedian 95% CI width across candidates: {df['width'].median():.3f}")

    # What would a genuinely useful method look like here? Simulate a score of known quality.
    print("\nsimulation: how often does a truly-0.50 score beat a truly-0.70 score on this set?")
    wins = 0
    trials = 1000
    for _ in range(trials):
        strength = rng.normal(0, 1, len(oly_ids))
        rate = lab.groupby("problem_id")["robust"].mean().loc[oly_ids].to_numpy()
        good = pd.Series(rate + rng.normal(0, 0.55, len(oly_ids)), index=oly_ids)   # informative
        null = pd.Series(strength, index=oly_ids)                                    # uninformative
        a_good = roc_auc_score(y_all, good.loc[lab["problem_id"]].to_numpy())
        a_null = roc_auc_score(y_all, null.loc[lab["problem_id"]].to_numpy())
        wins += a_null > a_good
    print(f"  an uninformative score outranks an informative one in {wins / trials:.1%} of draws")


if __name__ == "__main__":
    main()
