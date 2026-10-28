"""Do answer-format self-labels agree with the organizers' labels?

The pilot generates, for each problem, samples on the untouched statement and on each answer-format
variant, every sample scored against an exact target. This turns those samples into the same shape of
label the organizers produce (does accuracy hold up under perturbation) and asks the only question
that matters before spending compute on olympiad problems: does my label agree with theirs?

Unscorable samples are dropped, never counted as wrong. A truncated or answerless generation says
something about difficulty, not robustness, and treating it as an error is what turned the earlier
agreement experiment into a difficulty measure.
"""

from __future__ import annotations

import os
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--results", default=None)
    ap.add_argument("--decay-threshold", type=float, default=0.5,
                    help="a family counts as detrimental when accuracy falls by at least this much")
    args = ap.parse_args()
    results = Path(args.results or f"{args.base}/results/selflabel_pilot")

    samples = pd.read_json(results / "samples.jsonl", lines=True)
    samples["problem_id"] = samples["problem_id"].astype(str)
    total = len(samples)
    scorable = samples[samples["correct"].notna()].copy()
    print(f"samples: {total} generated, {len(scorable)} scorable "
          f"({len(scorable) / max(1, total):.0%}); unscorable are dropped, not counted wrong")
    print(f"  truncated: {samples['hit_limit'].mean():.0%} | no answer extracted: "
          f"{samples['answer'].isna().mean():.0%}")

    rows = []
    for pid, g in scorable.groupby("problem_id"):
        base = g[g["variant"] == "orig"]["correct"]
        if len(base) == 0:
            continue
        base_acc = float(base.mean())
        entry = {"problem_id": pid, "base_acc": base_acc, "n_base": int(len(base))}
        detrimental = 0
        families = 0
        for family, gf in g[g["variant"] != "orig"].groupby("variant"):
            acc = float(gf["correct"].mean())
            entry[f"acc_{family}"] = acc
            entry[f"n_{family}"] = int(len(gf))
            families += 1
            if base_acc - acc >= args.decay_threshold:
                detrimental += 1
        entry["n_families"] = families
        entry["n_detrimental"] = detrimental
        entry["self_robust"] = families > 0 and detrimental == 0
        entry["max_decay"] = max(
            (base_acc - entry[f"acc_{f}"] for f in ("scaled", "digit_sum", "modulo")
             if f"acc_{f}" in entry), default=np.nan)
        rows.append(entry)
    self_labels = pd.DataFrame(rows)
    print(f"\nproblems with a self-label: {len(self_labels)}")
    if self_labels.empty:
        return
    print(f"  base accuracy on the untouched problem: mean {self_labels['base_acc'].mean():.2f}")
    for family in ("scaled", "digit_sum", "modulo"):
        col = f"acc_{family}"
        if col in self_labels:
            sub = self_labels[self_labels[col].notna()]
            print(f"  {family:9s} n={len(sub):3d} accuracy {sub[col].mean():.2f} "
                  f"decay {(sub['base_acc'] - sub[col]).mean():+.2f}")
    print(f"  self-labelled robust: {int(self_labels['self_robust'].sum())} of {len(self_labels)}")

    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    official = (
        src[src["model_id"].isin(["qwen3-8b:low", LABEL_CK])]
        .groupby("problem_id")["n_detrimental_permutations"].sum().eq(0).rename("official_robust")
    )
    merged = self_labels.join(official, on="problem_id", how="inner")
    merged = merged[merged["official_robust"].notna()]
    print(f"\noverlap with official labels: {len(merged)} problems, "
          f"official robust {int(merged['official_robust'].sum())}, "
          f"self robust {int(merged['self_robust'].sum())}")
    if len(merged) < 10 or merged["official_robust"].nunique() < 2:
        print("  too few or single-class; no agreement statistic is meaningful yet")
        return

    y = merged["official_robust"].astype(int).to_numpy()
    agree = float((merged["self_robust"].astype(bool) == merged["official_robust"].astype(bool)).mean())
    base_rate = max(y.mean(), 1 - y.mean())
    print(f"  agreement: {agree:.2f} against a majority-class baseline of {base_rate:.2f}")
    for col in ("max_decay", "base_acc"):
        if merged[col].notna().any() and merged[col].nunique() > 1:
            m = merged[col].notna()
            print(f"  AUC of {col:9s} for the official label: "
                  f"{roc_auc_score(y[m.to_numpy()], merged.loc[m, col]):.3f}")
    self_labels.to_csv(results / "self_labels.csv", index=False)
    print(f"\nwrote {results / 'self_labels.csv'}")


if __name__ == "__main__":
    main()
