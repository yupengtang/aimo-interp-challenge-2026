"""How well does the low-effort judge predict the official train-main-v2 labels?

Reads the samples written by trainv2_generate.py (several per problem) and reports

1. single-signal AUCs over all samples, with a bootstrap interval over problem templates, and
2. the accuracy of a verdict rule when only k of the available samples are drawn, repeated many
   times, so that a rule is judged by its average and its spread and not by one draw.

The second part exists because the first four samples I drew made "all four agree" look like 0.87
balanced accuracy. Resampled, the same rule averages 0.68.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def load_labels(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["pid"] = df.problem.map(lambda p: hashlib.sha1(p.encode()).hexdigest()[:12])
    # Number-variants of one template share their opening once digits are masked.
    df["template"] = df.problem.map(lambda p: re.sub(r"[0-9]+", "#", p)[:60])
    lab = df[df.is_robust.notna()].copy()
    lab["y"] = lab.is_robust.astype(int)
    return lab


def load_samples(path: Path) -> dict[str, list[dict]]:
    per: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        per.setdefault(r["pid"], []).append(r)
    return per


def summarise(samples: list[dict], cap: int) -> tuple[int, int, float]:
    """(count of the modal answer, distinct answers, median tokens) with traces cut at `cap`."""
    answers = [
        r["answer"] if (r["answer"] is not None and not r["hit_limit"] and r["n_new_tokens"] <= cap) else None
        for r in samples
    ]
    counts = Counter(a for a in answers if a is not None)
    top = counts.most_common(1)[0][1] if counts else 0
    return top, len(counts), float(np.median([min(r["n_new_tokens"], cap) for r in samples]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cap", type=int, default=3072)
    ap.add_argument("--max-distinct", type=int, default=2)
    ap.add_argument("--max-median", type=float, default=2200)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = load_labels(args.parquet)
    per = load_samples(args.samples)
    y = lab.y.values.astype(bool)
    rng = np.random.default_rng(args.seed)
    print(f"{len(lab)} labelled rows, {lab.pid.nunique()} problems, {lab.template.nunique()} templates, "
          f"{min(len(v) for v in per.values())} samples per problem")

    full = {p: summarise(rs, 10**9) for p, rs in per.items()}
    text = lab.drop_duplicates("pid").set_index("pid").problem
    signals = {
        "fewer distinct answers": {p: -v[1] for p, v in full.items()},
        "modal answer share": {p: v[0] / len(per[p]) for p, v in full.items()},
        "shorter traces": {p: -float(np.mean([r["n_new_tokens"] for r in per[p]])) for p in per},
        "fewer $ in the statement": (-text.str.count(r"\$")).to_dict(),
    }
    groups = [g for _, g in lab.groupby("template")]
    print("\nsignal AUC [95% template bootstrap]")
    for name, values in signals.items():
        score = lab.pid.map(values)
        boots = []
        for _ in range(2000):
            b = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a = auc(b.y, b.pid.map(values))
            if np.isfinite(a):
                boots.append(a)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {name:26s} {auc(lab.y, score):.2f} [{lo:.2f}, {hi:.2f}]")

    rules = {
        f"distinct<={args.max_distinct} and median<{args.max_median:.0f}":
            lambda top, distinct, median, k: 1 <= distinct <= args.max_distinct and median < args.max_median,
        "all samples agree": lambda top, distinct, median, k: top == k,
    }
    print(f"\nrules with k={args.k} of the samples, cap {args.cap} tokens, {args.draws} draws")
    for name, rule in rules.items():
        acc, bal, rate, verdicts = [], [], [], []
        for _ in range(args.draws):
            v = {}
            for p, rs in per.items():
                idx = rng.choice(len(rs), size=args.k, replace=False)
                v[p] = bool(rule(*summarise([rs[i] for i in idx], args.cap), args.k))
            pred = lab.pid.map(v).values.astype(bool)
            acc.append((pred == y).mean())
            bal.append(0.5 * (pred[y].mean() + (~pred[~y]).mean()))
            rate.append(pred.mean())
            verdicts.append([v[p] for p in per])
        share = np.array(verdicts).mean(0)
        stable = float((share**2 + (1 - share) ** 2).mean())
        print(f"  {name:32s} accuracy {np.mean(acc):.3f}  balanced {np.mean(bal):.3f} +- {np.std(bal):.3f}  "
              f"predicted robust {np.mean(rate):.2f}  two draws agree {stable:.2f}")
    print(f"  always robust: accuracy {y.mean():.3f}, balanced 0.500")


if __name__ == "__main__":
    main()
