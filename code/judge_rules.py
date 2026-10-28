"""Score the low-effort judge against the official train-main-v2 labels.

Input: the output directories of nosol_experiment.py (one or more runs). Together they hold, per
problem, several solves of the original with the organizers' prompt and several no-solution variants
with their solves. This script reports

1. the AUC of the modal-answer share over all original samples, and of the NaN rate on variants per
   run, with a bootstrap interval over problem templates;
2. the accuracy of threshold rules when only k of the original samples are drawn, repeated many
   times, including an estimate in which the threshold is chosen with the scored problem's template
   held out;
3. the submitted rule, and what adding the variant signal changes.
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

BOXED = chr(92) + "boxed{"


def auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def answer_of(row: dict) -> str | None:
    """The stored answer, else the last boxed expression in the stored tail of the trace."""
    if row["answer"] is not None:
        return row["answer"]
    tail = row.get("tail") or ""
    i = tail.rfind(BOXED)
    if i < 0:
        return None
    depth, out = 1, []
    for ch in tail[i + len(BOXED):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    if depth != 0:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", "".join(out)).replace("\\,", "").replace(" ", "").strip("$.")
    if s.lower() in ("nan", "nosolution", "none", "undefined", "impossible"):
        return "NaN"
    try:
        v = float(s.replace(",", ""))
        return str(int(v)) if v.is_integer() else str(v)
    except ValueError:
        return s


def load_labels(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df["pid"] = df.problem.map(lambda p: hashlib.sha1(p.encode()).hexdigest()[:12])
    df["template"] = df.problem.map(lambda p: re.sub(r"[0-9]+", "#", p)[:60])
    lab = df[df.is_robust.notna()].copy()
    lab["y"] = lab.is_robust.astype(int)
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--cap", type=int, default=2560)
    ap.add_argument("--share", type=float, default=0.8)
    ap.add_argument("--draws", type=int, default=150)
    args = ap.parse_args()

    lab = load_labels(args.parquet)
    y = lab.y.values.astype(bool)
    templates = lab.template.values
    rng = np.random.default_rng(0)
    groups = [g for _, g in lab.groupby("template")]

    orig: dict[str, list[tuple[str | None, int]]] = {}
    variants: dict[str, dict[str, list[str | None]]] = {}
    nan_by_run = []
    for k, run in enumerate(args.runs):
        rows = [json.loads(line) for line in (run / "solves.jsonl").open()]
        per_run: dict[str, list[bool]] = {}
        for r in rows:
            a = answer_of(r)
            if r["kind"] == "orig":
                orig.setdefault(r["pid"], []).append((a, r["tokens"]))
            else:
                variants.setdefault(r["pid"], {}).setdefault(f"{k}:{r['kind']}", []).append(a)
                per_run.setdefault(r["pid"], []).append(a == "NaN")
        nan_by_run.append({p: float(np.mean(v)) for p, v in per_run.items()})
    n_orig = min(len(v) for v in orig.values())
    print(f"{len(lab)} labelled rows, {lab.pid.nunique()} problems, {lab.template.nunique()} templates; "
          f"{n_orig} original samples and up to {max(len(v) for v in variants.values())} variants per problem")

    def boot(values: dict) -> tuple[float, float, float]:
        bs = []
        for _ in range(2000):
            b = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a = auc(b.y, b.pid.map(values).fillna(0))
            if np.isfinite(a):
                bs.append(a)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return auc(lab.y, lab.pid.map(values).fillna(0)), lo, hi

    def share_of(samples, cap) -> float:
        c = Counter(a for a, t in samples if a is not None and t <= cap)
        return c.most_common(1)[0][1] / len(samples) if c else 0.0

    full = {p: share_of(v, 10**9) for p, v in orig.items()}
    print("\nAUC [95% template bootstrap]")
    print("  modal share, all original samples   %.2f [%.2f, %.2f]" % boot(full))
    for k, values in enumerate(nan_by_run):
        print(f"  NaN rate on variants, run {k + 1}         %.2f [%.2f, %.2f]" % boot(values))

    def balanced(pred) -> float:
        return 0.5 * (pred[y].mean() + (~pred[~y]).mean())

    def draw(k: int, cap: int, n_variants: int):
        share, nan = {}, {}
        for p in lab.pid.unique():
            pick = rng.choice(len(orig[p]), k, replace=False)
            share[p] = share_of([orig[p][i] for i in pick], cap)
            if n_variants:
                keys = rng.choice(list(variants[p]), min(n_variants, len(variants[p])), replace=False)
                nan[p] = float(np.mean([a == "NaN" for key in keys for a in variants[p][key]]))
        f = lambda m: lab.pid.map(m).values  # noqa: E731
        return f(share), (f(nan) if n_variants else None)

    thresholds = (0.5, 0.6, 0.7, 0.75, 0.8, 0.875, 0.9, 1.0)
    print("\nrules on the modal share alone, by number of original samples")
    for k in (8, 16, 24):
        if k > n_orig:
            continue
        shares = np.array([draw(k, 10**9, 0)[0] for _ in range(args.draws)])
        preds = {t: shares >= t - 1e-9 for t in thresholds}
        best = max(thresholds, key=lambda t: np.mean([balanced(preds[t][i]) for i in range(args.draws)]))
        b = [balanced(preds[best][i]) for i in range(args.draws)]
        hit = np.zeros_like(shares, dtype=bool)
        for tm in np.unique(templates):
            te = templates == tm
            yy = y[~te]
            pick = max(thresholds, key=lambda t: np.mean(
                [0.5 * (preds[t][i][~te][yy].mean() + (~preds[t][i][~te][~yy]).mean()) for i in range(0, args.draws, 3)]))
            hit[:, te] = preds[pick][:, te] == y[te]
        held = [0.5 * (hit[i][y].mean() + hit[i][~y].mean()) for i in range(args.draws)]
        print(f"  {k:2d} samples: best threshold {best}: balanced {np.mean(b):.3f} +- {np.std(b):.3f} | "
              f"threshold chosen with the template held out: accuracy {hit.mean():.3f}, balanced {np.mean(held):.3f}")

    k = min(args.samples, n_orig)
    acc, bal, rate = [], [], []
    for _ in range(args.draws):
        share, _ = draw(k, args.cap, 0)
        pred = share >= args.share - 1e-9
        acc.append((pred == y).mean())
        bal.append(balanced(pred))
        rate.append(pred.mean())
    print(f"\nsubmitted rule ({k} samples, cap {args.cap}, share >= {args.share}): accuracy {np.mean(acc):.3f}, "
          f"balanced {np.mean(bal):.3f} +- {np.std(bal):.3f}, predicted robust {np.mean(rate):.2f} "
          f"(always robust: accuracy {y.mean():.3f})")

    print("\nadding the variant signal at the same sample count (best of a small grid, in sample)")
    for n_variants in (0, 4):
        draws = [draw(k, args.cap, n_variants) for _ in range(args.draws)]
        best = 0.0
        for w in ((0.0,) if not n_variants else (0.25, 0.5, 1.0)):
            for t in (0.7, 0.75, 0.8, 0.875, 0.9):
                b = np.mean([balanced((s + w * (n if n is not None else 0)) >= t - 1e-9) for s, n in draws])
                best = max(best, b)
        print(f"  {n_variants} variants: balanced {best:.3f}")


if __name__ == "__main__":
    main()
