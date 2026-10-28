"""Can we recover ground-truth answers for the officially labelled MATH problems?

Accuracy decay needs a correct answer to compare against. The official parquets carry the problem text
but no answer, so this matches those texts against the hendrycks MATH copy already downloaded, and
extracts the final boxed answer. The match rate decides whether self-labelling is feasible at all.
"""
import os
import glob
import json
import re

import pandas as pd

B = os.environ.get("AIMO_BASE", "/path/to/aimo")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def boxed(text: str):
    i = text.rfind("\boxed{")
    if i < 0:
        return None
    depth, out = 1, []
    for ch in text[i + len("\boxed{"):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out) if depth == 0 else None


src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{B}/data/*.parquet")])
labelled = src[src["dataset_id"].astype(str).str.startswith("hendrycks-math")].drop_duplicates("problem_id")
print(f"officially labelled MATH problems: {len(labelled)}")

rows = []
for path in glob.glob(f"{B}/unlabeled/hendrycks_math/**/*.parquet", recursive=True):
    d = pd.read_parquet(path)
    cols = {c.lower(): c for c in d.columns}
    p = next((cols[c] for c in ("problem", "question") if c in cols), None)
    s = next((cols[c] for c in ("solution", "answer") if c in cols), None)
    if p and s:
        rows.append(d[[p, s]].rename(columns={p: "problem", s: "solution"}))
pool = pd.concat(rows, ignore_index=True).drop_duplicates("problem")
print(f"hendrycks MATH pool: {len(pool)} problems from {len(rows)} files")

pool["key"] = pool["problem"].map(norm)
lookup = dict(zip(pool["key"], pool["solution"]))
hits, answers = 0, []
for _, r in labelled.iterrows():
    sol = lookup.get(norm(r["original_problem"]))
    if sol is None:
        continue
    ans = boxed(str(sol))
    if ans is not None:
        hits += 1
        answers.append((r["problem_id"], ans[:40]))
print(f"matched with an extractable boxed answer: {hits} of {len(labelled)} ({hits / max(1, len(labelled)):.0%})")
print("examples:", answers[:8])
