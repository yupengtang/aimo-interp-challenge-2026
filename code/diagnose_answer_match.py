"""Why did none of the officially labelled MATH problems match the hendrycks copy?

Tries progressively looser keys and, for the first failure, shows the closest candidate so the exact
difference (escaping, truncation, reformatting) is visible rather than guessed at.
"""
import os
import difflib
import glob
import re

import pandas as pd

B = os.environ.get("AIMO_BASE", "/path/to/aimo")

src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{B}/data/*.parquet")])
labelled = src[src["dataset_id"].astype(str).str.startswith("hendrycks-math")].drop_duplicates("problem_id")

frames = []
for path in glob.glob(f"{B}/unlabeled/hendrycks_math/**/*.parquet", recursive=True):
    d = pd.read_parquet(path)
    cols = {c.lower(): c for c in d.columns}
    p = next((cols[c] for c in ("problem", "question") if c in cols), None)
    s = next((cols[c] for c in ("solution", "answer") if c in cols), None)
    if p and s:
        frames.append(d[[p, s]].rename(columns={p: "problem", s: "solution"}))
pool = pd.concat(frames, ignore_index=True).drop_duplicates("problem").reset_index(drop=True)

def ws(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()

def alnum(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())

for name, fn in (("whitespace+lower", ws), ("alphanumeric only", alnum)):
    keys = {fn(p): i for i, p in enumerate(pool["problem"])}
    hits = sum(1 for t in labelled["original_problem"] if fn(t) in keys)
    print(f"{name:20s} matched {hits} of {len(labelled)}")

print("\nfirst official problem, verbatim:")
first = str(labelled["original_problem"].iloc[0])
print(repr(first[:300]))

pool_alnum = [alnum(p) for p in pool["problem"]]
target = alnum(first)
best = difflib.get_close_matches(target, pool_alnum, n=1, cutoff=0.5)
if best:
    idx = pool_alnum.index(best[0])
    print(f"\nclosest pool problem (ratio {difflib.SequenceMatcher(None, target, best[0]).ratio():.3f}):")
    print(repr(str(pool["problem"].iloc[idx])[:300]))
    print("\nits solution tail:", repr(str(pool["solution"].iloc[idx])[-160:]))
else:
    print("\nno close candidate above ratio 0.5; checking substring containment")
    frag = target[:60]
    contained = [i for i, p in enumerate(pool_alnum) if frag and frag in p]
    print("pool problems containing the first 60 alphanumeric chars:", len(contained))
    if contained:
        print(repr(str(pool["problem"].iloc[contained[0]])[:300]))

print("\nofficial text lengths:", labelled["original_problem"].str.len().describe().round(0).to_dict())
print("pool text lengths    :", pool["problem"].str.len().describe().round(0).to_dict())
