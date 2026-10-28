"""Test on train-main-v2: is a configuration robust when its OWN accuracy on the original problem is near 0 or 1?

Reference answer per problem = majority answer over all gpt-oss samples (low/medium/high pooled), since train
has no gold answers. Own accuracy of a configuration = share of its samples equal to the reference.
Rules compared on the labelled rows that have samples:
  extreme  : robust iff own accuracy <= lo or >= hi
  high-acc : robust iff own accuracy >= hi (the v5-like direction: easy for this configuration)
"""
import collections
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(os.path.join(D, "..", "data", "train-main-v2.parquet") if not os.path.exists(os.path.join(D, "train-main-v2.parquet")) else os.path.join(D, "train-main-v2.parquet"))
d = d[d.is_robust.notna()].copy()
d["pid"] = [hashlib.sha1(p.encode()).hexdigest()[:12] for p in d.problem]

samples = {}
for f in glob.glob(os.path.join(D, "trainv2_gen", "samples_*.jsonl")):
    tag = os.path.basename(f)[len("samples_"):-len(".jsonl")]
    for line in open(f):
        r = json.loads(line)
        samples.setdefault(tag, collections.defaultdict(list))[r["pid"]].append(r["answer"])


def norm(a):
    if a is None:
        return None
    s = str(a).strip().replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s or None


ref = {}
for tag in [t for t in samples if t.startswith("gpt-oss")]:
    for pid, ans in samples[tag].items():
        ref.setdefault(pid, []).extend(norm(a) for a in ans)
ref = {pid: collections.Counter(a for a in v if a is not None).most_common(1)[0][0]
       for pid, v in ref.items() if any(a is not None for a in v)}


def tag_of(row):
    m = row.model_id.split("/")[-1]
    if "gpt-oss" in m:
        return f"gpt-oss-120b_{row.reasoning_effort}"
    for t in samples:
        if t.startswith(m) or m.startswith(t.split("_")[0]):
            return t
    return None


rows = []
for _, r in d.iterrows():
    t = tag_of(r)
    if t is None or r.pid not in samples.get(t, {}) or r.pid not in ref:
        continue
    ans = [norm(a) for a in samples[t][r.pid]]
    acc = np.mean([a == ref[r.pid] for a in ans])
    rows.append(dict(cfg=t, pid=r.pid, acc=acc, n=len(ans), robust=bool(r.is_robust), drop=r.max_drop))
g = pd.DataFrame(rows)
print(f"{len(g)} labelled rows with own samples; robust rate {g.robust.mean():.2f}")
print(g.groupby("cfg").agg(n=("robust", "size"), robust=("robust", "mean"), mean_acc=("acc", "mean")).round(2))
for lo, hi in [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.25, 0.75)]:
    p = (g.acc <= lo) | (g.acc >= hi)
    y = g.robust
    bal = 0.5 * ((p & y).sum() / max(y.sum(), 1) + (~p & ~y).sum() / max((~y).sum(), 1))
    print(f"extreme lo={lo} hi={hi}: acc {(p == y).mean():.3f} bal {bal:.3f} pred robust {p.mean():.2f}")
for hi in [0.5, 0.75, 1.0]:
    p = g.acc >= hi
    y = g.robust
    bal = 0.5 * ((p & y).sum() / max(y.sum(), 1) + (~p & ~y).sum() / max((~y).sum(), 1))
    print(f"high-acc >= {hi}: acc {(p == y).mean():.3f} bal {bal:.3f}")
print(g.assign(bucket=pd.cut(g.acc, [-0.01, 0.05, 0.5, 0.95, 1.0])).groupby("bucket", observed=True).robust.agg(["size", "mean"]).round(2))
