"""Does the evaluated model's own uncertainty over a short greedy trace predict the train labels?"""
import json, sys, glob, os
import numpy as np, pandas as pd
from collections import Counter
from nosol_local import load, lab, auc

d = sys.argv[1] if len(sys.argv) > 1 else "ownunc"
CFG = {"Qwen3.5-4B": ("Qwen/Qwen3.5-4B", "default"), "Skywork-OR1-Math-7B": ("Skywork/Skywork-OR1-Math-7B", "default"),
       "Olmo-3-7B-Think": ("allenai/Olmo-3-7B-Think", "default"), "DeepSeek-R1-0528-Qwen3-8B": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"),
       "gpt-oss-120b_low": ("openai/gpt-oss-120b", "low"), "gpt-oss-120b_medium": ("openai/gpt-oss-120b", "medium"), "gpt-oss-120b_high": ("openai/gpt-oss-120b", "high")}
rows = []
for f in sorted(glob.glob(f"{d}/trace_*.jsonl")):
    tag = os.path.basename(f)[6:-6]
    mid, eff = CFG[tag]
    recs = [json.loads(l) for l in open(f)]
    print(f"{tag:28s} traces {len(recs):2d} finished {np.mean([r['finished'] for r in recs]):.2f} mean tokens {np.mean([r['tokens'] for r in recs]):.0f}")
    for r in recs:
        e, t, m = np.array(r["entropy"]), np.array(r["top1"]), np.array(r["margin"])
        if len(e) == 0: continue
        rows.append(dict(model_id=mid, reasoning_effort=eff, pid=r["pid"], tokens=r["tokens"], finished=float(r["finished"]),
                         ent_mean=e.mean(), ent_first256=e[:256].mean(), ent_first512=e[:512].mean(), ent_last256=e[-256:].mean(),
                         ent_p90=np.percentile(e, 90), ent_max=e.max(), ent_std=e.std(), frac_ent_gt1=(e > 1.0).mean(),
                         top1_mean=t.mean(), top1_min_run=np.convolve(t, np.ones(16) / 16, "valid").min() if len(t) >= 16 else t.min(),
                         margin_mean=m.mean(), frac_top1_lt_half=(t < 0.5).mean()))
T = pd.DataFrame(rows)
m = lab.merge(T, on=["model_id", "reasoning_effort", "pid"], how="inner")
print(f"\nlabelled rows with the evaluated model's own trace: {len(m)} of {len(lab)}; robust {int(m.y.sum())}")
feats = [c for c in T.columns if c not in ("model_id", "reasoning_effort", "pid")]
z = m.copy()
for c in feats:
    g = z.groupby(["model_id", "reasoning_effort"])[c]
    z[c] = (z[c] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)
rng = np.random.default_rng(0); groups = [g for _, g in z.groupby("template")]
def ci(col, sign):
    bs = []
    for _ in range(1500):
        b = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]); a = auc(b.y, sign * b[col])
        if np.isfinite(a): bs.append(a)
    return np.percentile(bs, [2.5, 97.5])
print("\nfeature AUC, z-scored within configuration (sign chosen so that > 0.5 means: lower uncertainty / shorter = robust)")
res = []
for c in feats:
    sign = 1 if c in ("top1_mean", "top1_min_run", "margin_mean", "finished") else -1
    a = auc(z.y, sign * z[c]); lo, hi = ci(c, sign); res.append((a, c, lo, hi))
for a, c, lo, hi in sorted(res, reverse=True): print(f"  {c:18s} {a:.3f} [{lo:.2f}, {hi:.2f}]")
print("\nper configuration (n, AUC of -ent_mean, -tokens, finished):")
for (mid, eff), g in m.groupby(["model_id", "reasoning_effort"]):
    print(f"  {mid.split('/')[1][:24]:24s} {eff:8s} n={len(g):2d} robust={int(g.y.sum()):2d}  ent_mean {auc(g.y, -g.ent_mean):.2f}  tokens {auc(g.y, -g.tokens):.2f}  finished {auc(g.y, g.finished):.2f}")
# against and with the judge's modal share
runs = {"A": load("nosol/solves.jsonl"), "B": load("nosol2/solves.jsonl"), "C": load("nosol3/solves.jsonl")}
S = pd.concat([x.assign(run=k) for k, x in runs.items()], ignore_index=True); o = S[S.kind == "orig"]
share = o.groupby("pid").ans.apply(lambda a: Counter(x for x in a if x is not None).most_common(1)[0][1] / len(a))
z["share"] = z.pid.map(share)
from scipy.stats import rankdata, spearmanr
best = sorted(res, reverse=True)[0][1]; sign = 1 if best in ("top1_mean", "top1_min_run", "margin_mean", "finished") else -1
print(f"\njudge share on these rows: AUC {auc(z.y, z.share):.3f}; best own-model feature {best}: {sorted(res, reverse=True)[0][0]:.3f}; spearman {spearmanr(z.share, sign * z[best]).correlation:.2f}")
print(f"rank sum of the two: AUC {auc(z.y, rankdata(z.share) + rankdata(sign * z[best])):.3f}")
