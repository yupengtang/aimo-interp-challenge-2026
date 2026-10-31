"""Figure for Section 5: robust share by a configuration's own accuracy on the original problem.

Public sample (aimo-interp-challenge-sample-full), competition labelling rule (robust if no
non-expert perturbation lowers accuracy by more than 0.10, non-robust if one lowers it by 0.25
or more, cases in between dropped), expert-written perturbation types excluded. 71 cases.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"

d = pd.read_parquet("sample_full.parquet")
d = d[~d.permutation_type.str.contains("expert")]
g = d.groupby(["model_id", "problem_id"]).agg(acc=("base_accuracy", "first"), drop=("absolute_accuracy_decay", "max"))
g["robust"] = np.where(g["drop"] <= 0.10, 1.0, np.where(g["drop"] >= 0.25, 0.0, np.nan))
g = g.dropna(subset=["robust"])
assert len(g) == 71

edges = [(0.0, 0.05, "below\n0.05"), (0.05, 0.75, "0.05 to\n0.75"), (0.75, 0.95, "0.75 to\n0.95"), (0.95, 1.01, "0.95\nor more")]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


labels, share, lo, hi, ns = [], [], [], [], []
for a, b, name in edges:
    s = g[(g.acc >= a) & (g.acc < b)]
    k, n = int(s.robust.sum()), len(s)
    l, h = wilson(k, n)
    labels.append(name); share.append(k / n); lo.append(k / n - l); hi.append(h - k / n); ns.append(n)
    print(f"{name!r}: n={n} robust={k / n:.3f} ci=[{l:.2f}, {h:.2f}]")

plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.3, 1.7))
x = np.arange(len(labels))
ax.bar(x, share, width=0.58, color=BLUE, zorder=2)
ax.errorbar(x, share, yerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=2.5, zorder=3)
for xi, s, h, n in zip(x, share, hi, ns):
    ax.text(xi, min(s + h + 0.04, 1.06), f"{s:.0%} (n={n})", ha="center", va="bottom", color=INK, fontsize=6.8)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
ax.set_ylabel("robust share", color=INK)
ax.set_xlabel("own accuracy on the original problem", color=INK)
ax.grid(axis="y", color="#e4e3df", linewidth=0.5, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("fig_ushape.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("fig_ushape.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
