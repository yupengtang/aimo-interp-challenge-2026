"""How much would olympiad-domain labels be worth?

Compares three training sets on the same held-out olympiad problem, cycling over all ten:
  * MATH only (154 DeepSeek cases) - what we can build today;
  * MATH + the nine other olympiad problems;
  * the nine other olympiad problems only.

Features are problem-level: the last-token hidden state of the raw problem from one encoder. Olympiad
targets are the fraction of the official ":low" model runs that are robust on that problem, so a
single score is scored against all of that problem's per-model labels. Ten problems is far too few to
settle anything, but a consistent gap from adding in-domain labels is the argument for asking the
organizers whether we may build more.
"""

from __future__ import annotations

import os
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LABEL_CK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def safe(name: str) -> str:
    return name.replace("/", "__")


def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, w: np.ndarray, X_te: np.ndarray, alpha: float) -> np.ndarray:
    sc = StandardScaler().fit(X_tr)
    model = Ridge(alpha=alpha).fit(sc.transform(X_tr), y_tr, sample_weight=w)
    return model.predict(sc.transform(X_te))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--encoders", nargs="+", default=["Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[10.0, 100.0, 1000.0])
    args = ap.parse_args()

    feat_dir = f"{args.base}/results/cheap_features_h200"
    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    oly = src[src["dataset_id"] == "aimo-2025-reference"]
    lab = (
        oly.groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0)
        .rename("robust").reset_index()
    )
    lab = lab[lab["model_id"] != "gpt-5.2-2025-12-11"]
    oly_ids = sorted(lab["problem_id"].unique())
    rate = lab.groupby("problem_id")["robust"].mean()

    cases = pd.read_csv(f"{feat_dir}/labeled_cases.csv")
    math = cases[(cases["checkpoint"] == LABEL_CK) & ~cases["problem_id"].isin(oly_ids)]
    y_math = math["derived_label"].astype(float).to_numpy()

    for enc in args.encoders:
        hid = np.load(f"{feat_dir}/hidden_{safe(enc)}.npz")
        layers = sorted({k.split("|")[2] for k in hid.files})
        for layer in layers:
            X_math = np.stack([hid[f"{p}|raw|{layer}"] for p in math["problem_id"]]).astype(np.float64)
            X_oly = {p: hid[f"{p}|raw|{layer}"].astype(np.float64) for p in oly_ids}
            best = {}
            for alpha in args.alphas:
                preds = {"math": {}, "math+oly": {}, "oly": {}}
                for held in oly_ids:
                    others = [p for p in oly_ids if p != held]
                    X_o = np.stack([X_oly[p] for p in others])
                    y_o = rate.loc[others].to_numpy()
                    X_te = X_oly[held][None, :]
                    preds["math"][held] = fit_predict(X_math, y_math, np.ones(len(y_math)), X_te, alpha)[0]
                    # weight the nine in-domain problems so they are not swamped by 154 MATH rows
                    X_c = np.vstack([X_math, X_o])
                    y_c = np.concatenate([y_math, y_o])
                    w_c = np.concatenate([np.ones(len(y_math)), np.full(len(y_o), len(y_math) / len(y_o))])
                    preds["math+oly"][held] = fit_predict(X_c, y_c, w_c, X_te, alpha)[0]
                    preds["oly"][held] = fit_predict(X_o, y_o, np.ones(len(y_o)), X_te, alpha)[0]
                y_cases = lab["robust"].astype(int).to_numpy()
                row = {}
                for name, d in preds.items():
                    s = pd.Series(d).loc[lab["problem_id"]].to_numpy()
                    row[name] = round(roc_auc_score(y_cases, s), 3)
                best[alpha] = row
            line = f"{enc.split('/')[-1]:26s} {layer:9s}"
            for alpha, row in best.items():
                line += f" | a={alpha:g} math={row['math']:.2f} math+oly={row['math+oly']:.2f} oly={row['oly']:.2f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
