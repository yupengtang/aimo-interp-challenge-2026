"""Run the submission bundle exactly as the harness would, and time it.

The ingestion program imports solution.py once and then calls are_robust once per checkpoint, with a
single 3600 s limit covering every call. This does the same in one process, over all five checkpoints,
and checks three things:

  * the contract: a list of native bools, one per problem, order preserved;
  * the features: the runtime must produce every name the artifact expects, otherwise standardisation
    silently zeroes them and the score collapses to the intercept, which still "works" but is constant;
  * the budget: per-model and total wall clock against 3600 s, on the same class of GPU.

Official labels are joined at the end only as a sanity check. The olympiad set has ten distinct
problems, so nothing there is conclusive; see power_check.py.
"""

from __future__ import annotations

import os
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd

MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b",
]
ALIAS = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "gpt-oss-120b:low": "openai/gpt-oss-120b"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--n-problems", type=int, default=30, help="problems per model, as in the real test split")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                    help="restrict the run, e.g. to the four checkpoints that fit on a 48 GB partition")
    args = ap.parse_args()
    bundle = Path(args.bundle or f"{args.base}/bundle")
    sys.path.insert(0, str(bundle))

    import joblib

    artifact = joblib.load(bundle / "stability_crossmodel.joblib")
    expected_features = set(artifact["feature_names"])

    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    oly = src[src["dataset_id"] == "aimo-2025-reference"].drop_duplicates("problem_id")
    pool = [json.loads(l) for l in open(f"{args.base}/unlabeled/olympiad/problems.jsonl")]
    problems = oly["original_problem"].tolist()
    problems += [r["problem"] for r in pool[: max(0, args.n_problems - len(problems))]]
    problems = problems[: args.n_problems]
    print(f"{len(problems)} problems per model, {len(args.models)} models: "
          f"{[m.split('/')[-1] for m in args.models]}", flush=True)

    # Feature-name coverage, checked against the first model before the full run.
    import solution  # noqa: E402  (importing starts the budget clock, as in the harness)
    from scoring import problem_features, quartile_layers  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    probe_ck = args.models[0]
    tok = AutoTokenizer.from_pretrained(probe_ck, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(probe_ck, dtype="auto", device_map="auto", local_files_only=True).eval()
    feats = problem_features(
        mdl, tok, probe_ck, problems[0], "case_0", quartile_layers(mdl),
        mdl.get_input_embeddings().weight.device, ("typos", "distract", "wrap"),
    )
    missing = sorted(expected_features - set(feats))
    extra = sorted(set(feats) - expected_features)
    print(f"feature coverage: {len(feats)} produced, missing={missing}, unexpected={extra}")
    assert not missing, f"runtime does not produce {len(missing)} artifact features: {missing[:5]}"
    del mdl

    import torch

    torch.cuda.empty_cache()

    rows, predictions, t_run = [], {}, time.monotonic()
    for ck in args.models:
        t0 = time.monotonic()
        verdicts = solution.are_robust(ck, problems)
        dt = time.monotonic() - t0
        ok_type = type(verdicts) is list and all(type(v) is bool for v in verdicts)
        predictions[ck] = verdicts
        rows.append({
            "model": ck, "n": len(verdicts), "valid_contract": ok_type and len(verdicts) == len(problems),
            "robust_rate": round(sum(verdicts) / max(1, len(verdicts)), 2),
            "wall_s": round(dt, 1), "s_per_problem": round(dt / max(1, len(problems)), 2),
        })
        print(json.dumps(rows[-1]), flush=True)
    total = time.monotonic() - t_run
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\ntotal wall {total:.0f}s of the 3600 s limit, headroom {3600 - total:.0f}s")
    print("all verdicts constant per model:", [r["robust_rate"] in (0.0, 1.0) for r in rows])

    lab = (
        src[src["dataset_id"] == "aimo-2025-reference"]
        .groupby(["model_id", "problem_id"])["n_detrimental_permutations"].sum().eq(0).rename("robust").reset_index()
    )
    lab["checkpoint"] = lab["model_id"].map(ALIAS)
    lab = lab[lab["checkpoint"].notna()]
    index_of = {pid: i for i, pid in enumerate(oly["problem_id"].tolist()) if i < len(problems)}
    for ck, d in lab.groupby("checkpoint"):
        verdicts = predictions.get(ck)
        if verdicts is None:
            continue
        pairs = [(bool(verdicts[index_of[p]]), bool(t)) for p, t in zip(d["problem_id"], d["robust"]) if p in index_of]
        if not pairs:
            continue
        hits = sum(p == t for p, t in pairs)
        positives = sum(t for _, t in pairs)
        best_constant = max(positives, len(pairs) - positives) / len(pairs)
        print(f"{ck}: accuracy {hits / len(pairs):.2f} on {len(pairs)} labelled olympiad cases, "
              f"best constant {best_constant:.2f}. Ten distinct problems, so this is a sanity check, not evidence.")


if __name__ == "__main__":
    main()
