"""Does the batched feature path agree with the unbatched one, and is it actually faster?

Two questions, both raised by the Blackwell timing run:

  * Parity. The artifact was fitted on features computed one sequence at a time, with no padding. The
    batched path pads on the left, and padding can move hidden states. If the two disagree, the
    shipped scorer is being applied to features it was not trained on, which would be a silent
    accuracy bug rather than a visible failure.
  * Speed. Batching cut Qwen3.5 from 35.6 s to 7.6 s but left the 7 to 8 B models unchanged, which is
    not what batching does. The suspicion is that batch_features raises for those models and
    solution.py quietly falls back per problem, or that the OOM retry collapses the batch to one.

Exceptions are deliberately not caught here, so a failure shows itself.
"""

from __future__ import annotations

import os
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("AIMO_BASE", "/path/to/aimo"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-problems", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    bundle = Path(args.base) / "bundle"
    sys.path.insert(0, str(bundle))

    import joblib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scoring import (batch_features, chat_prompt, problem_features, quartile_layers,
                         score_features, threshold_for)

    artifact = joblib.load(bundle / "stability_crossmodel.joblib")
    src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{args.base}/data/*.parquet")])
    problems = (
        src[src["dataset_id"] == "aimo-2025-reference"].drop_duplicates("problem_id")
        ["original_problem"].tolist()[: args.n_problems]
    )
    ids = [f"case_{i}" for i in range(len(problems))]
    families = ("typos", "distract", "wrap")

    for ck in args.models:
        print(f"\n===== {ck}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(ck, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            ck, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        device = model.get_input_embeddings().weight.device
        layers = quartile_layers(model)
        prompt_lengths = [
            len(tokenizer(chat_prompt(tokenizer, ck, p), add_special_tokens=False)["input_ids"])
            for p in problems
        ]
        print(f"layers {layers} | prompt tokens: min {min(prompt_lengths)} median "
              f"{int(np.median(prompt_lengths))} max {max(prompt_lengths)}")

        torch.cuda.synchronize()
        t0 = time.monotonic()
        single = [problem_features(model, tokenizer, ck, p, ids[i], layers, device, families)
                  for i, p in enumerate(problems)]
        torch.cuda.synchronize()
        t_single = time.monotonic() - t0

        t0 = time.monotonic()
        batched = batch_features(model, tokenizer, ck, problems, layers, device, families,
                                 batch_size=args.batch_size, ids=ids)
        torch.cuda.synchronize()
        t_batched = time.monotonic() - t0

        print(f"unbatched {t_single:.1f}s ({t_single / len(problems):.2f}s per problem) | "
              f"batched {t_batched:.1f}s ({t_batched / len(problems):.2f}s per problem) | "
              f"speedup {t_single / max(t_batched, 1e-9):.2f}x")

        names = artifact["feature_names"]
        diffs = {n: max(abs(s.get(n, np.nan) - b.get(n, np.nan)) for s, b in zip(single, batched))
                 for n in names}
        worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
        print("largest per-feature absolute differences:")
        for name, d in worst:
            print(f"   {name:26s} {d:.6f}")

        thr = threshold_for(artifact, ck)
        s_scores = np.array([score_features(f, artifact, ck) for f in single])
        b_scores = np.array([score_features(f, artifact, ck) for f in batched])
        flips = int(((s_scores > thr) != (b_scores > thr)).sum())
        print(f"score max abs diff {np.max(np.abs(s_scores - b_scores)):.6f} | "
              f"verdict flips {flips} of {len(problems)} | threshold {thr:.4f}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
