"""Score the shipped stability bundle on the official train-main-v2 problems.

Writes the raw score, the shipped threshold and the features for every checkpoint and problem, so
the in-domain accuracy of the shipped method can be read off against the official labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(args.bundle))
    from scoring import problem_features, quartile_layers, score_features, threshold_for

    artifact = joblib.load(args.bundle / "stability_crossmodel.joblib")
    df = pd.read_parquet(args.parquet)
    models = args.models or list(dict.fromkeys(df["model_id"]))
    done = set()
    if args.out.exists():
        done = {(json.loads(l)["model_id"], json.loads(l)["pid"]) for l in args.out.open()}
    for model_id in models:
        problems = list(dict.fromkeys(df[df.model_id == model_id]["problem"]))
        todo = [p for p in problems if (model_id, hashlib.sha1(p.encode()).hexdigest()[:12]) not in done]
        if not todo:
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        device = model.get_input_embeddings().weight.device
        layers = quartile_layers(model)
        threshold = threshold_for(artifact, model_id)
        with args.out.open("a") as fo:
            for index, problem in enumerate(todo):
                pid = hashlib.sha1(problem.encode()).hexdigest()[:12]
                feats = problem_features(model, tokenizer, model_id, problem, f"case_{index}", layers,
                                         device, ("typos", "distract", "wrap"))
                score = score_features(feats, artifact, model_id)
                fo.write(json.dumps({"model_id": model_id, "pid": pid, "score": score,
                                     "threshold": threshold, "feats": feats}) + "\n")
        print(f"{model_id}: {len(todo)} problems scored", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
