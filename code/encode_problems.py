"""Forward-pass encodings of an unlabeled problem pool for every competition checkpoint.

Writes, per checkpoint, the last-token hidden state of the raw problem text at a few depths plus the
scalar log-prob statistics used in aimo_cheap_features.py. No generation and no labels. The pool is
used for label-free domain adaptation of probes (for example standardising olympiad encodings with
olympiad statistics instead of MATH statistics).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aimo_cheap_features import COMPETITION_MODELS, token_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, type=Path, help="jsonl with id and problem fields")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=COMPETITION_MODELS, choices=COMPETITION_MODELS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    problems = [json.loads(line) for line in args.problems.open()]
    print(f"{len(problems)} problems", flush=True)

    for checkpoint in args.models:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        device = model.get_input_embeddings().weight.device
        n_layers = model.config.get_text_config().num_hidden_layers
        layers = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers})
        rows, hid = [], {}
        for p in problems:
            feats, hidden = token_stats(model, tokenizer, p["problem"], layers, device)
            feats.update({"checkpoint": checkpoint, "id": p["id"], "source": p.get("source")})
            rows.append(feats)
            for k, v in hidden.items():
                hid[f"{p['id']}|raw|{k}"] = v
        safe = checkpoint.replace("/", "__")
        pd.DataFrame(rows).to_csv(args.out_dir / f"features_{safe}.csv", index=False)
        np.savez_compressed(args.out_dir / f"hidden_{safe}.npz", **hid)
        print(f"{checkpoint}: {len(problems)} problems in {time.time() - t0:.0f}s (layers {layers})", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
