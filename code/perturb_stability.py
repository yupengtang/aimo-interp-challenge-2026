"""Representation stability under meaning-preserving perturbations, forward passes only.

The official label asks whether a model's accuracy survives surface rewrites of a problem it can
already solve. Generating answers for each rewrite is far too slow for the evaluation budget, so this
measures the same thing inside the model: how far its state moves when the problem is rewritten in a
way that cannot change the answer.

For each problem, several cheap programmatic variants are built (typos, an irrelevant sentence, a
neutral wrapper). For the original and every variant we run one forward pass over the chat-formatted
prompt and record the last-token hidden state at four depths plus the next-token distribution. The
features are the similarity of the variant states to the original state and the divergence of the
next-token distributions, aggregated per perturbation family.

No labels are used or produced here. Official labels are joined afterwards for scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aimo_selfperturb import COMPETITION_MODELS, add_distractor, add_typos

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_variants(problem: str, rng: random.Random) -> list[tuple[str, str]]:
    """Meaning-preserving rewrites: the answer cannot change, only the surface."""
    return [
        ("orig", problem),
        ("typos_a", add_typos(problem, random.Random(rng.random()), n=2)),
        ("typos_b", add_typos(problem, random.Random(rng.random()), n=4)),
        ("distract_a", add_distractor(problem, random.Random(rng.random()))),
        ("distract_b", add_distractor(problem, random.Random(rng.random()))),
        ("wrap_pre", f"Here is a problem to solve.\n\n{problem}"),
        ("wrap_post", f"{problem}\n\nTake your time and be careful."),
    ]


@torch.no_grad()
def encode(model, tokenizer, text: str, layers: list[int], device, checkpoint: str):
    kwargs = {"reasoning_effort": "low"} if "gpt-oss" in checkpoint else {}
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{text}\n\n{INSTRUCTION}"}],
        tokenize=False, add_generation_prompt=True, **kwargs,
    )
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model(input_ids=ids, output_hidden_states=True)
    hidden = {l: out.hidden_states[l][0, -1].float() for l in layers}
    logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return hidden, logprobs, int(ids.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--pool", type=Path, help="optional jsonl of unlabeled problems to encode too")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=COMPETITION_MODELS, choices=COMPETITION_MODELS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = pd.concat([pd.read_parquet(p) for p in sorted(args.data_dir.glob("*.parquet"))])
    problems = (
        src.drop_duplicates("problem_id")[["problem_id", "original_problem", "dataset_id"]]
        .rename(columns={"problem_id": "id", "original_problem": "problem", "dataset_id": "source"})
        .to_dict("records")
    )
    if args.pool:
        problems += [
            {"id": r["id"], "problem": r["problem"], "source": "pool"}
            for r in (json.loads(line) for line in args.pool.open())
        ]
    print(f"{len(problems)} problems, {len(build_variants('x', random.Random(0)))} views each", flush=True)

    for checkpoint in args.models:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        device = model.get_input_embeddings().weight.device
        n_layers = model.config.get_text_config().num_hidden_layers
        layers = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers})
        rows = []
        for p in problems:
            rng = random.Random(p["id"])
            variants = build_variants(p["problem"], rng)
            base_hidden, base_lp, base_len = encode(model, tokenizer, variants[0][1], layers, device, checkpoint)
            rec = {"checkpoint": checkpoint, "id": p["id"], "source": p["source"], "prompt_tokens": base_len}
            per_family: dict[str, list[float]] = {}
            for name, text in variants[1:]:
                hidden, lp, n_tok = encode(model, tokenizer, text, layers, device, checkpoint)
                family = name.split("_")[0]
                for l in layers:
                    cos = float(torch.nn.functional.cosine_similarity(base_hidden[l], hidden[l], dim=0))
                    l2 = float(torch.linalg.vector_norm(base_hidden[l] - hidden[l]) / torch.linalg.vector_norm(base_hidden[l]))
                    rec[f"cos_l{l}_{name}"] = cos
                    rec[f"rel_l2_l{l}_{name}"] = l2
                    per_family.setdefault(f"cos_l{l}_{family}", []).append(cos)
                kl = float((base_lp.exp() * (base_lp - lp)).sum())
                js = 0.5 * float((base_lp.exp() * (base_lp - lp)).sum() + (lp.exp() * (lp - base_lp)).sum())
                rec[f"kl_{name}"] = kl
                rec[f"js_{name}"] = js
                rec[f"dtok_{name}"] = n_tok - base_len
                per_family.setdefault(f"kl_{family}", []).append(kl)
            for key, vals in per_family.items():
                rec[f"{key}_mean"] = float(np.mean(vals))
                rec[f"{key}_min"] = float(np.min(vals))
            rows.append(rec)
        safe = checkpoint.replace("/", "__")
        pd.DataFrame(rows).to_csv(args.out_dir / f"stability_{safe}.csv", index=False)
        print(f"{checkpoint}: {len(rows)} problems in {time.time() - t0:.0f}s, layers {layers}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
