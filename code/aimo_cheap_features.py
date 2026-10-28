"""Cheap white-box signals on the official AIMO labeled cases, plus generation throughput.

For every (local checkpoint, problem) pair that has an official robustness label, run one
forward pass over the raw problem text and the chat-formatted prompt and record
membership-inference style statistics (mean log-prob, Min-K%, Max-K%, zlib ratio) and the
last-token hidden state at a few layers. Then time greedy generation on a few prompts.

Only official public data is used (aimo-interp HF datasets). No labels are created here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Public dataset aliases that correspond to checkpoints cached on the Codabench workers.
ALIAS_TO_CHECKPOINT = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "gpt-oss-120b:low": "openai/gpt-oss-120b",
}
COMPETITION_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b",
]


def load_labeled_cases(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        df["source_file"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["checkpoint"] = df["model_id"].map(ALIAS_TO_CHECKPOINT)
    df = df[df["checkpoint"].notna()].copy()
    if "model_is_robust" not in df.columns:
        df["model_is_robust"] = np.nan
    # Aggregate per (checkpoint, problem): a case is robust only if no row reports a detrimental variant,
    # which matches every labeled public row (label == n_detrimental_permutations == 0).
    grouped = (
        df.groupby(["checkpoint", "problem_id"], as_index=False)
        .agg(
            original_problem=("original_problem", "first"),
            n_rows=("problem_id", "size"),
            n_detrimental=("n_detrimental_permutations", "sum"),
            base_accuracy=("base_accuracy", "mean"),
            max_decay=("absolute_accuracy_decay", "max"),
            official_label=("model_is_robust", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
            sources=("source_file", lambda s: ",".join(sorted(set(s)))),
        )
    )
    grouped["derived_label"] = grouped["n_detrimental"] == 0
    return grouped


def token_stats(model, tokenizer, text: str, layers: list[int], device) -> tuple[dict, dict]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    logits = out.logits[0, :-1].float()
    targets = ids[0, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(1, targets[:, None])[:, 0].cpu().numpy()
    probs = logprobs.exp()
    entropy = -(probs * logprobs).sum(-1).cpu().numpy()
    feats = {"n_tokens": int(tok_lp.size), "mean_logprob": float(tok_lp.mean()) if tok_lp.size else math.nan}
    for k in (0.1, 0.2, 0.3):
        m = max(1, int(round(tok_lp.size * k)))
        srt = np.sort(tok_lp)
        feats[f"min_{int(k * 100)}_logprob"] = float(srt[:m].mean())
        feats[f"max_{int(k * 100)}_logprob"] = float(srt[-m:].mean())
    feats["mean_entropy"] = float(entropy.mean()) if entropy.size else math.nan
    feats["zlib_ratio"] = float(-tok_lp.sum() / max(1, len(zlib.compress(text.encode("utf-8")))))
    hidden = {f"layer_{l}": out.hidden_states[l][0, -1].float().cpu().numpy() for l in layers}
    return feats, hidden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--throughput-prompts", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--models", nargs="+", default=COMPETITION_MODELS, choices=COMPETITION_MODELS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_labeled_cases(args.data_dir)
    cases.to_csv(args.out_dir / "labeled_cases.csv", index=False)
    print("labeled cases per checkpoint:", cases.groupby("checkpoint")["derived_label"].agg(["size", "sum"]).to_dict())
    agree = cases["official_label"].dropna()
    if len(agree):
        print("official vs derived label agreement:", float((cases.loc[agree.index, "derived_label"] == agree.astype(bool)).mean()))

    all_problems = cases.drop_duplicates("problem_id")[["problem_id", "original_problem"]]
    throughput = []
    for checkpoint in args.models:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        load_s = time.time() - t0
        device = model.get_input_embeddings().weight.device
        n_layers = model.config.get_text_config().num_hidden_layers
        layers = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers})
        rows, hid = [], {}
        # Features for every labeled problem on every competition model (unlabeled for most models,
        # still useful as unlabeled data); labels are joined later only where they exist.
        t1 = time.time()
        for _, r in all_problems.iterrows():
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": r["original_problem"]}], tokenize=False, add_generation_prompt=True
            )
            for view, text in (("raw", r["original_problem"]), ("chat", chat)):
                feats, hidden = token_stats(model, tokenizer, text, layers, device)
                feats.update({"checkpoint": checkpoint, "problem_id": r["problem_id"], "view": view})
                rows.append(feats)
                for k, v in hidden.items():
                    hid[f"{r['problem_id']}|{view}|{k}"] = v
        feat_s = time.time() - t1
        pd.DataFrame(rows).to_csv(args.out_dir / f"features_{checkpoint.replace('/', '__')}.csv", index=False)
        np.savez_compressed(args.out_dir / f"hidden_{checkpoint.replace('/', '__')}.npz", **hid)

        # Greedy generation throughput on a few prompts.
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in all_problems["original_problem"].head(args.throughput_prompts)
        ]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        torch.cuda.synchronize()
        t2 = time.time()
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        gen_s = time.time() - t2
        new_tokens = int((gen[:, batch["input_ids"].shape[1]:] != tokenizer.pad_token_id).sum())
        throughput.append(
            {
                "checkpoint": checkpoint,
                "load_s": round(load_s, 1),
                "forward_features_s_per_problem_view": round(feat_s / max(1, 2 * len(all_problems)), 3),
                "gen_batch": len(prompts),
                "gen_new_tokens": new_tokens,
                "gen_s": round(gen_s, 1),
                "gen_tokens_per_s": round(new_tokens / gen_s, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
        )
        print(json.dumps(throughput[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    pd.DataFrame(throughput).to_csv(args.out_dir / "throughput.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
