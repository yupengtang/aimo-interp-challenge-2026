"""Generate the samples needed to build answer-format self-labels, and score them exactly.

For every problem: sample answers to the untouched statement, then to each answer-format variant
(report the answer modulo 97, twice the answer, the sum of its digits). The variant's correct answer
is a closed form of the known answer, so each sample is scored against an exact target rather than
against the model's own opinion. Accuracy decay between the original and a variant is what a
self-derived robustness label is built from.

Why this validates before it is used: the problems here are the ones that already carry official
labels, so the self-derived label can be compared against the official one. Only if they agree is it
worth generating labels on olympiad problems, where no labels exist.

Truncation is recorded per sample and never silently counted as a wrong answer. At 8192 tokens about a
fifth of DeepSeek's samples still fail to finish, and treating those as incorrect would turn the label
into a difficulty measure, which is the exact confound that ruined the earlier agreement experiment.

Output is appended per batch so an interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aimo_selfperturb import build_prompt, extract_answer, normalize
from answer_format_perturbations import build_variants

MATH_DATASETS = ("hendrycks-math-train-reference-500", "hendrycks-math-test-reference-100")


def load_problems(data_dir: Path, answers: dict[str, str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))]
    src = pd.concat(frames, ignore_index=True)
    src = src[src["dataset_id"].isin(MATH_DATASETS)]
    df = src.drop_duplicates("problem_id")[["problem_id", "original_problem", "dataset_id"]].copy()
    df["answer"] = df["problem_id"].astype(str).map(answers)
    return df[df["answer"].notna()].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--orig-samples", type=int, default=4)
    ap.add_argument("--variant-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--cache-implementation", default="static", choices=["static", "dynamic"])
    ap.add_argument("--limit-problems", type=int, default=0, help="0 means all")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "samples.jsonl"
    timing_path = args.out_dir / "timing.jsonl"

    answers = json.loads(args.answers.read_text())
    problems = load_problems(args.data_dir, answers)
    if args.limit_problems:
        problems = problems.head(args.limit_problems)

    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["problem_id"], r["variant"], r["sample"]))

    jobs = []
    variant_counts = {"orig": 0}
    for _, row in problems.iterrows():
        pid, problem, answer = str(row["problem_id"]), row["original_problem"], str(row["answer"])
        entries = [("orig", problem, answer)]
        for family, text, expected in build_variants(problem, answer):
            entries.append((family, text, expected))
        for variant, text, expected in entries:
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            n = args.orig_samples if variant == "orig" else args.variant_samples
            for s in range(n):
                if (pid, variant, s) not in done:
                    jobs.append((pid, variant, s, text, expected))
    print(f"{len(problems)} problems with answers | variants per kind: {variant_counts}")
    print(f"{len(jobs)} sequences to generate ({len(done)} already done)", flush=True)
    if not jobs:
        print("nothing to do")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", local_files_only=True
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device

    prompts = [build_prompt(tokenizer, args.model, text) for _, _, _, text, _ in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(prompts[i]))
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.cache_implementation == "static":
            gen_kwargs["cache_implementation"] = "static"
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        torch.cuda.synchronize()
        dt = time.time() - t0

        new = gen[:, enc["input_ids"].shape[1]:]
        total_new = 0
        with out_path.open("a") as fo:
            for row, i in zip(new.tolist(), idx):
                pid, variant, sample, _, expected = jobs[i]
                n_new = next((k for k, t in enumerate(row) if t in eos_ids), len(row))
                total_new += n_new
                text = tokenizer.decode(row[:n_new], skip_special_tokens=True)
                got = extract_answer(text)
                hit_limit = n_new >= args.max_new_tokens
                fo.write(json.dumps({
                    "problem_id": pid, "variant": variant, "sample": sample,
                    "expected": expected, "answer": got,
                    # Only a finished sample with an answer can be judged; the rest stay unscored.
                    "correct": None if (got is None or hit_limit) else (got == normalize(expected)),
                    "n_new_tokens": n_new, "hit_limit": hit_limit,
                }) + "\n")
        with timing_path.open("a") as ft:
            ft.write(json.dumps({
                "batch": len(idx), "wall_s": round(dt, 2), "new_tokens": total_new,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }) + "\n")
        print(f"batch {start // args.batch_size}: {len(idx)} seqs, {total_new} tok, {dt:.0f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
