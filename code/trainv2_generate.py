"""Sample each train-main-v2 problem several times with one checkpoint.

The official train set shows that robustness is mostly a property of the problem: configurations
that share a problem almost always share its label. This collects what a difficulty-style signal
needs, per problem and checkpoint: the answers of several independent samples, how long each trace
ran, and whether it finished. Nothing is scored against a reference, so no answers are required.

Output is appended per batch, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aimo_selfperturb import INSTRUCTION, build_prompt, extract_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--cache-implementation", default="static", choices=["static", "dynamic"])
    ap.add_argument("--effort", default=None, help="reasoning effort for gpt-oss (low, medium, high)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1] + (f"_{args.effort}" if args.effort else "")
    out_path = args.out_dir / f"samples_{tag}.jsonl"
    timing_path = args.out_dir / f"timing_{tag}.jsonl"

    problems = list(dict.fromkeys(pd.read_parquet(args.parquet)["problem"].tolist()))
    pid = {p: hashlib.sha1(p.encode()).hexdigest()[:12] for p in problems}
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["pid"], r["sample"]))
    jobs = [(pid[p], s, p) for p in problems for s in range(args.samples) if (pid[p], s) not in done]
    print(f"{len(problems)} problems, {len(jobs)} sequences to generate ({len(done)} done)", flush=True)
    if not jobs:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", local_files_only=True
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    if args.effort:
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{text}\n\n{INSTRUCTION}"}], tokenize=False,
            add_generation_prompt=True, reasoning_effort=args.effort) for _, _, text in jobs]
    else:
        prompts = [build_prompt(tokenizer, args.model, text) for _, _, text in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(prompts[i]))
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": True,
                      "temperature": args.temperature, "top_p": args.top_p}
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
                p, sample, _ = jobs[i]
                n_new = next((k for k, t in enumerate(row) if t in eos_ids), len(row))
                total_new += n_new
                text = tokenizer.decode(row[:n_new], skip_special_tokens=True)
                fo.write(json.dumps({
                    "pid": p, "sample": sample, "answer": extract_answer(text),
                    "n_new_tokens": n_new, "hit_limit": n_new >= args.max_new_tokens,
                    "tail": text[-300:],
                }) + "\n")
        with timing_path.open("a") as ft:
            ft.write(json.dumps({"batch": len(idx), "wall_s": round(dt, 2), "new_tokens": total_new,
                                 "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}) + "\n")
        print(f"batch {start // args.batch_size}: {len(idx)} seqs, {total_new} tok, {dt:.0f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
