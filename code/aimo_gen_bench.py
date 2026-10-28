"""Generation throughput versus batch size and KV-cache implementation.

HF generate on a single GPU spends most of each decoding step in Python and kernel launch overhead
at small batch sizes, so tokens per second should grow almost linearly with batch size until the GPU
saturates. This measures where that stops for each competition checkpoint, with the dynamic cache
and with the static cache (which lets transformers compile the forward pass). The first call of a
configuration includes compilation, so both the first and a repeated call are timed.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64, 128])
    ap.add_argument("--caches", nargs="+", default=["dynamic", "static"])
    ap.add_argument("--new-tokens", type=int, default=512)
    args = ap.parse_args()

    problems = (
        pd.read_parquet(args.data_dir / "full_validation-00000-of-00001.parquet")
        .drop_duplicates("problem_id")["original_problem"]
        .tolist()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.models:
        try:
            run_checkpoint(checkpoint, problems, args)
        except Exception as e:  # one model running out of memory must not end the sweep
            print(json.dumps({"checkpoint": checkpoint, "fatal": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def run_checkpoint(checkpoint: str, problems: list[str], args: argparse.Namespace) -> None:
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        device = model.get_input_embeddings().weight.device
        kwargs = {"reasoning_effort": "low"} if "gpt-oss" in checkpoint else {}
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{p}\n\n{INSTRUCTION}"}],
                tokenize=False, add_generation_prompt=True, **kwargs,
            )
            for p in problems
        ]
        for cache in args.caches:
            for bs in args.batch_sizes:
                batch = [prompts[i % len(prompts)] for i in range(bs)]
                enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                gen_kwargs = dict(
                    max_new_tokens=args.new_tokens,
                    min_new_tokens=args.new_tokens,  # fixed length so calls are comparable
                    do_sample=True, temperature=0.6, top_p=0.95,
                )
                if cache == "static":
                    gen_kwargs["cache_implementation"] = "static"
                for call in ("first", "repeat"):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time.time()
                    try:
                        with torch.no_grad():
                            out = model.generate(**enc, **gen_kwargs)
                        torch.cuda.synchronize()
                        dt = time.time() - t0
                        new = int(out.shape[1] - enc["input_ids"].shape[1]) * bs
                        rec = {
                            "checkpoint": checkpoint, "cache": cache, "batch": bs, "call": call,
                            "prompt_len": int(enc["input_ids"].shape[1]), "wall_s": round(dt, 2),
                            "tok_per_s": round(new / dt, 1), "ms_per_step": round(1000 * dt / args.new_tokens, 1),
                            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                        }
                    except Exception as e:  # record OOM or compile failures and keep going
                        torch.cuda.empty_cache()
                        rec = {"checkpoint": checkpoint, "cache": cache, "batch": bs, "call": call,
                               "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    print(json.dumps(rec), flush=True)
                    with args.out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    if "error" in rec:
                        break
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
