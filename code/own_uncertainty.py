"""One greedy trace per problem from the evaluated model itself, with per-token uncertainty.

The judge experiments measure how settled a strong model is on a problem. This measures the model
that is actually being evaluated: it solves each train-main-v2 problem once, greedily, with the
organizers' solve prompt, for a capped number of tokens, and the entropy, top-1 probability and
top-1/top-2 margin of every generated token are recorded. Nothing is stored but those three numbers
per step, so memory stays flat.

Output: one json line per problem with the per-step arrays, the number of tokens and whether the
trace finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

SOLVE = (
    "You are a precise math problem solver. Solve the given problem step by step, then output your "
    "final answer on the last line in the exact format:\nANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)


class Recorder(LogitsProcessor):
    def __init__(self):
        self.entropy, self.top1, self.margin = [], [], []

    def __call__(self, input_ids, scores):
        logp = torch.log_softmax(scores.float(), dim=-1)
        p = logp.exp()
        self.entropy.append(-(p * logp).sum(-1).cpu())
        top = p.topk(2, dim=-1).values
        self.top1.append(top[:, 0].cpu())
        self.margin.append((top[:, 0] - top[:, 1]).cpu())
        return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=36)
    ap.add_argument("--cache-implementation", default="dynamic", choices=["static", "dynamic"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1] + (f"_{args.effort}" if args.effort else "")
    out_path = args.out_dir / f"trace_{tag}.jsonl"

    problems = list(dict.fromkeys(pd.read_parquet(args.parquet)["problem"].tolist()))
    pid = {p: hashlib.sha1(p.encode()).hexdigest()[:12] for p in problems}
    done = {json.loads(l)["pid"] for l in out_path.open()} if out_path.exists() else set()
    todo = [p for p in problems if pid[p] not in done]
    print(f"{tag}: {len(todo)} problems to trace", flush=True)
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", local_files_only=True
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    kwargs = {"reasoning_effort": args.effort} if args.effort else {}

    def prompt(p: str) -> str:
        messages = [{"role": "system", "content": SOLVE}, {"role": "user", "content": p}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except Exception:  # template without a system role
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": SOLVE + "\n\n" + p}], tokenize=False, add_generation_prompt=True, **kwargs)

    order = sorted(todo, key=lambda p: len(p))
    for start in range(0, len(order), args.batch_size):
        chunk = order[start:start + args.batch_size]
        enc = tokenizer([prompt(p) for p in chunk], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        rec = Recorder()
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                      "logits_processor": LogitsProcessorList([rec])}
        if args.cache_implementation == "static":
            gen_kwargs["cache_implementation"] = "static"
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        new = gen[:, enc["input_ids"].shape[1]:]
        ent, top1, margin = (torch.stack(x, 1) for x in (rec.entropy, rec.top1, rec.margin))
        print(f"  batch of {len(chunk)}: {new.shape[1]} steps in {time.time() - t0:.0f}s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        with out_path.open("a") as fo:
            for i, p in enumerate(chunk):
                row = new[i].tolist()
                n = next((k for k, t in enumerate(row) if t in eos_ids), len(row))
                text = tokenizer.decode(row[:n], skip_special_tokens=True)
                fo.write(json.dumps({
                    "pid": pid[p], "tokens": n, "finished": n < len(row) or n < args.max_new_tokens,
                    "entropy": [round(float(v), 4) for v in ent[i, :n]],
                    "top1": [round(float(v), 4) for v in top1[i, :n]],
                    "margin": [round(float(v), 4) for v in margin[i, :n]],
                    "tail": text[-200:],
                }) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
