"""Does a judge's reaction to an inserted contradiction predict the official robustness label?

In the public sample the perturbation that most often breaks a model is `expert_no_solution`: one
added condition contradicts the rest, and the model is expected to answer NaN. This script builds
such variants automatically and measures how often the judge notices.

For every train-main-v2 problem, with gpt-oss-120b at low effort and the organizers' solve prompt:
  1. solve the original several times,
  2. write a few no-solution variants of it,
  3. solve each variant several times.
Everything is appended to jsonl files so a run can be resumed and analysed offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOLVE = (
    "You are a precise math problem solver. Solve the given problem step by step, then output your "
    "final answer on the last line in the exact format:\nANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
REWRITE = (
    "You edit math problems. Insert exactly one extra condition into the problem below so that the "
    "conditions contradict each other and the problem has no solution. The added condition must look "
    "natural, use the problem's own objects, and must not mention that anything is impossible. Keep "
    "every other word of the problem unchanged. Output only the edited problem, nothing else."
)
REWRITE_SUBTLE = (
    "You edit math problems so that they have no solution, the way a careful examiner would: the edit "
    "must look like an ordinary part of the problem, and the contradiction must only show up once the "
    "problem is actually worked out. Use one of two edits. (a) Add one sentence stating a bound or a "
    "value for a quantity that follows from the other conditions, chosen so that it cannot hold. "
    "(b) Change one or two of the given numbers so that the described situation cannot exist. Never "
    "state two plainly opposite facts, and never say that anything is impossible. Keep the rest of "
    "the problem and its final question unchanged. Output only the edited problem, nothing else.\n\n"
    "Example of (a).\nProblem: Three airline companies operate flights from Dodola island. Each "
    "company has a different schedule of departures. The first company departs every 100 days, the "
    "second every 120 days, and the third every 150 days. What is the greatest positive integer d for "
    "which it is true that there will be d consecutive days without a flight from Dodola island, "
    "regardless of the departure times of the various airlines\nEdited: Three airline companies "
    "operate flights from Dodola island. Each company has a different schedule of departures. The "
    "first company departs every 100 days, the second every 120 days, and the third every 150 days. "
    "Over a full synchronized cycle of 600 days, the system registers a cumulative total of at least "
    "21 individual departures. What is the greatest positive integer d for which it is true that "
    "there will be d consecutive days without a flight from Dodola island, regardless of the "
    "departure times of the various airlines\n\n"
    "Example of (b).\nProblem: In triangle $PQR$ we have $PQ=13$, $QR=14$ and $RP=15$. Find the "
    "length of the altitude from $P$.\nEdited: In triangle $PQR$ we have $PQ=13$, $QR=31$ and "
    "$RP=15$. Find the length of the altitude from $P$."
)
FINAL = "<|channel|>final<|message|>"


def final_text(tokenizer, row: list[int], eos_ids: set[int]) -> tuple[str | None, int, bool]:
    """(final-channel text, tokens generated, finished)."""
    end = next((k for k, t in enumerate(row) if t in eos_ids), None)
    if end is None:
        return None, len(row), False
    text = tokenizer.decode(row[:end], skip_special_tokens=False)
    if FINAL in text:
        text = text.split(FINAL)[-1]
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    return text, end, True


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    found = re.findall(r"ANSWER:\s*([^\n]+)", text)
    if not found:
        return None
    a = found[-1].strip().strip("$*` .").replace(",", "")
    if a.lower() == "nan":
        return "NaN"
    try:
        v = float(a)
        return str(int(v)) if v.is_integer() else str(v)
    except ValueError:
        return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--orig-samples", type=int, default=8)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--variant-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rewrite", default="plain", choices=["plain", "subtle"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants_path = args.out_dir / "variants.jsonl"
    solves_path = args.out_dir / "solves.jsonl"

    problems = list(dict.fromkeys(pd.read_parquet(args.parquet)["problem"].tolist()))
    pid = {p: hashlib.sha1(p.encode()).hexdigest()[:12] for p in problems}

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

    def chat(system: str, user: str, effort: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, reasoning_effort=effort,
        )

    def generate(prompts: list[str], max_new: int, seed: int):
        out = []
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            torch.manual_seed(seed + start)
            torch.cuda.manual_seed_all(seed + start)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_p=0.95)
            new = gen[:, enc["input_ids"].shape[1]:]
            print(f"  batch of {len(idx)}: {new.shape[1]} steps in {time.time() - t0:.0f}s", flush=True)
            for row, i in zip(new.tolist(), idx):
                out.append((i, final_text(tokenizer, row, eos_ids)))
        return [r for _, r in sorted(out)]

    # 1. no-solution variants
    variants: dict[str, list[str]] = {}
    if variants_path.exists():
        for line in variants_path.open():
            r = json.loads(line)
            variants.setdefault(r["pid"], []).append(r["variant"])
    todo = [(p, v) for p in problems for v in range(len(variants.get(pid[p], [])), args.variants)]
    if todo:
        print(f"writing {len(todo)} variants", flush=True)
        res = generate([chat(REWRITE_SUBTLE if args.rewrite == "subtle" else REWRITE, p, "medium") for p, _ in todo], 4096, args.seed)
        with variants_path.open("a") as fo:
            for (p, v), (text, n, finished) in zip(todo, res):
                ok = bool(finished and text and text.strip() != p.strip() and len(text) < 3 * len(p) + 400)
                fo.write(json.dumps({"pid": pid[p], "index": v, "variant": text if ok else None,
                                     "tokens": n, "finished": finished}) + "\n")
                variants.setdefault(pid[p], []).append(text if ok else None)

    # 2. solve originals and variants with the organizers' prompt
    done = set()
    if solves_path.exists():
        done = {(json.loads(l)["pid"], json.loads(l)["kind"], json.loads(l)["sample"]) for l in solves_path.open()}
    jobs = []
    for p in problems:
        for s in range(args.orig_samples):
            if (pid[p], "orig", s) not in done:
                jobs.append((pid[p], "orig", s, p))
        for v, text in enumerate(variants.get(pid[p], [])):
            if text is None:
                continue
            for s in range(args.variant_samples):
                if (pid[p], f"nosol{v}", s) not in done:
                    jobs.append((pid[p], f"nosol{v}", s, text))
    print(f"solving {len(jobs)} sequences", flush=True)
    res = generate([chat(SOLVE, text, args.effort) for _, _, _, text in jobs], args.max_new_tokens, args.seed + 1000)
    with solves_path.open("a") as fo:
        for (p, kind, s, _), (text, n, finished) in zip(jobs, res):
            fo.write(json.dumps({"pid": p, "kind": kind, "sample": s, "answer": extract_answer(text),
                                 "tokens": n, "finished": finished, "tail": (text or "")[-200:]}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
