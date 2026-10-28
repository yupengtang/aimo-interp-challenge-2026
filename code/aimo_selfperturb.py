"""Label-free behavioural robustness signal, measured against the official public labels.

For each (checkpoint, problem) case, sample several answers to the original problem and to a few
cheap programmatic variants (an irrelevant sentence, harmless typos). The signal is how often the
variant answers agree with the model's modal answer on the original. No labels are produced or used
to build variants; the official labels are only joined afterwards to score the signal. Generation
cost on the evaluation-class GPU is logged per batch so the 3600 s budget can be planned.

Output is appended to a JSONL file so an interrupted job resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Neutral sentences that carry no quantity the problem could use.
DISTRACTORS = [
    "This problem was shared in a study group that meets on Tuesday evenings.",
    "A student once remarked that the problem looked harder than it really was.",
    "The problem statement was copied from a notebook with a blue cover.",
    "Several people read this problem aloud before anyone tried to solve it.",
]


def split_math(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_math, chunk) so that edits never touch LaTeX or Asymptote code."""
    parts, pos = [], 0
    for m in re.finditer(r"\[asy\].*?\[/asy\]|\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", text, flags=re.S):
        if m.start() > pos:
            parts.append((False, text[pos:m.start()]))
        parts.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        parts.append((False, text[pos:]))
    return parts


def add_typos(text: str, rng: random.Random, n: int = 2) -> str:
    parts = split_math(text)
    slots = [
        (i, m.start())
        for i, (is_math, chunk) in enumerate(parts)
        if not is_math
        for m in re.finditer(r"\b[a-zA-Z]{5,}\b", chunk)
    ]
    for i, start in rng.sample(slots, min(n, len(slots))):
        chunk = parts[i][1]
        j = start + rng.randrange(1, 3)
        chunk = chunk[:j] + chunk[j + 1] + chunk[j] + chunk[j + 2:]
        parts[i] = (False, chunk)
    return "".join(chunk for _, chunk in parts)


def add_distractor(text: str, rng: random.Random) -> str:
    sentence = rng.choice(DISTRACTORS)
    parts = split_math(text)
    # Insert after the first sentence end that lies outside math, else prepend.
    for i, (is_math, chunk) in enumerate(parts):
        if is_math:
            continue
        m = re.search(r"[.!?]\s+", chunk)
        if m:
            parts[i] = (False, chunk[: m.end()] + sentence + " " + chunk[m.end():])
            return "".join(c for _, c in parts)
    return sentence + " " + text


def extract_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx >= 0:
        depth, out = 1, []
        for ch in text[idx + len("\\boxed{"):]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
        if depth == 0:
            return normalize("".join(out))
    m = re.findall(r"ANSWER:\s*([^\n]+)", text)
    return normalize(m[-1]) if m else None


def normalize(ans: str) -> str:
    a = ans.strip().strip("$").strip().rstrip(".")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace("\\!", "").replace("\\,", "").replace(" ", "")
    a = re.sub(r"^\\?(dfrac|tfrac)", r"\\frac", a)
    try:
        v = float(a.replace(",", ""))
        return str(int(v)) if v.is_integer() else str(v)
    except ValueError:
        return a


def load_cases(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        df["source_file"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["checkpoint"] = df["model_id"].map(ALIAS_TO_CHECKPOINT)
    df = df[df["checkpoint"].notna()]
    g = df.groupby(["checkpoint", "problem_id"], as_index=False).agg(
        original_problem=("original_problem", "first"),
        dataset_id=("dataset_id", "first"),
        n_detrimental=("n_detrimental_permutations", "sum"),
    )
    g["label_robust"] = g["n_detrimental"] == 0
    return g


def build_prompt(tokenizer, checkpoint: str, problem: str) -> str:
    kwargs = {"reasoning_effort": "low"} if "gpt-oss" in checkpoint else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=COMPETITION_MODELS, choices=COMPETITION_MODELS)
    ap.add_argument("--orig-samples", type=int, default=4)
    ap.add_argument("--variant-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--cache-implementation", default=None, choices=[None, "static"],
                    help="static lets transformers compile the decoding step, which is much faster for some models")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict to these dataset_id values, e.g. the MATH sets that have enough labelled cases")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "generations.jsonl"
    timing_path = args.out_dir / "timing.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["checkpoint"], r["problem_id"], r["variant"], r["sample"]))

    cases = load_cases(args.data_dir)
    cases.to_csv(args.out_dir / "cases.csv", index=False)
    olympiad = cases[cases["dataset_id"] == "aimo-2025-reference"].drop_duplicates("problem_id")
    if args.datasets:
        cases = cases[cases["dataset_id"].isin(args.datasets)]
        print(f"restricted to {args.datasets}: {len(cases)} cases", flush=True)

    for checkpoint in args.models:
        own = cases[cases["checkpoint"] == checkpoint]
        # Unlabeled checkpoints still run the olympiad problems, for cost and answer statistics.
        todo_cases = own if len(own) else olympiad.assign(checkpoint=checkpoint, label_robust=None)
        jobs = []
        for _, c in todo_cases.iterrows():
            rng = random.Random(f"{c['problem_id']}")
            variants = [("orig", c["original_problem"], args.orig_samples)]
            variants.append(("distract", add_distractor(c["original_problem"], rng), args.variant_samples))
            variants.append(("typos", add_typos(c["original_problem"], rng), args.variant_samples))
            for name, text, k in variants:
                for s in range(k):
                    if (checkpoint, c["problem_id"], name, s) not in done:
                        jobs.append((c["problem_id"], name, s, text))
        print(f"{checkpoint}: {len(todo_cases)} cases, {len(jobs)} sequences to generate", flush=True)
        if not jobs:
            continue

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        device = model.get_input_embeddings().weight.device
        prompts = [build_prompt(tokenizer, checkpoint, text) for _, _, _, text in jobs]
        order = sorted(range(len(jobs)), key=lambda i: len(prompts[i]))
        bs = args.batch_size // 2 if "gpt-oss" in checkpoint else args.batch_size

        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            torch.cuda.synchronize()
            t0 = time.time()
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if args.cache_implementation:
                gen_kwargs["cache_implementation"] = args.cache_implementation
            with torch.no_grad():
                gen = model.generate(**enc, **gen_kwargs)
            torch.cuda.synchronize()
            dt = time.time() - t0
            new = gen[:, enc["input_ids"].shape[1]:]
            eos_ids = model.generation_config.eos_token_id
            eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
            total_new = 0
            with out_path.open("a") as fo:
                for row, i in zip(new.tolist(), idx):
                    pid, variant, sample, _ = jobs[i]
                    n_new = next((k for k, t in enumerate(row) if t in eos_ids), len(row))
                    total_new += n_new
                    text = tokenizer.decode(row[:n_new], skip_special_tokens=True)
                    fo.write(json.dumps({
                        "checkpoint": checkpoint, "problem_id": pid, "variant": variant, "sample": sample,
                        "n_new_tokens": n_new, "hit_limit": n_new >= args.max_new_tokens,
                        "answer": extract_answer(text), "tail": text[-300:],
                    }) + "\n")
            with timing_path.open("a") as ft:
                ft.write(json.dumps({
                    "checkpoint": checkpoint, "batch": len(idx), "wall_s": round(dt, 2),
                    "new_tokens": total_new, "max_len": int(new.shape[1]),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                }) + "\n")
            print(f"{checkpoint} batch {b // bs}: {len(idx)} seqs, {total_new} tok, {dt:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    summarize(args.out_dir, cases)


def summarize(out_dir: Path, cases: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score

    gens = pd.read_json(out_dir / "generations.jsonl", lines=True)
    rows = []
    for (ck, pid), g in gens.groupby(["checkpoint", "problem_id"]):
        orig = g[g["variant"] == "orig"]["answer"].dropna().tolist()
        mode, mode_n = Counter(orig).most_common(1)[0] if orig else (None, 0)
        var = g[g["variant"] != "orig"]
        rows.append({
            "checkpoint": ck, "problem_id": pid,
            "orig_consistency": mode_n / max(1, (g["variant"] == "orig").sum()),
            "variant_agreement": float((var["answer"] == mode).mean()) if mode is not None and len(var) else 0.0,
            "all_agreement": float((g["answer"] == mode).mean()) if mode is not None else 0.0,
            "hit_limit_rate": float(g["hit_limit"].mean()),
            "mean_new_tokens": float(g["n_new_tokens"].mean()),
        })
    s = pd.DataFrame(rows).merge(cases[["checkpoint", "problem_id", "dataset_id", "label_robust"]], how="left")
    s.to_csv(out_dir / "case_signals.csv", index=False)
    lab = s[s["label_robust"].notna()]
    for (ck, ds), g in lab.groupby(["checkpoint", "dataset_id"]):
        y = g["label_robust"].astype(int)
        line = f"{ck} {ds} n={len(g)} robust={int(y.sum())}"
        if y.nunique() == 2:
            for f in ("orig_consistency", "variant_agreement", "all_agreement", "hit_limit_rate", "mean_new_tokens"):
                line += f" {f}_auc={roc_auc_score(y, g[f]):.3f}"
        print(line)
    t = pd.read_json(out_dir / "timing.jsonl", lines=True)
    print(t.groupby("checkpoint").agg(seqs=("batch", "sum"), wall_s=("wall_s", "sum"), tok=("new_tokens", "sum"), peak=("peak_mem_gb", "max")))


if __name__ == "__main__":
    main()
