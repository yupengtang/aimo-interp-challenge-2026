"""Judge targets and gpt-oss encodings for an unlabelled problem pool.

The black-box judge (bundle_v3: gpt-oss-120b, low effort, 8 samples, 3072 tokens) scores how
consistently gpt-oss answers a problem. This script computes that score on a pool of unlabelled
problems, and the hidden states gpt-oss produces for the same chat prompt, so a probe can learn the
judge's verdict from one forward pass.

  python judge_pool.py --problems pool.jsonl --out out_dir [--skip-generate] [--skip-encode]

Writes out_dir/judge.jsonl (one line per problem, resumable) and out_dir/hidden.npz
(for each layer in LAYERS: last prompt token and mean over the prompt tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_v3 as judge  # noqa: E402  (the exact judge prompt, sampling and answer parsing used for the targets)

LAYERS = (9, 18, 27, 36)


def prompt_for(tokenizer, problem: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\n{judge.INSTRUCTION}"}], tokenize=False,
        add_generation_prompt=True, reasoning_effort=judge.JUDGE_EFFORT,
    )


def summarize(samples):
    answers = [a for a, _ in samples if a is not None]
    lengths = [n for _, n in samples]
    modal = max((answers.count(a) for a in set(answers)), default=0)
    return {
        "answers": [a for a, _ in samples],
        "tokens": lengths,
        "distinct": len(set(answers)),
        "finished": len(answers) / len(samples),
        "modal_share": modal / len(samples),
        "median_tokens": statistics.median(lengths),
        "verdict": bool(judge._verdict(samples)),
    }


def generate(problems, out_path: Path) -> None:
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open()}
    todo = [p for p in problems if p["id"] not in done]
    print(f"generate: {len(done)} done, {len(todo)} to go", flush=True)
    _, tokenizer, _, _ = judge._judge
    todo.sort(key=lambda p: len(p["problem"]))
    per_batch = max(1, judge.BATCH_SIZE // judge.SAMPLES)
    with out_path.open("a") as fh:
        for pos in range(0, len(todo), per_batch):
            chunk = todo[pos:pos + per_batch]
            batch = [prompt_for(tokenizer, p["problem"]) for p in chunk for _ in range(judge.SAMPLES)]
            t0 = time.time()
            answers = judge._generate(batch, judge.SEED + pos, time_limit_s=1e9)
            for i, p in enumerate(chunk):
                row = {"id": p["id"], "source": p.get("source"), **summarize(answers[i * judge.SAMPLES:(i + 1) * judge.SAMPLES])}
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"batch {pos // per_batch}: {len(chunk)} problems in {time.time() - t0:.0f}s", flush=True)


def encode(problems, out_path: Path) -> None:
    import torch

    model, tokenizer, device, _ = judge._judge
    feats = {f"{kind}{layer}": [] for layer in LAYERS for kind in ("last", "mean")}
    t0 = time.time()
    for p in problems:
        ids = tokenizer(prompt_for(tokenizer, p["problem"]), return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states
        for layer in LAYERS:
            h = hs[layer][0].float()
            feats[f"last{layer}"].append(h[-1].cpu().numpy())
            feats[f"mean{layer}"].append(h.mean(0).cpu().numpy())
    np.savez_compressed(out_path, ids=np.array([p["id"] for p in problems]),
                        **{k: np.stack(v).astype(np.float32) for k, v in feats.items()})
    print(f"encode: {len(problems)} problems in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-encode", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    problems = [json.loads(line) for line in args.problems.open()]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not judge._load_judge():
        raise SystemExit("judge failed to load")
    if not args.skip_encode:
        encode(problems, args.out / "hidden.npz")
    if not args.skip_generate:
        generate(problems, args.out / "judge.jsonl")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
