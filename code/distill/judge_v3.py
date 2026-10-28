"""Codabench entry point: robustness verdicts from answer agreement on the problem.

The official train set shows that robustness is mostly a property of the problem: configurations
that share a problem almost always share its label, and the problems that lose accuracy under
perturbation are the ones a strong model cannot answer consistently. So each problem is judged once.
gpt-oss-120b answers it several times at low reasoning effort, and the problem is called robust when
the samples settle on at most two different answers and the typical trace is short. The verdict is
reused for every checkpoint and effort that is asked about the same problem.

The harness calls `are_robust` once per checkpoint and effort and limits the whole run to 3600 s, so
the judge is loaded once and kept, verdicts are cached by problem text, and a batch is only started
when it can finish in the time that is left. Problems that cannot be judged in time, or at all if the
judge fails to load, fall back to a prior from the problem text. The function never raises.
"""

from __future__ import annotations

import os
import re
import sys
import time

_START = time.monotonic()
RUN_LIMIT_S = 3600.0
SAFETY_MARGIN_S = 300.0

JUDGE_ID = "openai/gpt-oss-120b"
JUDGE_EFFORT = "low"
SAMPLES = 8
MAX_NEW_TOKENS = 3072
BATCH_SIZE = 96
# A problem is robust when its samples give at most this many different answers ...
MAX_DISTINCT = 2
# ... and the median trace stays under this many tokens.
MAX_MEDIAN_TOKENS = 2200
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 20261101
INSTRUCTION = "Please reason step by step, and put your final answer within " + chr(92) + "boxed{}."
BOXED = chr(92) + "boxed{"
# Seconds per decoding step before anything has been measured; replaced by the first batch.
STEP_TIME_GUESS_S = 0.1
# Prior used when the judge is unavailable: short statements with little LaTeX lean robust.
PRIOR_MAX_DOLLARS = 14

_verdicts: dict[str, bool] = {}
_judge = None  # (model, tokenizer, device, eos_ids), or False once loading has failed
_step_time_s = STEP_TIME_GUESS_S
stats = {"batches": 0, "judged": 0, "prior": 0, "load_s": 0.0, "gen_s": 0.0}


def _log(message: str) -> None:
    """One line on stderr per event, so the run log shows whether the judge ran and how long it took."""
    try:
        print(f"[judge {time.monotonic() - _START:7.1f}s] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _time_left() -> float:
    return RUN_LIMIT_S - SAFETY_MARGIN_S - (time.monotonic() - _START)


def _prior(problem: str) -> bool:
    return problem.count("$") <= PRIOR_MAX_DOLLARS


def _normalize(ans: str) -> str:
    a = ans.strip().strip("$").strip().rstrip(".")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace("\\!", "").replace("\\,", "").replace(" ", "")
    a = re.sub(r"^\\?(dfrac|tfrac)", r"\\frac", a)
    try:
        v = float(a.replace(",", ""))
        return str(int(v)) if v.is_integer() else str(v)
    except (ValueError, OverflowError):
        return a


def _extract_answer(text: str) -> str | None:
    idx = text.rfind(BOXED)
    if idx < 0:
        return None
    depth, out = 1, []
    for ch in text[idx + len(BOXED):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _normalize("".join(out))
        out.append(ch)
    return None


def _load_judge():
    global _judge
    if _judge is None:
        started = time.monotonic()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            tokenizer = AutoTokenizer.from_pretrained(JUDGE_ID, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                JUDGE_ID, dtype="auto", device_map="auto", local_files_only=True
            ).eval()
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            _judge = (model, tokenizer, model.get_input_embeddings().weight.device, eos_ids)
        except Exception as exc:
            _judge = False
            _log(f"judge failed to load: {type(exc).__name__}: {str(exc)[:200]}")
        stats["load_s"] = time.monotonic() - started
        _log(f"judge load took {stats['load_s']:.0f}s, loaded={bool(_judge)}")
    return _judge


def _generate(prompts: list[str], seed: int, time_limit_s: float) -> list[tuple[str | None, int]]:
    """One padded batch. Returns (answer, tokens) per sequence; the answer is None if it did not finish."""
    global _step_time_s
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    model, tokenizer, device, eos_ids = _judge

    class _Deadline(StoppingCriteria):
        def __init__(self, deadline: float):
            self.deadline = deadline

        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() >= self.deadline

    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
            stopping_criteria=StoppingCriteriaList([_Deadline(started + time_limit_s)]),
        )
    new = gen[:, enc["input_ids"].shape[1]:]
    elapsed = time.monotonic() - started
    stats["gen_s"] += elapsed
    stats["batches"] += 1
    if new.shape[1]:
        _step_time_s = elapsed / new.shape[1]
    answers: list[tuple[str | None, int]] = []
    for row in new.tolist():
        end = next((k for k, t in enumerate(row) if t in eos_ids), None)
        if end is None:  # still writing when the batch stopped
            answers.append((None, MAX_NEW_TOKENS))
            continue
        answers.append((_extract_answer(tokenizer.decode(row[:end], skip_special_tokens=True)), end))
    return answers


def _verdict(samples: list[tuple[str | None, int]]) -> bool:
    distinct = len({a for a, _ in samples if a is not None})
    lengths = sorted(n for _, n in samples)
    mid = len(lengths) // 2
    median = lengths[mid] if len(lengths) % 2 else 0.5 * (lengths[mid - 1] + lengths[mid])
    return 1 <= distinct <= MAX_DISTINCT and median < MAX_MEDIAN_TOKENS


def _judge_problems(problems: list[str]) -> None:
    """Fill `_verdicts` for as many of `problems` as the remaining time allows."""
    import torch

    tokenizer = _judge[1]
    prompts = {
        p: tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{p}\n\n{INSTRUCTION}"}], tokenize=False,
            add_generation_prompt=True, reasoning_effort=JUDGE_EFFORT,
        )
        for p in problems
    }
    # Every sample of a problem goes in the same batch, and similar lengths are grouped.
    order = sorted(problems, key=lambda p: (len(prompts[p]), p))
    per_batch = max(1, BATCH_SIZE // SAMPLES)
    pos = 0
    while pos < len(order):
        budget = _time_left()
        if budget < min(_step_time_s * MAX_NEW_TOKENS, 240.0):
            _log(f"out of time with {len(order) - pos} problems unjudged ({budget:.0f}s left)")
            return
        chunk = order[pos:pos + per_batch]
        batch = [prompts[p] for p in chunk for _ in range(SAMPLES)]
        try:
            answers = _generate(batch, SEED + len(_verdicts), time_limit_s=budget)
        except Exception as exc:
            torch.cuda.empty_cache()
            _log(f"batch of {len(batch)} failed: {type(exc).__name__}: {str(exc)[:200]}")
            if per_batch == 1:
                return
            per_batch = max(1, per_batch // 2)
            continue
        for i, p in enumerate(chunk):
            _verdicts[p] = _verdict(answers[i * SAMPLES:(i + 1) * SAMPLES])
        pos += len(chunk)
        _log(f"batch {stats['batches']}: {len(chunk)} problems, {_step_time_s * 1000:.0f} ms/step, "
             f"{sum(_verdicts[p] for p in chunk)} robust")


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict one boolean per problem, in the order given.

    Args:
        model_id: checkpoint the harness wants judged. The verdict does not depend on it.
        reasoning_effort: decoding condition of that checkpoint. The verdict does not depend on it.
        problems: problem statements for this checkpoint and effort.

    Returns:
        One native bool per problem. Never raises: any failure degrades to the text prior.
    """
    if not problems:
        return []
    try:
        todo = list(dict.fromkeys(p for p in problems if p not in _verdicts))
        if todo and _time_left() > 0 and _load_judge():
            _judge_problems(todo)
    except Exception as exc:
        _log(f"unexpected {type(exc).__name__}: {str(exc)[:200]}")
    out = []
    for p in problems:
        if p in _verdicts:
            stats["judged"] += 1
            out.append(bool(_verdicts[p]))
        else:
            stats["prior"] += 1
            out.append(bool(_prior(p)))
    _log(f"{model_id} {reasoning_effort}: {len(problems)} problems, {sum(out)} robust, "
         f"totals judged={stats['judged']} prior={stats['prior']} batches={stats['batches']}")
    return out
