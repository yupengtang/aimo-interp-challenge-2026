"""Codabench entry point: robustness verdicts from how settled a configuration is on the problem.

Each problem is first judged once: gpt-oss-120b solves it sixteen times at low reasoning effort with
the organizers' own solve prompt. The most common answer is the reference, and the problem is called
settled when at least 80% of the samples give it.

Validation and test are balanced within each problem (every problem has robust and non-robust
configurations), so a verdict that depends only on the problem cannot separate them. For the gpt-oss
configurations the configuration itself can be sampled: a configuration whose own answers nearly all
match the reference keeps its accuracy under perturbation, while one that gets the problem only part of
the time loses some just from sampling. On train-main-v2 the first group is robust 95% of the time and
the rest about half. Low effort reuses the judge's samples, medium and high are sampled at their own
effort. The small checkpoints keep the problem-level verdict.

The harness calls `are_robust` once per checkpoint and effort and limits the whole run to 3600 s. The
judge is loaded once and kept, verdicts are cached by problem text, and a batch is only started when
it can finish in the time left. Anything that cannot be computed in time falls back to the
problem-level verdict, then to a prior from the text. The function never raises.
"""

from __future__ import annotations

import os
import re
import sys
import time

_START = time.monotonic()
RUN_LIMIT_S = float(os.environ.get("AIMO_RUN_LIMIT_S", "3600"))  # override only for offline tests
SAFETY_MARGIN_S = 300.0

JUDGE_ID = "openai/gpt-oss-120b"
SOLVE_EFFORT = "low"
SAMPLES = 16
MAX_NEW_TOKENS = 2560
BATCH_SIZE = 128
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 20261101
STEP_TIME_GUESS_S = 0.1
# A problem is robust when at least this share of its samples give the same answer. A sample that
# does not finish in time agrees with nothing.
MIN_MODAL_SHARE = 0.8
# Prior used when the judge is unavailable: short statements with little LaTeX lean robust.
PRIOR_MAX_DOLLARS = 14
# Own-effort sampling for gpt-oss medium/high: samples per problem, token cap per effort, and the share
# of the configuration's answers that must match the reference for a robust verdict.
OWN_SAMPLES = 8
OWN_MAX_NEW_TOKENS = {"medium": 4096, "high": 6144}
MIN_OWN_MATCH = 7 / 8
# Keep this much time for the calls still to come before starting own-effort sampling.
OWN_RESERVE_S = 600.0

SOLVE = (
    "You are a precise math problem solver. Solve the given problem step by step, then output your "
    "final answer on the last line in the exact format:\nANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)
FINAL = "<|channel|>final<|message|>"
BOXED = chr(92) + "boxed{"

_verdicts: dict[str, bool] = {}
_reference: dict[str, str] = {}  # problem -> modal answer of the judge, when one exists
_judge_answers: dict[str, list[str | None]] = {}
_own: dict[tuple[str, str], bool] = {}  # (problem, effort) -> own-effort verdict
_judge = None  # (model, tokenizer, device, eos_ids), or False once loading has failed
_step_time_s = STEP_TIME_GUESS_S
stats = {"batches": 0, "judged": 0, "prior": 0, "own": 0, "load_s": 0.0, "gen_s": 0.0}


def _log(message: str) -> None:
    try:
        print(f"[judge {time.monotonic() - _START:7.1f}s] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _time_left() -> float:
    return RUN_LIMIT_S - SAFETY_MARGIN_S - (time.monotonic() - _START)


def _prior(problem: str) -> bool:
    return problem.count("$") <= PRIOR_MAX_DOLLARS


def _normalize(ans: str) -> str:
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    a = a.strip().strip("$*` .").replace("\\!", "").replace("\\,", "").replace(" ", "").replace(",", "")
    if a.lower() in ("nan", "nosolution", "none", "undefined", "impossible"):
        return "NaN"
    try:
        v = float(a)
        return str(int(v)) if v.is_integer() else str(v)
    except (ValueError, OverflowError):
        return a


def _extract_answer(text: str | None) -> str | None:
    """The answer after the last `ANSWER:`, else the last boxed expression."""
    if not text:
        return None
    found = re.findall(r"ANSWER:\s*([^\n]+)", text)
    if found:
        return _normalize(found[-1])
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


def _chat(system: str, user: str, effort: str) -> str:
    return _judge[1].apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, reasoning_effort=effort,
    )


def _generate(prompts: list[str], max_new: int, seed: int) -> list[str | None] | None:
    """Decode `prompts` in padded batches. Returns the final-channel text of each sequence (None if
    it did not finish), or None for the whole call when the time left cannot cover it."""
    global _step_time_s
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    model, tokenizer, device, eos_ids = _judge

    class _Deadline(StoppingCriteria):
        def __init__(self, deadline: float):
            self.deadline = deadline

        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() >= self.deadline

    texts: list[str | None] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
    size = BATCH_SIZE
    pos = 0
    while pos < len(order):
        budget = _time_left()
        if budget < min(_step_time_s * max_new, 240.0):
            _log(f"out of time with {len(order) - pos} sequences left ({budget:.0f}s)")
            return None
        idx = order[pos:pos + size]
        try:
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            torch.manual_seed(seed + pos)
            torch.cuda.manual_seed_all(seed + pos)
            started = time.monotonic()
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                    stopping_criteria=StoppingCriteriaList([_Deadline(started + budget)]),
                )
        except Exception as exc:
            torch.cuda.empty_cache()
            _log(f"batch of {len(idx)} failed: {type(exc).__name__}: {str(exc)[:200]}")
            if size == 1:
                return None
            size = max(1, size // 2)
            continue
        new = gen[:, enc["input_ids"].shape[1]:]
        elapsed = time.monotonic() - started
        stats["gen_s"] += elapsed
        stats["batches"] += 1
        if new.shape[1]:
            _step_time_s = elapsed / new.shape[1]
        for row, i in zip(new.tolist(), idx):
            end = next((k for k, t in enumerate(row) if t in eos_ids), None)
            if end is None:
                continue
            text = tokenizer.decode(row[:end], skip_special_tokens=False)
            if FINAL in text:
                text = text.split(FINAL)[-1]
            texts[i] = re.sub(r"<\|[^|]*\|>", "", text).strip()
        _log(f"batch {stats['batches']}: {len(idx)} sequences, {new.shape[1]} steps, {_step_time_s * 1000:.0f} ms/step")
        pos += len(idx)
    return texts


def _judge_problems(problems: list[str]) -> None:
    """Fill `_verdicts` for as many of `problems` as the remaining time allows."""
    order = sorted(problems, key=lambda p: (len(p), p))
    per_call = max(1, BATCH_SIZE // SAMPLES)
    # One generate call per batch of problems, so a verdict is kept even if a later batch runs out of time.
    for start in range(0, len(order), per_call):
        chunk = order[start:start + per_call]
        prompts = [_chat(SOLVE, p, SOLVE_EFFORT) for p in chunk for _ in range(SAMPLES)]
        texts = _generate(prompts, MAX_NEW_TOKENS, SEED + len(_verdicts))
        if texts is None:
            return
        for i, p in enumerate(chunk):
            answers = [_extract_answer(t) for t in texts[i * SAMPLES:(i + 1) * SAMPLES]]
            counts: dict[str, int] = {}
            for a in answers:
                if a is not None:
                    counts[a] = counts.get(a, 0) + 1
            share = max(counts.values()) / SAMPLES if counts else 0.0
            _verdicts[p] = share >= MIN_MODAL_SHARE - 1e-9
            _judge_answers[p] = answers
            if counts:
                _reference[p] = max(counts, key=lambda a: (counts[a], a))


def _own_verdicts(problems: list[str], effort: str) -> None:
    """Fill `_own` for (problem, effort) by sampling the gpt-oss configuration at its own effort.

    Low effort reuses the judge's samples. A problem without a reference answer is left to the
    problem-level verdict."""
    todo = [p for p in dict.fromkeys(problems) if p in _reference and (p, effort) not in _own]
    if not todo:
        return
    if effort == "low":
        for p in todo:
            answers = _judge_answers.get(p) or []
            match = sum(a == _reference[p] for a in answers) / max(len(answers), 1)
            _own[(p, effort)] = match >= MIN_OWN_MATCH - 1e-9
        return
    max_new = OWN_MAX_NEW_TOKENS.get(effort)
    if max_new is None:
        return
    per_call = max(1, BATCH_SIZE // OWN_SAMPLES)
    for start in range(0, len(todo), per_call):
        if _time_left() - OWN_RESERVE_S < _step_time_s * max_new:
            _log(f"no time for own-effort sampling at {effort}; {len(todo) - start} problems keep the problem verdict")
            return
        chunk = todo[start:start + per_call]
        prompts = [_chat(SOLVE, p, effort) for p in chunk for _ in range(OWN_SAMPLES)]
        texts = _generate(prompts, max_new, SEED + 7919 + len(_own))
        if texts is None:
            return
        for i, p in enumerate(chunk):
            answers = [_extract_answer(t) for t in texts[i * OWN_SAMPLES:(i + 1) * OWN_SAMPLES]]
            match = sum(a == _reference[p] for a in answers) / OWN_SAMPLES
            _own[(p, effort)] = match >= MIN_OWN_MATCH - 1e-9


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict one boolean per problem, in the order given.

    Args:
        model_id: checkpoint the harness wants judged. gpt-oss configurations get an own-effort verdict.
        reasoning_effort: decoding condition of that checkpoint ("default", "low", "medium", "high").
        problems: problem statements for this checkpoint and effort.

    Returns:
        One native bool per problem. Never raises: any failure degrades to the problem-level verdict,
        then to the text prior.
    """
    if not problems:
        return []
    own_effort = None
    if str(model_id).startswith("openai/gpt-oss"):
        own_effort = reasoning_effort if reasoning_effort in ("low", "medium", "high") else "medium"
    try:
        todo = list(dict.fromkeys(p for p in problems if p not in _verdicts))
        if todo and _time_left() > 0 and _load_judge():
            _judge_problems(todo)
        if own_effort and _judge:
            _own_verdicts(problems, own_effort)
    except Exception as exc:
        _log(f"unexpected {type(exc).__name__}: {str(exc)[:200]}")
    out = []
    for p in problems:
        if own_effort and (p, own_effort) in _own:
            stats["own"] += 1
            out.append(bool(_own[(p, own_effort)]))
        elif p in _verdicts:
            stats["judged"] += 1
            out.append(bool(_verdicts[p]))
        else:
            stats["prior"] += 1
            out.append(bool(_prior(p)))
    _log(f"{model_id} {reasoning_effort}: {len(problems)} problems, {sum(out)} robust, "
         f"own={stats['own']} judged={stats['judged']} prior={stats['prior']} batches={stats['batches']}")
    return out
