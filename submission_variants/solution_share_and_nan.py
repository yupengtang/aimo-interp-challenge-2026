"""Codabench entry point: robustness verdicts from two things a strong model shows about a problem.

Each problem is judged once, by gpt-oss-120b at low reasoning effort with the organizers' solve
prompt, and the verdict is reused for every checkpoint and effort that is asked about it.

  1. Agreement: the judge solves the problem sixteen times; the share of the most common answer says
     how settled it is.
  2. Contradiction: the judge writes a few variants of the problem with one inserted condition that
     makes it unsolvable, then solves those; the share of NaN answers says how readily a contradiction
     in this problem is noticed.

RULE picks how the two are combined (see `_verdict`). The harness calls `are_robust` once per
checkpoint and effort and limits the whole run to 3600 s. The judge is loaded once and kept, verdicts
are cached by problem text, and a batch is only started when it can finish in the time left. A
problem whose variants could not be solved in time is judged on agreement alone, and one that could
not be judged at all falls back to a prior from its text. The function never raises.
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
SOLVE_EFFORT = "low"
WRITE_EFFORT = "medium"
SAMPLES = 16
MAX_NEW_TOKENS = 2560
VARIANTS = 4
VARIANT_SAMPLES = 4
VARIANT_MAX_NEW = 1536
BATCH_SIZE = 128
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 20261101
STEP_TIME_GUESS_S = 0.1
# A problem is robust when at least this share of its samples give the same answer. A sample that
# does not finish in time agrees with nothing.
MIN_MODAL_SHARE = 0.8
# Share of NaN answers on the no-solution variants that counts as noticing the contradiction.
MIN_NAN_RATE = 0.2
# "and": settled and notices contradictions; "nan": notices contradictions; "share": settled.
RULE = "and"
# Variants are only written and solved while at least this much time is left, so that problems
# arriving in later calls can still be judged on agreement.
VARIANT_RESERVE_S = 1200.0
# Prior used when the judge is unavailable: short statements with little LaTeX lean robust.
PRIOR_MAX_DOLLARS = 14

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
FINAL = "<|channel|>final<|message|>"
BOXED = chr(92) + "boxed{"

_verdicts: dict[str, bool] = {}
_share: dict[str, float] = {}           # problem -> modal answer share on the original
_variants: dict[str, list[str]] = {}    # problem -> usable no-solution variants, until they are solved
_nan_rate: dict[str, float] = {}        # problem -> share of NaN answers on its variants
_judge = None  # (model, tokenizer, device, eos_ids), or False once loading has failed
_step_time_s = STEP_TIME_GUESS_S
stats = {"batches": 0, "full": 0, "share_only": 0, "prior": 0, "load_s": 0.0, "gen_s": 0.0}


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


def _verdict(problem: str) -> bool:
    settled = _share[problem] >= MIN_MODAL_SHARE - 1e-9
    if problem not in _nan_rate or RULE == "share":
        return settled
    notices = _nan_rate[problem] >= MIN_NAN_RATE - 1e-9
    return notices if RULE == "nan" else (settled and notices)


def _judge_problems(problems: list[str]) -> None:
    """Fill `_verdicts` for as many of `problems` as the remaining time allows."""
    order = sorted(problems, key=lambda p: (len(p), p))
    per_call = max(1, BATCH_SIZE // (SAMPLES + VARIANTS))
    for start in range(0, len(order), per_call):
        chunk = order[start:start + per_call]
        with_variants = RULE != "share" and _time_left() > VARIANT_RESERVE_S

        # 1. Solve the original and, in the same batch, write the variants.
        solve = [p for p in chunk if p not in _share]
        write = [p for p in chunk if p not in _variants and p not in _nan_rate] if with_variants else []
        if solve or write:
            jobs = [("solve", p) for p in solve for _ in range(SAMPLES)]
            jobs += [("write", p) for p in write for _ in range(VARIANTS)]
            prompts = [_chat(SOLVE, p, SOLVE_EFFORT) if kind == "solve" else _chat(REWRITE, p, WRITE_EFFORT)
                       for kind, p in jobs]
            texts = _generate(prompts, MAX_NEW_TOKENS if solve else VARIANT_MAX_NEW, SEED + len(_share) + 1000 * len(_nan_rate))
            if texts is None:
                return
            for p in solve:
                answers = [_extract_answer(t) for (kind, q), t in zip(jobs, texts) if kind == "solve" and q == p]
                counts: dict[str, int] = {}
                for a in answers:
                    if a is not None:
                        counts[a] = counts.get(a, 0) + 1
                _share[p] = max(counts.values()) / SAMPLES if counts else 0.0
                _verdicts[p] = _verdict(p)
            for p in write:
                got = [t for (kind, q), t in zip(jobs, texts) if kind == "write" and q == p]
                _variants[p] = [t for t in got if t and t.strip() != p.strip() and len(t) < 3 * len(p) + 400]

        # 2. Solve the variants.
        need = [p for p in chunk if p in _share and _variants.get(p)]
        if need and _time_left() > VARIANT_RESERVE_S:
            jobs = [(p, v) for p in need for v in _variants[p] for _ in range(VARIANT_SAMPLES)]
            texts = _generate([_chat(SOLVE, v, SOLVE_EFFORT) for _, v in jobs], VARIANT_MAX_NEW,
                              SEED + 20_000 + len(_nan_rate))
            if texts is None:
                return
            for p in need:
                mine = [_extract_answer(t) for (q, _), t in zip(jobs, texts) if q == p]
                _nan_rate[p] = sum(a == "NaN" for a in mine) / len(mine)
                _variants.pop(p, None)
                _verdicts[p] = _verdict(p)


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
        todo = list(dict.fromkeys(p for p in problems if p not in _share or (
            RULE != "share" and p not in _nan_rate and _variants.get(p) != [])))
        if todo and _time_left() > 0 and _load_judge():
            _judge_problems(todo)
    except Exception as exc:
        _log(f"unexpected {type(exc).__name__}: {str(exc)[:200]}")
    out = []
    for p in problems:
        if p in _verdicts:
            stats["full" if (p in _nan_rate or RULE == "share") else "share_only"] += 1
            out.append(bool(_verdicts[p]))
        else:
            stats["prior"] += 1
            out.append(bool(_prior(p)))
    _log(f"{model_id} {reasoning_effort}: {len(problems)} problems, {sum(out)} robust, "
         f"full={stats['full']} share_only={stats['share_only']} prior={stats['prior']} batches={stats['batches']}")
    return out
