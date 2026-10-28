"""Codabench entry point: robustness verdicts from gpt-oss-120b's hidden states on the problem.

Robustness on the official train set is mostly a property of the problem: the problems that lose
accuracy under perturbation are the ones a strong model cannot answer consistently. Our black-box
judge measures that by sampling gpt-oss eight times and counting distinct answers. This entry point
reads the same information from inside the model instead: gpt-oss encodes the problem once (no
generation), and eight linear probes on its hidden states (last token and mean, four depths),
trained to predict the judge's answer agreement on an unlabelled pool of olympiad problems,
are averaged into the verdict.

The verdict is computed once per problem and reused for every checkpoint and effort asked about it.
If gpt-oss cannot be loaded or a forward pass fails, that problem gets the constant fallback
(not robust), which is also the "without the white-box component" ablation. The function never raises.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_START = time.monotonic()
RUN_LIMIT_S = 4200.0
SAFETY_MARGIN_S = 300.0

ENCODER_ID = "openai/gpt-oss-120b"
ENCODER_EFFORT = "low"
INSTRUCTION = "Please reason step by step, and put your final answer within " + chr(92) + "boxed{}."
PROBE_PATH = Path(__file__).resolve().parent / "probe.npz"
FALLBACK = False

_verdicts: dict[str, bool] = {}
_encoder = None  # (model, tokenizer, device), or False once loading has failed
_probe = None
stats = {"encoded": 0, "fallback": 0, "load_s": 0.0}


def _log(message: str) -> None:
    try:
        print(f"[probe {time.monotonic() - _START:7.1f}s] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _time_left() -> float:
    return RUN_LIMIT_S - SAFETY_MARGIN_S - (time.monotonic() - _START)


def _load():
    """Load the probe weights and gpt-oss once."""
    global _encoder, _probe
    if _encoder is None:
        started = time.monotonic()
        try:
            import numpy as np
            from transformers import AutoModelForCausalLM, AutoTokenizer

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            with np.load(PROBE_PATH) as p:
                _probe = {k: p[k] for k in p.files}
            tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                ENCODER_ID, dtype="auto", device_map="auto", local_files_only=True
            ).eval()
            _encoder = (model, tokenizer, model.get_input_embeddings().weight.device)
        except Exception as exc:
            _encoder = False
            _log(f"load failed: {type(exc).__name__}: {str(exc)[:200]}")
        stats["load_s"] = time.monotonic() - started
        _log(f"load took {stats['load_s']:.0f}s, loaded={bool(_encoder)}")
    return _encoder


def _features(problem: str) -> dict:
    """Last-token and mean hidden state of the judge prompt at each layer the probes use."""
    import torch

    model, tokenizer, device = _encoder
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}], tokenize=False,
        add_generation_prompt=True, reasoning_effort=ENCODER_EFFORT,
    )
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        states = model(**ids, output_hidden_states=True).hidden_states
    out = {}
    for name in _probe["features"]:
        name = str(name)
        pool = "last" if name.startswith("last") else "mean"
        h = states[int(name[len(pool):])][0].float()
        out[name] = (h[-1] if pool == "last" else h.mean(0)).cpu().numpy()
    return out


def _verdict(problem: str) -> bool:
    """Average of the standardised probe scores, against the threshold fixed on the pool."""
    import numpy as np

    feats = _features(problem)
    scores = []
    for k, name in enumerate(_probe["features"]):
        x = (feats[str(name)] - _probe[f"mean_{k}"]) / _probe[f"scale_{k}"]
        s = float(x @ _probe[f"coef_{k}"] + _probe[f"intercept_{k}"])
        scores.append((s - float(_probe[f"center_{k}"])) / float(_probe[f"spread_{k}"]))
    return bool(np.mean(scores) >= float(_probe["threshold"]))


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict one boolean per problem, in the order given.

    Args:
        model_id: checkpoint the harness wants judged. The verdict does not depend on it.
        reasoning_effort: decoding condition of that checkpoint. The verdict does not depend on it.
        problems: problem statements for this checkpoint and effort.

    Returns:
        One native bool per problem. Never raises: any failure degrades to the constant fallback.
    """
    if not problems:
        return []
    out = []
    for p in problems:
        if p not in _verdicts:
            try:
                if _time_left() > 0 and _load():
                    _verdicts[p] = _verdict(p)
                    stats["encoded"] += 1
            except Exception as exc:
                _log(f"problem failed: {type(exc).__name__}: {str(exc)[:200]}")
        if p in _verdicts:
            out.append(bool(_verdicts[p]))
        else:
            stats["fallback"] += 1
            out.append(FALLBACK)
    _log(f"{model_id} {reasoning_effort}: {len(problems)} problems, {sum(out)} robust, "
         f"encoded={stats['encoded']} fallback={stats['fallback']}")
    return out
