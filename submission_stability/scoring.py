"""Feature extraction and linear scoring for the stability method.

One forward pass per view of a problem, no generation. Features are how far the model's own state moves
between the original problem and each answer-preserving rewrite: cosine similarity of the last prompt
token's hidden state at four depths, and the divergence of the next-token distribution. Depths are
named by quartile so that one coefficient vector fits checkpoints with different layer counts.

Views are batched across problems, which matters for more than convenience: the competition breaks
ties by average runtime, and one padded batch of sixteen short prompts costs about what a single
prompt costs once kernel launch overhead dominates. `problem_features` keeps the unbatched path for
reference and for the case where a batch runs out of memory.

Scoring is a dot product, so inference needs numpy only; scikit-learn is not imported here and the
artifact carries plain arrays.
"""

from __future__ import annotations

import numpy as np
import torch

from perturbations import FAMILY_OF, build_variants

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DEPTHS = ("q1", "q2", "q3", "q4")


def chat_prompt(tokenizer, checkpoint: str, text: str) -> str:
    kwargs = {"reasoning_effort": "low"} if "gpt-oss" in checkpoint else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{text}\n\n{INSTRUCTION}"}],
        tokenize=False, add_generation_prompt=True, **kwargs,
    )


def quartile_layers(model) -> list[int]:
    n = model.config.get_text_config().num_hidden_layers
    return sorted({n // 4, n // 2, (3 * n) // 4, n})


def _aggregate(gathered: dict[str, list[float]], base_len: int) -> dict[str, float]:
    feats = {"prompt_tokens": float(base_len)}
    for key, values in gathered.items():
        feats[f"{key}_mean"] = float(np.mean(values))
        feats[f"{key}_min"] = float(np.min(values))
    return feats


@torch.no_grad()
def encode_view(model, tokenizer, checkpoint: str, text: str, layers: list[int], device):
    prompt = chat_prompt(tokenizer, checkpoint, text)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model(input_ids=ids, output_hidden_states=True)
    hidden = {l: out.hidden_states[l][0, -1].float() for l in layers}
    logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return hidden, logprobs, int(ids.shape[1])


def problem_features(model, tokenizer, checkpoint: str, problem: str, problem_id: str,
                     layers: list[int], device, families: tuple[str, ...]) -> dict[str, float]:
    """Unbatched reference path: one problem, one forward pass per view."""
    variants = [v for v in build_variants(problem, problem_id)
                if v[0] == "orig" or FAMILY_OF[v[0]] in families]
    base_hidden, base_lp, base_len = encode_view(model, tokenizer, checkpoint, variants[0][1], layers, device)
    gathered: dict[str, list[float]] = {}
    for name, text in variants[1:]:
        hidden, lp, _ = encode_view(model, tokenizer, checkpoint, text, layers, device)
        family = FAMILY_OF[name]
        for depth, l in zip(DEPTHS, layers):
            cos = float(torch.nn.functional.cosine_similarity(base_hidden[l], hidden[l], dim=0))
            gathered.setdefault(f"cos_{depth}_{family}", []).append(cos)
        gathered.setdefault(f"kl_{family}", []).append(float((base_lp.exp() * (base_lp - lp)).sum()))
    return _aggregate(gathered, base_len)


@torch.no_grad()
def _encode_batch(model, tokenizer, prompts: list[str], layers: list[int], device):
    """Run one padded batch and return, per sequence, the final-token states and next-token logprobs.

    Padding is on the left, so the final real token is at position -1 for every sequence and no
    gather over variable lengths is needed.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model(**enc, output_hidden_states=True)
    hidden = {l: out.hidden_states[l][:, -1, :].float() for l in layers}
    logprobs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    lengths = enc["attention_mask"].sum(dim=1).tolist()
    return hidden, logprobs, [int(n) for n in lengths]


def batch_features(model, tokenizer, checkpoint: str, problems: list[str], layers: list[int],
                   device, families: tuple[str, ...], batch_size: int = 16,
                   ids: list[str] | None = None) -> list[dict[str, float]]:
    """Features for several problems, batching every view of every problem together.

    Returns one feature dict per input problem, in order. A batch that raises (most likely out of
    memory) is retried at half the size, down to one sequence, so a long problem cannot lose the run.
    `ids` seed the rewrites, so pass stable ids to keep a problem's variants identical no matter how
    the problems were split into chunks.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    jobs: list[tuple[int, str, str]] = []
    for index, problem in enumerate(problems):
        seed_id = ids[index] if ids is not None else f"case_{index}"
        for name, text in build_variants(problem, seed_id):
            if name == "orig" or FAMILY_OF[name] in families:
                jobs.append((index, name, chat_prompt(tokenizer, checkpoint, text)))
    # Group similar lengths together so padding waste stays small.
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][2]))

    states: dict[tuple[int, str], tuple[dict[int, torch.Tensor], torch.Tensor, int]] = {}
    pos = 0
    while pos < len(order):
        size = min(batch_size, len(order) - pos)
        while True:
            chunk = order[pos:pos + size]
            try:
                hidden, logprobs, lengths = _encode_batch(
                    model, tokenizer, [jobs[i][2] for i in chunk], layers, device
                )
                break
            except Exception:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
        for row, i in enumerate(chunk):
            index, name, _ = jobs[i]
            states[(index, name)] = ({l: hidden[l][row] for l in layers}, logprobs[row], lengths[row])
        pos += len(chunk)

    out: list[dict[str, float]] = []
    for index in range(len(problems)):
        base_hidden, base_lp, base_len = states[(index, "orig")]
        gathered: dict[str, list[float]] = {}
        for (other_index, name), (hidden, lp, _) in states.items():
            if other_index != index or name == "orig":
                continue
            family = FAMILY_OF[name]
            for depth, l in zip(DEPTHS, layers):
                cos = float(torch.nn.functional.cosine_similarity(base_hidden[l], hidden[l], dim=0))
                gathered.setdefault(f"cos_{depth}_{family}", []).append(cos)
            gathered.setdefault(f"kl_{family}", []).append(float((base_lp.exp() * (base_lp - lp)).sum()))
        out.append(_aggregate(gathered, base_len))
    return out


def score_features(feats: dict[str, float], artifact: dict, model_id: str) -> float:
    """Standardise with this checkpoint's own statistics, then apply the linear model."""
    names = artifact["feature_names"]
    x = np.array([feats.get(n, np.nan) for n in names], dtype=float)
    calib = artifact["per_model"].get(model_id)
    if calib is None:  # unseen checkpoint: fall back to the training distribution
        mean, scale = np.asarray(artifact["train_mean"]), np.asarray(artifact["train_scale"])
    else:
        mean, scale = np.asarray(calib["mean"]), np.asarray(calib["scale"])
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    # A missing feature contributes nothing rather than poisoning the whole score.
    z = np.where(np.isfinite(z), z, 0.0)
    return float(np.dot(np.asarray(artifact["coef"], dtype=float), z) + artifact["intercept"])


def threshold_for(artifact: dict, model_id: str) -> float:
    calib = artifact["per_model"].get(model_id)
    if calib is not None:
        return float(calib["threshold"])
    thresholds = [c["threshold"] for c in artifact["per_model"].values()]
    return float(np.median(thresholds)) if thresholds else 0.0
