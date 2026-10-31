"""Codabench entry point: robustness verdicts from representation stability.

For each problem the evaluated model is shown the original statement and several rewrites that cannot
change the answer, one forward pass each. A model whose internal state barely moves under those
rewrites is treated as robust on that problem. Nothing is generated, so the whole run stays far inside
the 3600 s budget, and the method needs no labels for the checkpoint it is judging.

Feature extraction runs one sequence at a time on purpose. `scoring.batch_features` is faster, but a
padded batch shifts hidden states slightly, and the artifact was fitted on unpadded single-sequence
features: measured on six problems, the two paths flipped one verdict in six for both Qwen3.5-4B and
DeepSeek. The saving was not worth a mismatch between training and inference, because the unbatched
path already costs about 0.25 s per problem for an 8B model, so a full run is seconds of compute once
the checkpoints are loaded from a warm cache.

The harness calls this once per model and gives the run a single time limit, so the budget is tracked
from import and split across the models still to come. If time runs short the feature set is narrowed
and then abandoned, and if anything fails the batch still returns one valid boolean per problem.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import budget  # noqa: E402  (import starts the clock, so keep it before the heavy imports)
import joblib  # noqa: E402

ARTIFACT_PATH = HERE / "stability_crossmodel.joblib"
# Families in the order they are dropped when time is short; wrap contributes least on MATH.
FAMILY_TIERS = (("typos", "distract", "wrap"), ("typos", "distract"), ("distract",))
# A constant answer scores near the base rate on a balanced set, which is the safe fallback.
FALLBACK = False


def _load_artifact():
    try:
        return joblib.load(ARTIFACT_PATH)
    except Exception:
        return None


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """Predict one boolean per problem, in the order given.

    Args:
        model_id: checkpoint the harness wants judged, exactly as cached on the worker.
        problems: problem statements for this checkpoint.

    Returns:
        One native bool per problem. Never raises: any failure degrades to the fallback verdict.
    """

    if not problems:
        return []
    deadline = budget.start_model(model_id)
    artifact = _load_artifact()
    if artifact is None:
        return [FALLBACK] * len(problems)

    model = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from scoring import problem_features, quartile_layers, score_features, threshold_for

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        device = model.get_input_embeddings().weight.device
        layers = quartile_layers(model)
        threshold = threshold_for(artifact, model_id)

        families = FAMILY_TIERS[0]
        verdicts: list[bool] = []
        started = time.monotonic()
        for index, problem in enumerate(problems):
            remaining = len(problems) - index
            left = budget.time_left(deadline)
            if left <= 0:
                verdicts.extend([FALLBACK] * remaining)
                break
            if index:
                # Narrow the feature set if the measured pace will not finish in time. Each tier drops
                # one perturbation family, so the cost falls roughly in proportion to the views left.
                per_problem = (time.monotonic() - started) / index
                views = len(families)
                for tier in FAMILY_TIERS:
                    if per_problem * remaining <= left:
                        break
                    families = tier
                    per_problem *= len(tier) / views
                    views = len(tier)
            try:
                feats = problem_features(
                    model, tokenizer, model_id, problem, f"case_{index}", layers, device, families
                )
                verdicts.append(bool(score_features(feats, artifact, model_id) > threshold))
            except Exception:
                torch.cuda.empty_cache()
                verdicts.append(FALLBACK)
        verdicts = verdicts[:len(problems)]
        verdicts.extend([FALLBACK] * (len(problems) - len(verdicts)))
        return [bool(v) for v in verdicts]
    except Exception:
        return [FALLBACK] * len(problems)
    finally:
        try:
            if model is not None:
                del model
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
