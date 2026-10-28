"""Wall-clock budget shared by every are_robust call in one evaluation run.

The ingestion program imports solution.py once and then calls are_robust once per model, with no
per-call timeout; the 3600 s limit covers the whole prediction run. The clock therefore starts when
this module is first imported, and each call gets an even share of what is left for the models that
have not been seen yet.
"""

from __future__ import annotations

import time
from pathlib import Path

RUN_LIMIT_S = 3600.0
# Kept back for model teardown, writing predictions and clock skew between the harness and us.
SAFETY_MARGIN_S = 240.0

_START = time.monotonic()
_seen_models: set[str] = set()


def expected_model_count() -> int:
    """Main track evaluates five checkpoints, the small track four (no gpt-oss-120b)."""
    small = any(p.name.casefold() == "small.txt" for p in Path(__file__).resolve().parent.iterdir())
    return 4 if small else 5


def elapsed() -> float:
    return time.monotonic() - _START


def remaining() -> float:
    return RUN_LIMIT_S - SAFETY_MARGIN_S - elapsed()


def start_model(model_id: str) -> float:
    """Register a model call and return the absolute monotonic deadline for it."""
    _seen_models.add(model_id)
    models_left = max(1, expected_model_count() - len(_seen_models) + 1)
    share = max(0.0, remaining()) / models_left
    return time.monotonic() + share


def time_left(deadline: float) -> float:
    return deadline - time.monotonic()
