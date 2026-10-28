"""Answer-preserving rewrites of a problem statement.

Every rewrite leaves the mathematics untouched: LaTeX spans and Asymptote blocks are never edited, and
nothing is added that carries a quantity the problem could use. The point is to change the surface and
nothing else, so that any movement in the model's state is a reaction to the surface.
"""

from __future__ import annotations

import random
import re

# Neutral sentences, chosen so none introduces a number, name or unit.
DISTRACTORS = (
    "This problem was shared in a study group that meets on Tuesday evenings.",
    "A student once remarked that the problem looked harder than it really was.",
)


def split_math(text: str) -> list[tuple[bool, str]]:
    """Split into (is_protected, chunk); protected chunks are LaTeX or Asymptote code."""
    parts, pos = [], 0
    pattern = r"\[asy\].*?\[/asy\]|\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)"
    for m in re.finditer(pattern, text, flags=re.S):
        if m.start() > pos:
            parts.append((False, text[pos:m.start()]))
        parts.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        parts.append((False, text[pos:]))
    return parts


def add_typos(text: str, rng: random.Random, n: int = 2) -> str:
    """Transpose two letters inside a few long prose words."""
    parts = split_math(text)
    slots = [
        (i, m.start())
        for i, (protected, chunk) in enumerate(parts)
        if not protected
        for m in re.finditer(r"\b[a-zA-Z]{5,}\b", chunk)
    ]
    for i, start in rng.sample(slots, min(n, len(slots))):
        chunk = parts[i][1]
        j = start + rng.randrange(1, 3)
        parts[i] = (False, chunk[:j] + chunk[j + 1] + chunk[j] + chunk[j + 2:])
    return "".join(chunk for _, chunk in parts)


def add_distractor(text: str, sentence: str) -> str:
    """Insert an irrelevant sentence after the first prose sentence, else at the front."""
    parts = split_math(text)
    for i, (protected, chunk) in enumerate(parts):
        if protected:
            continue
        m = re.search(r"[.!?]\s+", chunk)
        if m:
            parts[i] = (False, chunk[: m.end()] + sentence + " " + chunk[m.end():])
            return "".join(c for _, c in parts)
    return sentence + " " + text


def build_variants(problem: str, problem_id: str) -> list[tuple[str, str]]:
    """The original followed by six rewrites, in a fixed order, deterministic per problem."""
    rng = random.Random(problem_id)
    return [
        ("orig", problem),
        ("typos_a", add_typos(problem, random.Random(rng.random()), n=2)),
        ("typos_b", add_typos(problem, random.Random(rng.random()), n=4)),
        ("distract_a", add_distractor(problem, DISTRACTORS[0])),
        ("distract_b", add_distractor(problem, DISTRACTORS[1])),
        ("wrap_pre", f"Here is a problem to solve.\n\n{problem}"),
        ("wrap_post", f"{problem}\n\nTake your time and be careful."),
    ]


FAMILY_OF = {
    "typos_a": "typos", "typos_b": "typos",
    "distract_a": "distract", "distract_b": "distract",
    "wrap_pre": "wrap", "wrap_post": "wrap",
}
