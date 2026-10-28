"""Answer-format perturbations for self-labelling, outside the organizers' seven types.

The organizers confirmed on 2026-09-16 that the website governs, and it forbids extra labelled data
only for the same perturbation types as their validation set: rephrase, rename, domain, distract,
typos, expert perturbations, expert no-solution. Every family here changes only what the model is
asked to report, never the problem, so it is none of those.

Two properties make this family worth trying after the previous attempt failed:

  * **No text surgery.** An instruction is appended; the problem body is untouched. The previous
    module mangled LaTeX because it edited inside the statement, and its self-check was circular
    because it used the same broken parser to verify itself. Neither failure can occur here.
  * **Exact ground truth.** The transformed answer is computed from the known answer by a closed
    form, so no annotation is needed and correctness is checkable by string comparison.

Coverage is the open question, and it is reported rather than assumed: these transforms need an
integer answer. That is near-universal for AIME-style problems, whose answers are integers by
construction, and partial on MATH.
"""

from __future__ import annotations

import re

FAMILIES = ("modulo", "digit_sum", "scaled")


def parse_integer_answer(answer: str) -> int | None:
    """Return the answer as an int, or None when it is not a plain integer.

    Rejects fractions, radicals, symbolic forms and decimals rather than guessing at them, because a
    wrong ground truth would silently poison every label built from it.
    """
    if answer is None:
        return None
    text = str(answer).strip()
    text = text.replace("$", "").replace(",", "").replace("\\!", "").replace(" ", "")
    text = re.sub(r"^\\text\{(.*)\}$", r"\1", text)
    # Some sources store whole-number answers as floats ("142.0"); those are integers written badly.
    # A genuine fraction ("142.5") or a symbolic answer is still rejected.
    float_form = re.fullmatch(r"([+-]?\d+)\.0+", text)
    if float_form:
        text = float_form.group(1)
    if not re.fullmatch(r"[+-]?\d+", text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def build_variants(problem: str, answer: str) -> list[tuple[str, str, str]]:
    """Return (family, perturbed_problem, expected_answer) for each applicable transform."""
    value = parse_integer_answer(answer)
    if value is None:
        return []

    # Modulus 97 rather than 1000: AIME-style answers are already in 0..999 by construction, so
    # "modulo 1000" is the identity on exactly the problems this family is meant to cover, and every
    # such variant would be discarded as testing nothing.
    variants: list[tuple[str, str, int]] = [
        ("modulo", "Report your final answer modulo 97.", value % 97),
        ("scaled", "Report twice your final answer.", 2 * value),
    ]
    if value >= 0:
        variants.append(
            ("digit_sum", "After solving, report the sum of the decimal digits of your final answer.",
             sum(int(d) for d in str(value)))
        )

    out = []
    for family, instruction, expected in variants:
        # A transform that leaves the answer unchanged tests nothing, so drop it.
        if expected == value:
            continue
        out.append((family, f"{problem}\n\n{instruction}", str(expected)))
    return out


def coverage(answers: dict[str, str]) -> dict[str, float]:
    """How many problems the family can label at all, which is what sank the previous attempt."""
    total = len(answers)
    usable = sum(1 for a in answers.values() if parse_integer_answer(a) is not None)
    per_family = {f: 0 for f in FAMILIES}
    for a in answers.values():
        for family, _, _ in build_variants("x", a):
            per_family[family] += 1
    return {
        "problems": total,
        "integer_answers": usable,
        "integer_fraction": round(usable / total, 3) if total else 0.0,
        **{f"variants_{k}": v for k, v in per_family.items()},
    }
