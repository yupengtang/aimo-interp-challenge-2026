"""Build problem_id -> ground-truth answer for the officially labelled MATH problems.

Accuracy decay under our own perturbations needs a correct answer to score against, and the official
parquets carry only the problem text. Answers come from the hendrycks MATH copy by exact text match
after whitespace and case normalisation, which a diagnostic confirmed matches all 164, then from the
final boxed value of the worked solution.

The marker is assembled with chr(92) rather than written as an escape: earlier versions of this script
were transmitted through a shell heredoc that collapsed the escape into a backspace character, so the
marker silently never matched and every extraction failed. A self-test below fails loudly instead.
"""
import glob
import json
import os
import re

import pandas as pd

B = os.environ.get("AIMO_BASE", "/path/to/aimo")
OUT = f"{B}/data_derived/answers_math.json"
MARKER = chr(92) + "boxed{"

assert len(MARKER) == 7 and ord(MARKER[0]) == 92, f"marker is mangled: {MARKER!r}"
_probe = "weighs $" + chr(92) + "boxed{3}$ treeks"
assert MARKER in _probe, "self-test failed: marker does not match a known-good sample"
print(f"marker ok: {MARKER!r}")


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def boxed(text: str):
    i = text.rfind(MARKER)
    if i < 0:
        return None
    depth, out = 1, []
    for ch in text[i + len(MARKER):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip() if depth == 0 else None


src = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{B}/data/*.parquet")])
labelled = src[src["dataset_id"].astype(str).str.startswith("hendrycks-math")].drop_duplicates("problem_id")
print(f"stage 0: officially labelled MATH problems = {len(labelled)}")

best = {}
for path in sorted(glob.glob(f"{B}/unlabeled/hendrycks_math/**/*.parquet", recursive=True)):
    d = pd.read_parquet(path)
    if not {"problem", "solution"} <= set(d.columns):
        continue
    for problem, solution in zip(d["problem"], d["solution"]):
        best.setdefault(norm(problem), solution)
worked = sum(1 for v in best.values() if MARKER in str(v))
print(f"stage 1: distinct problems = {len(best)}, containing a boxed value = {worked}")

matched = extracted = 0
answers, unextracted = {}, []
for pid, text in zip(labelled["problem_id"], labelled["original_problem"]):
    solution = best.get(norm(text))
    if solution is None:
        continue
    matched += 1
    ans = boxed(str(solution))
    if ans is None:
        unextracted.append(pid)
        continue
    extracted += 1
    answers[str(pid)] = ans
print(f"stage 2: text matched = {matched}")
print(f"stage 3: boxed answer extracted = {extracted}")
if unextracted:
    print(f"  matched but no boxed value (first 5): {unextracted[:5]}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(answers, f, indent=1, sort_keys=True)
print(f"wrote {len(answers)} answers to {OUT}")
print("examples:", list(answers.items())[:10])
