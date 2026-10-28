"""CPU test of bundle_v6's control flow with a fake judge (no model)."""
import importlib
import sys

sys.path.insert(0, "submission_variants")


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, reasoning_effort=None):
        kind = "WRITE" if messages[0]["content"].startswith("You edit") else "SOLVE"
        return f"{kind}|{reasoning_effort}|{messages[1]['content']}"


def fresh(rule):
    import solution_share_and_nan as solution
    s = importlib.reload(solution)
    s.RULE = rule
    s._judge = (None, FakeTok(), None, set())
    calls = []

    def fake_generate(prompts, max_new, seed):
        calls.append((len(prompts), max_new))
        out = []
        for i, p in enumerate(prompts):
            kind, effort, text = p.split("|", 2)
            if kind == "WRITE":
                assert effort == "medium" and max_new in (s.MAX_NEW_TOKENS, s.VARIANT_MAX_NEW)
                out.append(text + " [contradiction]" if "nowrite" not in text else text)  # unusable when unchanged
            else:
                assert effort == "low"
                if "[contradiction]" in text:
                    assert max_new == s.VARIANT_MAX_NEW
                    out.append("ANSWER: NaN" if "notices" in text else "ANSWER: 7")
                elif "settled" in text:
                    out.append("ANSWER: 42")
                else:
                    out.append(f"ANSWER: {i % 5}")
        return out

    s._generate = fake_generate
    return s, calls


probs = {
    "settled notices": True, "settled blind": False, "mixed notices": False, "mixed blind": False,
    "settled nowrite": True,  # no usable variant: agreement alone
}
s, calls = fresh("and")
names = list(probs)
out1 = s.are_robust("m1", "default", names[:3])
out2 = s.are_robust("m2", "low", names)          # two new problems, three cached
out3 = s.are_robust("m3", "default", names)      # nothing new
assert out1 == [probs[n] for n in names[:3]], out1
assert out2 == out3 == [probs[n] for n in names], (out2, out3)
assert len(calls) == 4, calls                     # (solve+write, variants) for each of the two calls with new problems
assert calls[0] == (3 * 20, s.MAX_NEW_TOKENS) and calls[1] == (3 * 16, s.VARIANT_MAX_NEW), calls
assert calls[2] == (2 * 20, s.MAX_NEW_TOKENS) and calls[3] == (1 * 16, s.VARIANT_MAX_NEW), calls
print("and  :", out2, s.stats)

s, calls = fresh("nan")
assert s.are_robust("m", "default", names) == [True, False, True, False, True]
print("nan  :", s.stats)

s, calls = fresh("share")
assert s.are_robust("m", "default", names) == [True, True, False, False, True]
assert all(n % 16 == 0 and m == s.MAX_NEW_TOKENS for n, m in calls), calls   # no variants written or solved
print("share:", s.stats, calls)

# more problems than fit in one batch: chunks of BATCH_SIZE // 20
s, calls = fresh("and")
many = [f"settled notices {i}" for i in range(14)]
assert s.are_robust("m", "default", many) == [True] * 14
assert [c[0] for c in calls] == [120, 96, 120, 96, 40, 32], calls
print("chunks:", calls)

# out of time before the variants: agreement only, and a later call does not redo the originals
s, calls = fresh("and")
s._time_left = lambda: 1000.0
assert s.are_robust("m", "default", ["settled blind"]) == [True]
assert calls == [(16, s.MAX_NEW_TOKENS)], calls
assert s.are_robust("m2", "default", ["settled blind"]) == [True] and len(calls) == 1
print("low time:", s.stats)

# generation gives up (returns None): prior
s, calls = fresh("and")
s._generate = lambda prompts, max_new, seed: None
assert s.are_robust("m", "default", ["a $x$", "$".join("y" * 40)]) == [True, False]
print("no generation:", s.stats)
print("ALL OK")
