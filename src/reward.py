"""
Reward function for DAPO training on MATH-500.

Score = 1.0 if the last \\boxed{...} in the completion matches gold,
else 0.0. We try string normalization first, then math_verify symbolic
equivalence as a fallback.
"""

from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _math_equal(pred: str, gold: str) -> bool:
    if _normalize(pred) == _normalize(gold):
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        return False


def correctness_reward(completion: str, gold: str) -> float:
    pred = extract_boxed(completion)
    if pred is None:
        return 0.0
    return 1.0 if _math_equal(pred, gold) else 0.0


def make_reward_fn():
    """trl reward-fn signature: fn(prompts, completions, **kwargs) -> list[float]"""
    def fn(prompts, completions, **kwargs):
        golds = kwargs["gold"]
        return [correctness_reward(c, g) for c, g in zip(completions, golds)]
    return fn
