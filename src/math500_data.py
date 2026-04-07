"""
MATH-500 loader.

Returns a list of dicts:
    {"prompt": str, "gold": str}

Run directly to download + smoke-test:
    python src/data.py
"""

from __future__ import annotations

import random

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n"
)


def load_math500(n: int | None = None):
    """Load HuggingFaceH4/MATH-500 (500 test problems)."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for ex in ds:
        out.append({
            "prompt": PROMPT_TEMPLATE.format(question=ex["problem"]),
            "gold": ex["answer"].strip(),
        })
        if n is not None and len(out) >= n:
            break
    return out


def load_math500_split(n_test: int = 100, seed: int = 42):
    """
    Deterministic train/test split of MATH-500.
    Returns (train, test). Default: 400 train / 100 test.
    """
    full = load_math500()
    rng = random.Random(seed)
    indices = list(range(len(full)))
    rng.shuffle(indices)
    test_idx = set(indices[:n_test])
    train = [full[i] for i in indices[n_test:]]
    test = [full[i] for i in indices[:n_test]]
    return train, test


if __name__ == "__main__":
    print("Loading MATH-500 (first 5)...")
    for ex in load_math500(n=5):
        print(f"  gold={ex['gold']!r}")
        print(f"  prompt={ex['prompt'][:100]!r}")
        print()

    full = load_math500()
    print(f"MATH-500 total size: {len(full)}")

    train, test = load_math500_split()
    print(f"Train size: {len(train)}  Test size: {len(test)}")
    assert len(train) + len(test) == len(full)
    train_prompts = {ex["prompt"] for ex in train}
    test_prompts = {ex["prompt"] for ex in test}
    assert train_prompts.isdisjoint(test_prompts), "train/test overlap!"
