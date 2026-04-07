"""
Recreate Figure 2(a) from "Beyond the 80/20 Rule" paper.

Generates responses for MATH-500 questions using Qwen3-8B,
computes per-token entropy, and plots the entropy distribution
on a log-scale Y-axis.

Usage:
    python compute_entropy.py                   # default: 50 questions, 4 responses each
    python compute_entropy.py --n_questions 500 --n_responses 16
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Data ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n"
)


def load_math500(n: int | None = None):
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


# ── Entropy computation ─────────────────────────────────────────────────

def compute_token_entropies(logits: torch.Tensor) -> np.ndarray:
    """
    Given logits of shape (seq_len, vocab_size), compute per-token entropy.
    H_t = -sum_j p_{t,j} * log(p_{t,j})
    """
    # Use float32 for numerical stability
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)      # (seq_len, V)
    probs = torch.exp(log_probs)                        # (seq_len, V)
    entropy = -(probs * log_probs).sum(dim=-1)          # (seq_len,)
    return entropy.cpu().numpy()


# ── Generation with entropy collection ──────────────────────────────────

@torch.no_grad()
def generate_and_collect_entropy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 1.0,
):
    """
    Generate a response token-by-token, collecting the entropy of the
    output distribution at each step.

    Returns:
        response_text: str
        entropies: np.ndarray of shape (n_generated_tokens,)
    """
    # Prepare the prompt using the chat template (thinking mode)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    # We use model.generate with output_scores=True to get logits at each step
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=0,           # no top-k filtering — full distribution
        top_p=1.0,         # no nucleus filtering — full distribution
        output_scores=True,
        return_dict_in_generate=True,
    )

    # outputs.scores is a tuple of (n_generated_tokens,) tensors, each (batch=1, vocab)
    # These are the logits BEFORE temperature scaling by default,
    # but with temperature=T in generate(), the scores are already divided by T.
    # We compute entropy from these scores directly.
    scores = torch.stack(outputs.scores, dim=0).squeeze(1)  # (n_tokens, vocab)

    entropies = compute_token_entropies(scores)

    # Decode only the generated part
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response_text, entropies


# ── Plotting ────────────────────────────────────────────────────────────

def plot_entropy_distribution(all_entropies: np.ndarray, save_path: str):
    """
    Recreate Figure 2(a): histogram of token entropies with log-scale Y-axis.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Compute the 80th percentile (paper reports 0.672)
    p80 = np.percentile(all_entropies, 80)

    # Histogram with fine bins
    bins = np.linspace(0, max(all_entropies.max(), 5.0), 200)
    ax.hist(all_entropies, bins=bins, color="#4a90d9", edgecolor="none", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Entropy", fontsize=14)
    ax.set_ylabel("Frequency (log scale)", fontsize=14)
    ax.set_title("(a) Distribution of token entropy", fontsize=15, fontweight="bold")

    # Mark the 80th percentile
    ax.axvline(p80, color="red", linestyle="--", linewidth=1.5)
    ax.text(
        p80 + 0.08, ax.get_ylim()[1] * 0.3,
        f"The 80th percentile: {p80:.3f}",
        color="red", fontsize=11,
    )

    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute token entropy distribution (Figure 2a)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--n_questions", type=int, default=50,
                        help="Number of MATH-500 questions to use")
    parser.add_argument("--n_responses", type=int, default=4,
                        help="Number of responses to generate per question")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="entropy_results")
    parser.add_argument("--plot_only", type=str, default=None,
                        help="Path to existing .npy file — skip generation, just plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    entropy_path = os.path.join(args.output_dir, "all_entropies.npy")
    plot_path = os.path.join(args.output_dir, "entropy_distribution.png")
    stats_path = os.path.join(args.output_dir, "stats.json")

    # ── Plot-only mode ──
    if args.plot_only:
        print(f"Loading entropies from {args.plot_only}")
        all_entropies = np.load(args.plot_only)
        print(f"Total tokens: {len(all_entropies):,}")
        plot_entropy_distribution(all_entropies, plot_path)
        return

    # ── Load model ──
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    # ── Load data ──
    data = load_math500(n=args.n_questions)
    print(f"Loaded {len(data)} questions")

    # ── Generate and collect entropies ──
    all_entropies_list = []
    total_tokens = 0
    t0 = time.time()

    for q_idx, example in enumerate(data):
        for r_idx in range(args.n_responses):
            response_text, entropies = generate_and_collect_entropy(
                model, tokenizer,
                example["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            all_entropies_list.append(entropies)
            total_tokens += len(entropies)

            elapsed = time.time() - t0
            print(
                f"[Q {q_idx+1}/{len(data)}, R {r_idx+1}/{args.n_responses}] "
                f"tokens={len(entropies):,}  total={total_tokens:,}  "
                f"elapsed={elapsed:.0f}s"
            )

        # Save periodically (every 10 questions)
        if (q_idx + 1) % 10 == 0:
            merged = np.concatenate(all_entropies_list)
            np.save(entropy_path, merged)
            print(f"  [checkpoint] saved {len(merged):,} entropies to {entropy_path}")

    # ── Save final results ──
    all_entropies = np.concatenate(all_entropies_list)
    np.save(entropy_path, all_entropies)
    print(f"\nSaved {len(all_entropies):,} token entropies to {entropy_path}")

    # ── Compute and save statistics ──
    stats = {
        "total_tokens": int(len(all_entropies)),
        "mean_entropy": float(all_entropies.mean()),
        "median_entropy": float(np.median(all_entropies)),
        "p80_entropy": float(np.percentile(all_entropies, 80)),
        "p90_entropy": float(np.percentile(all_entropies, 90)),
        "p95_entropy": float(np.percentile(all_entropies, 95)),
        "frac_below_0.01": float((all_entropies < 0.01).mean()),
        "frac_above_p80": float((all_entropies >= np.percentile(all_entropies, 80)).mean()),
        "n_questions": args.n_questions,
        "n_responses": args.n_responses,
        "temperature": args.temperature,
        "model": args.model,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    # ── Plot ──
    plot_entropy_distribution(all_entropies, plot_path)


if __name__ == "__main__":
    main()
