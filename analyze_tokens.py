"""
Analyze token entropy in LLM chain-of-thought reasoning (Figure 2a/b/c).

GPU-efficient version:
  - Batches multiple responses per question in a single forward pass
  - Uses a LogitsProcessor to compute entropy on-the-fly during generation,
    avoiding storing full (n_steps × vocab_size) score tensors in memory
  - Only scalar entropy values are kept, reducing memory by ~150,000x per token

Usage:
    python analyze_tokens.py generate --n_questions 50 --n_responses 4
    python analyze_tokens.py analyze --min_freq 20 --top_n 100
    python analyze_tokens.py all --n_questions 50 --n_responses 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from datasets import load_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# ── Entropy collector (runs inside generate, no extra memory) ───────────

class EntropyCollector(LogitsProcessor):
    """
    Computes per-token entropy on-the-fly during generation.

    Instead of storing the full (vocab_size,) logit vector for every token
    (which costs ~600 KB per token in float32 for Qwen3's 152k vocab),
    this computes entropy immediately and stores only a single float per
    token — a ~150,000x reduction in memory.
    """

    def __init__(self):
        super().__init__()
        self.entropies: list[torch.Tensor] = []  # each is (batch_size,)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # scores shape: (batch_size, vocab_size)
        # Compute entropy on GPU, move only the scalar(s) to CPU
        with torch.no_grad():
            log_probs = torch.log_softmax(scores.float(), dim=-1)
            probs = torch.exp(log_probs)
            ent = -(probs * log_probs).sum(dim=-1)  # (batch_size,)
            self.entropies.append(ent.cpu())
        return scores  # pass through unchanged

    def get_entropies(self) -> torch.Tensor:
        """Returns (n_steps, batch_size) tensor of entropies."""
        return torch.stack(self.entropies, dim=0)

    def reset(self):
        self.entropies = []


# ── Batched generation ──────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompt: str,
    batch_size: int = 4,
    max_new_tokens: int = 4096,
    temperature: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate `batch_size` responses for the same prompt in one call.

    Returns:
        List of (token_ids, entropies) tuples, one per response.
        Each token_ids is np.ndarray of int, entropies is np.ndarray of float.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    single_ids = tokenizer(text, return_tensors="pt").input_ids  # (1, prompt_len)
    prompt_len = single_ids.shape[1]

    # Repeat prompt for the batch
    input_ids = single_ids.repeat(batch_size, 1).to(model.device)  # (B, prompt_len)

    # Create a fresh entropy collector
    collector = EntropyCollector()

    # Ensure pad token is set (needed for batched generation with EOS)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        logits_processor=LogitsProcessorList([collector]),
        pad_token_id=pad_token_id,
    )
    # outputs.shape: (batch_size, prompt_len + n_generated)

    # Entropy matrix: (n_steps, batch_size)
    ent_matrix = collector.get_entropies()  # (n_steps, B)

    # Extract per-response results, trimming at EOS
    eos_id = tokenizer.eos_token_id
    results = []

    for b in range(batch_size):
        gen_ids = outputs[b, prompt_len:].cpu().numpy()
        ent_vals = ent_matrix[:, b].numpy()

        # Find first EOS in generated tokens to trim padding
        eos_positions = np.where(gen_ids == eos_id)[0]
        if len(eos_positions) > 0:
            end = int(eos_positions[0]) + 1  # include the EOS token
        else:
            end = len(gen_ids)

        # Trim both to the same length
        end = min(end, len(gen_ids), len(ent_vals))
        results.append((gen_ids[:end], ent_vals[:end]))

    return results


# ── Generation loop ─────────────────────────────────────────────────────

def run_generation(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto",
    )
    model.eval()

    data = load_math500(n=args.n_questions)
    print(f"Loaded {len(data)} questions")
    print(f"Generating {args.n_responses} responses per question "
          f"(batch size: {args.batch_size})")

    all_token_ids = []
    all_entropies = []
    total_tokens = 0
    t0 = time.time()

    for q_idx, example in enumerate(data):
        # Generate in batches of batch_size
        n_remaining = args.n_responses
        q_tokens = 0

        while n_remaining > 0:
            bs = min(args.batch_size, n_remaining)

            results = generate_batch(
                model, tokenizer,
                example["prompt"],
                batch_size=bs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            for token_ids, entropies in results:
                all_token_ids.append(token_ids)
                all_entropies.append(entropies)
                q_tokens += len(entropies)

            n_remaining -= bs

        total_tokens += q_tokens
        elapsed = time.time() - t0
        tps = total_tokens / elapsed if elapsed > 0 else 0
        print(
            f"[Q {q_idx+1}/{len(data)}] "
            f"tokens_this_q={q_tokens:,}  total={total_tokens:,}  "
            f"elapsed={elapsed:.0f}s  tok/s={tps:.0f}"
        )

        # Checkpoint every 10 questions
        if (q_idx + 1) % 10 == 0:
            _save_arrays(output_dir, all_token_ids, all_entropies)
            print(f"  [checkpoint] saved {total_tokens:,} tokens")

    _save_arrays(output_dir, all_token_ids, all_entropies)

    # Save generation config for reproducibility
    config = {
        "model": args.model,
        "n_questions": args.n_questions,
        "n_responses": args.n_responses,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Saved {total_tokens:,} tokens to {output_dir}/")


def _save_arrays(output_dir, all_token_ids, all_entropies):
    ids = np.concatenate(all_token_ids)
    ent = np.concatenate(all_entropies)
    np.save(os.path.join(output_dir, "token_ids.npy"), ids)
    np.save(os.path.join(output_dir, "all_entropies.npy"), ent)


# ── Analysis (no GPU needed) ───────────────────────────────────────────

def run_analysis(args):
    output_dir = args.output_dir

    ids_path = os.path.join(output_dir, "token_ids.npy")
    ent_path = os.path.join(output_dir, "all_entropies.npy")

    print(f"Loading data from {output_dir}/")
    token_ids = np.load(ids_path)
    entropies = np.load(ent_path)
    assert len(token_ids) == len(entropies)
    print(f"Total tokens: {len(token_ids):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Aggregate by token type ──
    token_entropy_sum = defaultdict(float)
    token_count = defaultdict(int)

    for tid, ent in zip(token_ids, entropies):
        tid = int(tid)
        token_entropy_sum[tid] += float(ent)
        token_count[tid] += 1

    token_avg_entropy = {
        tid: token_entropy_sum[tid] / token_count[tid]
        for tid in token_entropy_sum
    }

    # ── Filter by minimum frequency ──
    min_freq = args.min_freq
    frequent_tokens = {
        tid: token_avg_entropy[tid]
        for tid in token_avg_entropy
        if token_count[tid] >= min_freq
    }
    print(f"Unique token types: {len(token_avg_entropy):,}")
    print(f"Frequent tokens (count >= {min_freq}): {len(frequent_tokens):,}")

    if len(frequent_tokens) == 0:
        print(f"\nNo tokens with frequency >= {min_freq}. "
              f"Try --min_freq {max(1, min_freq // 10)}")
        return

    # ── Sort ──
    sorted_by_entropy = sorted(frequent_tokens.items(), key=lambda x: x[1])
    n_show = min(args.top_n, len(sorted_by_entropy))

    bottom = sorted_by_entropy[:n_show]
    top = sorted_by_entropy[-n_show:][::-1]

    # ── Global stats ──
    p80 = float(np.percentile(entropies, 80))
    pct_below_001 = float((entropies < 1e-2).mean() * 100)

    print(f"\n{'='*70}")
    print(f"ENTROPY STATISTICS")
    print(f"{'='*70}")
    print(f"  Total tokens:            {len(entropies):,}")
    print(f"  Mean entropy:            {entropies.mean():.4f}")
    print(f"  Median entropy:          {np.median(entropies):.4f}")
    print(f"  80th percentile:         {p80:.4f}  (paper reports: 0.672)")
    print(f"  % tokens with H < 0.01:  {pct_below_001:.1f}%  (paper: ~50.6%)")

    print(f"\n{'='*70}")
    print(f"TOP {n_show} HIGH-ENTROPY TOKENS  ('forking' tokens)")
    print(f"{'='*70}")
    _print_token_table(top, tokenizer, token_count)

    print(f"\n{'='*70}")
    print(f"TOP {n_show} LOW-ENTROPY TOKENS  ('path-following' tokens)")
    print(f"{'='*70}")
    _print_token_table(bottom, tokenizer, token_count)

    # ── Save full ranking ──
    full_ranking = []
    for tid, avg_ent in sorted_by_entropy:
        text = tokenizer.decode([tid])
        full_ranking.append({
            "token_id": tid,
            "text": text,
            "repr": repr(text),
            "avg_entropy": round(avg_ent, 5),
            "count": token_count[tid],
        })
    ranking_path = os.path.join(output_dir, "token_entropy_ranking.json")
    with open(ranking_path, "w") as f:
        json.dump(full_ranking, f, indent=2, ensure_ascii=False)
    print(f"\nFull ranking saved to {ranking_path}")

    # ── Plots ──
    _plot_entropy_histogram(entropies, p80, output_dir)
    _plot_token_bars(top, bottom, tokenizer, token_count, output_dir)

    # ── Stats file ──
    stats = {
        "total_tokens": int(len(entropies)),
        "mean_entropy": float(entropies.mean()),
        "median_entropy": float(np.median(entropies)),
        "p80_entropy": float(p80),
        "pct_below_0.01": round(pct_below_001, 2),
        "n_frequent_tokens": len(frequent_tokens),
        "min_freq": min_freq,
    }
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


def _print_token_table(items, tokenizer, token_count):
    print(f"  {'Rank':<6} {'Token':<25} {'Repr':<30} {'AvgEntropy':>12} {'Count':>8}")
    print(f"  {'─'*6} {'─'*25} {'─'*30} {'─'*12} {'─'*8}")
    for rank, (tid, avg_ent) in enumerate(items, 1):
        text = tokenizer.decode([tid])
        display = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        if len(display) > 23:
            display = display[:20] + "..."
        print(f"  {rank:<6} {display:<25} {repr(text):<30} {avg_ent:>12.4f} {token_count[tid]:>8,}")


def _plot_entropy_histogram(entropies, p80, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, min(float(entropies.max()), 6.0), 200)
    ax.hist(entropies, bins=bins, color="#4a90d9", edgecolor="none", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Entropy", fontsize=14)
    ax.set_ylabel("Frequency (log scale)", fontsize=14)
    ax.set_title("(a) Distribution of token entropy", fontsize=15, fontweight="bold")
    ax.axvline(p80, color="red", linestyle="--", linewidth=1.5)
    ax.text(p80 + 0.08, ax.get_ylim()[1] * 0.3,
            f"The 80th percentile: {p80:.3f}", color="red", fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2a_entropy_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close(fig)


def _plot_token_bars(top, bottom, tokenizer, token_count, output_dir):
    n = min(30, len(top), len(bottom))
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, items, color, title in [
        (axes[0], top[:n], "#e74c3c",
         f"(b) Top {n} HIGH-entropy tokens\n(forking tokens)"),
        (axes[1], bottom[:n], "#3498db",
         f"(c) Top {n} LOW-entropy tokens\n(path-following tokens)"),
    ]:
        labels, values = [], []
        for tid, avg_ent in items:
            text = tokenizer.decode([tid]).replace('\n', '\\n')
            if len(text) > 15:
                text = text[:12] + "..."
            labels.append(f"{text} ({token_count[tid]})")
            values.append(avg_ent)

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=color, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("Average Entropy", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig2bc_token_entropy_bars.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze high/low entropy tokens (Figure 2a/b/c)"
    )
    sub = parser.add_subparsers(dest="command")

    # -- generate --
    gen = sub.add_parser("generate", help="Generate responses and collect entropies")
    gen.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    gen.add_argument("--n_questions", type=int, default=50)
    gen.add_argument("--n_responses", type=int, default=4)
    gen.add_argument("--batch_size", type=int, default=4,
                     help="Responses generated per forward pass. "
                          "Increase for faster throughput, decrease if OOM.")
    gen.add_argument("--max_new_tokens", type=int, default=4096)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--output_dir", type=str, default="entropy_results")

    # -- analyze --
    ana = sub.add_parser("analyze", help="Analyze saved token/entropy data")
    ana.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                     help="Tokenizer for decoding token IDs")
    ana.add_argument("--output_dir", type=str, default="entropy_results")
    ana.add_argument("--min_freq", type=int, default=20,
                     help="Min token frequency to include (paper uses 100)")
    ana.add_argument("--top_n", type=int, default=100)

    # -- all --
    both = sub.add_parser("all", help="Generate then analyze")
    both.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    both.add_argument("--n_questions", type=int, default=50)
    both.add_argument("--n_responses", type=int, default=4)
    both.add_argument("--batch_size", type=int, default=4)
    both.add_argument("--max_new_tokens", type=int, default=4096)
    both.add_argument("--temperature", type=float, default=1.0)
    both.add_argument("--output_dir", type=str, default="entropy_results")
    both.add_argument("--min_freq", type=int, default=20)
    both.add_argument("--top_n", type=int, default=100)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command in ("generate", "all"):
        run_generation(args)
    if args.command in ("analyze", "all"):
        run_analysis(args)


if __name__ == "__main__":
    main()
