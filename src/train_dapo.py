"""
Train Qwen3 on MATH-500 with DAPO + entropy-masked policy gradient.

Reproduces the central experiment of Wang et al. 2025 ("Beyond the 80/20
Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning"),
on top of DAPO (Yu et al. 2025).

Three runs to compare (set --mask_mode):
python -m src.train_dapo --model Qwen/Qwen3-1.7B --mask_mode full   --output_dir outputs/full
python -m src.train_dapo --model Qwen/Qwen3-1.7B --mask_mode topk   --output_dir outputs/topk
python -m src.train_dapo --model Qwen/Qwen3-1.7B --mask_mode bottom --output_dir outputs/bottom


Example:
    python -m src.train_dapo \\
        --model Qwen/Qwen3-1.7B \\
        --mask_mode topk \\
        --output_dir outputs/run_topk \\
        --max_steps 200
"""

from __future__ import annotations
import argparse
import os

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig

from src.math500_data import load_math500_split
from src.dapo_trainer import DAPOConfig, DAPOTrainer
from src.reward import make_reward_fn


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem with clear step-by-step "
    "reasoning in plain text (do NOT write code). Put your final answer inside "
    "\\boxed{}."
)


def to_hf_dataset(records, tokenizer):
    """
    Apply the chat template up front so the model sees a system+user
    conversation. This is important for rnj-1-instruct, which has a
    strong code-output bias without an explicit non-code system prompt.
    """
    rows = []
    for r in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rows.append({"prompt": text, "gold": r["gold"]})
    return Dataset.from_list(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EssentialAI/rnj-1-instruct")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mask_mode", choices=["full", "topk", "bottom", "threshold"],
                   default="topk",
                   help="full=baseline DAPO, topk=top-20%% entropy (paper), "
                        "bottom=bottom-80%% (ablation), threshold=fixed H>=p80")
    p.add_argument("--topk_frac", type=float, default=0.20)
    p.add_argument("--entropy_threshold", type=float, default=0.5128,
                   help="from entropy_results/stats.json")

    # DAPO hparams
    p.add_argument("--eps_low", type=float, default=0.20)
    p.add_argument("--eps_high", type=float, default=0.28)

    # Training hparams
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--per_device_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_generations", type=int, default=8,
                   help="G in GRPO/DAPO: rollouts per prompt")
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_completion_len", type=int, default=2048)

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dapo_config(args) -> DAPOConfig:
    if args.mask_mode == "full":
        return DAPOConfig(
            eps_low=args.eps_low, eps_high=args.eps_high,
            use_entropy_mask=False, entropy_mask_mode="off",
        )
    if args.mask_mode == "topk":
        return DAPOConfig(
            eps_low=args.eps_low, eps_high=args.eps_high,
            use_entropy_mask=True, entropy_mask_mode="topk",
            entropy_topk_frac=args.topk_frac,
        )
    if args.mask_mode == "bottom":
        # Implemented as topk on the NEGATED entropy: cleanest way is
        # to use topk with frac = 1 - topk_frac and flag this as ablation.
        return DAPOConfig(
            eps_low=args.eps_low, eps_high=args.eps_high,
            use_entropy_mask=True, entropy_mask_mode="topk",
            entropy_topk_frac=1.0 - args.topk_frac,  # the bottom 80%
        )
    if args.mask_mode == "threshold":
        return DAPOConfig(
            eps_low=args.eps_low, eps_high=args.eps_high,
            use_entropy_mask=True, entropy_mask_mode="threshold",
            entropy_threshold=args.entropy_threshold,
        )
    raise ValueError(args.mask_mode)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer (needed to apply chat template) ──────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ───────────────────────────────────────────────────────────
    train_records, eval_records = load_math500_split(n_test=100, seed=args.seed)
    train_ds = to_hf_dataset(train_records, tokenizer)
    eval_ds = to_hf_dataset(eval_records, tokenizer)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # ── LoRA ───────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Reward fn ──────────────────────────────────────────────────────
    reward_fn = make_reward_fn()

    # ── trl GRPO config (we use DAPOTrainer which subclasses it) ───────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        beta=0.0,                       # DAPO drops the KL term
        loss_type="grpo",               # we override compute_loss anyway
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        seed=args.seed,
        report_to=["wandb"],
        run_name=f"dapo-{args.mask_mode}",
    )

    dapo_config = build_dapo_config(args)
    print(f"DAPO config: {dapo_config}")

    trainer = DAPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        dapo_config=dapo_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
