"""
DAPO trainer with entropy-masked policy gradient (Wang et al. 2025,
"Beyond the 80/20 Rule").

This subclasses trl.GRPOTrainer and overrides compute_loss to:
  1. Use Clip-Higher (asymmetric PPO clip: eps_low=0.2, eps_high=0.28)
  2. Use token-level loss aggregation (sum / N_tokens, not per-seq mean)
  3. Drop the KL term (beta=0)
  4. Mask the policy-gradient loss to the top-k% highest-entropy tokens
     in each completion. This is the experiment we're running.

Set DAPOConfig(use_entropy_mask=False) to get the baseline run that
trains on all completion tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from trl import GRPOTrainer


@dataclass
class DAPOConfig:
    """
    Hyperparameters specific to DAPO + entropy masking.

    Defaults mirror Wang et al.'s
    recipe/rlvr_with_high_entropy_tokens_only/run_only_top20_high_entropy_tokens_dapo_qwen3_14b.sh
    (clip_ratio_low=0.2, clip_ratio_high=0.28, clip_ratio_c=10.0,
     entropy_top_ratio=0.2).
    """
    eps_low: float = 0.2          # PPO clip lower
    eps_high: float = 0.28        # PPO clip upper (Clip-Higher)
    clip_ratio_c: float = 10.0    # DAPO triple-clip ceiling for negative-adv tokens

    # Entropy mask. Modes:
    #   "topk"      -> keep the GLOBAL top `entropy_top_ratio` of response tokens
    #                  (matches get_global_entropy_top_mask in the paper's verl fork)
    #   "bottom"    -> ablation: keep the GLOBAL bottom (1 - entropy_top_ratio)
    #   "off"       -> baseline, train on all response tokens
    use_entropy_mask: bool = True
    entropy_mask_mode: Literal["topk", "bottom", "off"] = "topk"
    entropy_top_ratio: float = 0.20      # Wang et al. main result: top 20%


def _per_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    H_t = -sum_v softmax(logits)_v * log_softmax(logits)_v
    logits: (B, T, V) -> (B, T)
    """
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def get_global_entropy_top_mask(
    entropy: torch.Tensor,        # (B, T)
    response_mask: torch.Tensor,  # (B, T) — 1 on completion tokens
    top_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Direct port of `get_global_entropy_top_mask` from
    verl/trainer/ppo/core_algos.py in the paper's fork:

    Select the top `top_ratio` fraction of response tokens BY ENTROPY,
    ranked GLOBALLY across the entire batch (not per sequence).
    """
    flat_entropy = entropy.flatten()
    flat_mask = response_mask.flatten().bool()

    response_entropy = flat_entropy[flat_mask]
    if response_entropy.numel() == 0:
        return torch.zeros_like(entropy)

    # ceil(N * r)
    top_k = max(1, int(response_entropy.numel() * top_ratio + 0.9999))
    _, topk_idx = torch.topk(response_entropy, k=top_k)

    response_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
    top_positions = response_positions[topk_idx]

    flat_out = torch.zeros_like(flat_entropy)
    flat_out[top_positions] = 1.0
    return flat_out.view_as(entropy)


def build_entropy_mask(
    entropies: torch.Tensor,        # (B, T)
    completion_mask: torch.Tensor,  # (B, T)
    cfg: DAPOConfig,
) -> torch.Tensor:
    """
    Returns a 0/1 mask of shape (B, T). All returned masks are
    restricted to completion tokens.
    """
    if not cfg.use_entropy_mask or cfg.entropy_mask_mode == "off":
        return completion_mask

    if cfg.entropy_mask_mode == "topk":
        top_mask = get_global_entropy_top_mask(
            entropies, completion_mask, top_ratio=cfg.entropy_top_ratio
        )
        return top_mask * completion_mask

    if cfg.entropy_mask_mode == "bottom":
        # Ablation: keep the COMPLEMENT of the top set.
        top_mask = get_global_entropy_top_mask(
            entropies, completion_mask, top_ratio=cfg.entropy_top_ratio
        )
        bottom_mask = (1.0 - top_mask) * completion_mask
        return bottom_mask

    raise ValueError(f"Unknown entropy_mask_mode: {cfg.entropy_mask_mode}")


class DAPOTrainer(GRPOTrainer):
    """
    Drop-in replacement for trl.GRPOTrainer with DAPO modifications.

    Pass `dapo_config=DAPOConfig(...)` at init. Set the GRPOConfig with
    `beta=0.0` (disable KL) and `loss_type="dapo"` is NOT required —
    we override the loss computation entirely.
    """

    def __init__(self, *args, dapo_config: DAPOConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dapo_config = dapo_config or DAPOConfig()
        if self.beta != 0.0:
            print(
                f"[DAPOTrainer] WARNING: beta={self.beta} but DAPO drops the KL term. "
                "Set GRPOConfig(beta=0.0) to match the paper."
            )

    # ------------------------------------------------------------------
    # Core DAPO loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Inputs (provided by GRPOTrainer's _prepare_inputs):
            prompt_completion_ids: (B, P+T)
            completion_mask:       (B, T)
            advantages:            (B,)            -- group-relative, per-sequence
            old_per_token_logps:   (B, T)          -- log π_old(a_t | s_t), the rollout policy
        """
        prompt_completion_ids = inputs["prompt_completion_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"].float()      # (B, T)
        advantages = inputs["advantages"]                        # (B,)
        old_per_token_logps = inputs["old_per_token_logps"]      # (B, T)

        # Forward pass on the *current* policy
        outputs = model(
            input_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # (B, P+T, V)

        # We only care about the positions that PRODUCE completion tokens.
        # The token at position p+t is produced by the logits at position p+t-1.
        completion_len = completion_mask.shape[1]
        # Slice logits aligned to completion tokens
        comp_logits = logits[:, -completion_len - 1 : -1, :]      # (B, T, V)
        comp_ids = prompt_completion_ids[:, -completion_len:]     # (B, T)

        # log π_θ(a_t | s_t)
        log_probs = torch.log_softmax(comp_logits.float(), dim=-1)
        new_per_token_logps = log_probs.gather(
            dim=-1, index=comp_ids.unsqueeze(-1)
        ).squeeze(-1)                                              # (B, T)

        # PPO ratio
        log_ratio = new_per_token_logps - old_per_token_logps      # (B, T)
        ratio = log_ratio.exp()

        # Broadcast per-sequence advantage to per-token
        adv = advantages.unsqueeze(1).expand_as(ratio)             # (B, T)

        # DAPO policy loss: clip-higher + triple-clip ceiling.
        # Mirrors verl/trainer/ppo/core_algos.compute_policy_loss_vanilla
        # in the paper's fork.
        eps_low = self.dapo_config.eps_low
        eps_high = self.dapo_config.eps_high
        clip_c = self.dapo_config.clip_ratio_c

        pg_losses1 = -adv * ratio
        pg_losses2 = -adv * torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)     # standard PPO min-clip equiv

        # Triple clip: only kicks in when advantage < 0 and ratio blows up
        pg_losses3 = -adv * clip_c
        clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)

        per_token_loss = torch.where(adv < 0, clip_pg_losses2, clip_pg_losses1)  # (B, T)

        # Entropy mask (computed from the CURRENT policy logits)
        with torch.no_grad():
            entropies = _per_token_entropy(comp_logits)            # (B, T)
            mask = build_entropy_mask(entropies, completion_mask, self.dapo_config)

        # Token-level aggregation: sum / total kept tokens (NOT per-seq mean).
        # This is DAPO § 3.3: long sequences contribute proportionally.
        denom = mask.sum().clamp(min=1.0)
        loss = (per_token_loss * mask).sum() / denom

        # ------- logging -------
        with torch.no_grad():
            self._metrics["loss"] = loss.detach().item()
            self._metrics["mean_entropy"] = (
                (entropies * completion_mask).sum() / completion_mask.sum().clamp(min=1)
            ).item()
            self._metrics["mask_frac"] = (
                mask.sum() / completion_mask.sum().clamp(min=1)
            ).item()
            self._metrics["clip_frac"] = (
                ((ratio < 1 - eps_low) | (ratio > 1 + eps_high)).float() * mask
            ).sum().item() / denom.item()

        if return_outputs:
            return loss, outputs
        return loss
