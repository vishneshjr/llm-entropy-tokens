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
    """Hyperparameters specific to DAPO + entropy masking."""
    eps_low: float = 0.2          # PPO clip lower (paper default)
    eps_high: float = 0.28        # PPO clip upper (Clip-Higher; paper default)

    # Entropy mask
    use_entropy_mask: bool = True
    entropy_mask_mode: Literal["topk", "threshold", "off"] = "topk"
    entropy_topk_frac: float = 0.20      # Wang et al. main result: top 20%
    entropy_threshold: float = 0.5128    # from entropy_results/stats.json p80
    # Note: "topk" recomputes per-sequence each step (robust to drift).
    # "threshold" uses a fixed entropy value — useful for ablation against
    # the analysis in entropy_results/entropy_distribution.png.


def _per_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    H_t = -sum_v softmax(logits)_v * log_softmax(logits)_v
    logits: (B, T, V) -> (B, T)
    """
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def build_entropy_mask(
    entropies: torch.Tensor,        # (B, T)
    completion_mask: torch.Tensor,  # (B, T) — 1 on completion tokens
    cfg: DAPOConfig,
) -> torch.Tensor:
    """
    Returns a 0/1 mask of shape (B, T): 1 on the tokens whose loss
    we keep, 0 elsewhere. Always restricted to completion tokens.
    """
    if not cfg.use_entropy_mask or cfg.entropy_mask_mode == "off":
        return completion_mask

    if cfg.entropy_mask_mode == "threshold":
        keep = (entropies >= cfg.entropy_threshold).float()
        return keep * completion_mask

    # Default: top-k% per sequence, restricted to completion tokens.
    B, T = entropies.shape
    masked_ent = entropies.masked_fill(completion_mask == 0, float("-inf"))
    n_keep = (completion_mask.sum(dim=-1) * cfg.entropy_topk_frac).long().clamp(min=1)

    keep = torch.zeros_like(entropies)
    for b in range(B):
        k = int(n_keep[b].item())
        if k == 0:
            continue
        # topk over the (T,) row; tokens not in completion_mask are -inf
        _, idx = torch.topk(masked_ent[b], k=min(k, T))
        keep[b, idx] = 1.0
    return keep * completion_mask


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

        # Clip-Higher: asymmetric clip
        eps_low = self.dapo_config.eps_low
        eps_high = self.dapo_config.eps_high
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
        per_token_loss = -torch.min(unclipped, clipped)            # (B, T)

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
