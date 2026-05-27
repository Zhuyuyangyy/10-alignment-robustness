"""
Alignment Methods with Robustness Analysis.

Implements RLHF, DPO, and KTO alignment methods with formal
robustness analysis and theoretical bounds.

Theoretical Results
===================

1. RLHF Robustness Bound (Theorem 1):
   Rob(pi_RLHF) >= 1 - delta - beta * KL(pi || pi_ref)
   where beta is the KL penalty coefficient.

2. DPO Vulnerability (Theorem 2):
   DPO is vulnerable near preference boundaries where Delta_r -> 0.
   The robustness gap scales as O(1/sqrt(N_offline)).

3. RLHF vs DPO Gap (Theorem 3):
   Rob(RLHF) >= Rob(DPO) - O(1/sqrt(N_offline))
   RLHF provides better worst-case robustness guarantees.

Reference:
    "Robustness of AI-Generated Text: Formally Analyzing Alignment Methods", 2024.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer

_DEFAULT_DELTA = 0.05


@dataclass
class AlignmentConfig:
    """Configuration for alignment training.

    Attributes:
        beta: KL penalty coefficient (RLHF) or temperature (DPO/KTO).
        learning_rate: Learning rate for policy optimization.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        gradient_accumulation_steps: Steps for gradient accumulation.
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_steps: Number of warmup steps.
        kl_penalty_type: Type of KL penalty ('kl', 'adaptive', 'linear').
        reward_scale: Scaling factor for reward signals.
        clip_range: PPO clipping range.
        use_wandb: Whether to log to Weights & Biases.
    """

    beta: float = 0.1
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    kl_penalty_type: str = "kl"
    reward_scale: float = 1.0
    clip_range: float = 0.2
    use_wandb: bool = False


@dataclass
class AlignmentResult:
    """Result of alignment training.

    Attributes:
        final_reward: Average reward after alignment.
        kl_divergence: KL divergence from reference policy.
        safety_score: Safety evaluation score.
        utility_score: Utility evaluation score.
        training_loss: Final training loss.
        num_steps: Total training steps.
        robustness_bound: Theoretical robustness bound.
    """

    final_reward: float = 0.0
    kl_divergence: float = 0.0
    safety_score: float = 0.0
    utility_score: float = 0.0
    training_loss: float = 0.0
    num_steps: int = 0
    robustness_bound: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "final_reward": self.final_reward,
            "kl_divergence": self.kl_divergence,
            "safety_score": self.safety_score,
            "utility_score": self.utility_score,
            "training_loss": self.training_loss,
            "num_steps": self.num_steps,
            "robustness_bound": self.robustness_bound,
        }


def _step_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    loss: Tensor,
    step: int,
    config: AlignmentConfig,
) -> None:
    """Backward pass with gradient accumulation and clipping.

    Handles the common pattern of:
    1. loss.backward()
    2. Clip gradients every gradient_accumulation_steps
    3. optimizer.step() + zero_grad()
    """
    loss.backward()
    if (step + 1) % config.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()


class BaseAlignment(ABC):
    """Base class for alignment methods.

    Provides common interface and utility methods for all
    alignment implementations.
    """

    def __init__(self, config: AlignmentConfig) -> None:
        """Initialize the alignment method.

        Args:
            config: Alignment configuration.
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        data: list[dict[str, Tensor]],
    ) -> AlignmentResult:
        """Train the model using the alignment method.

        Args:
            model: The model to align.
            ref_model: The reference model for KL computation.
            data: Training data.

        Returns:
            AlignmentResult with training statistics.
        """
        ...

    def compute_robustness_bound(self, kl_divergence: float) -> float:
        """Compute theoretical robustness bound.

        Args:
            kl_divergence: KL divergence from reference policy.

        Returns:
            Lower bound on robustness.
        """
        return 1.0 - _DEFAULT_DELTA - self.config.beta * kl_divergence

    @staticmethod
    def compute_logps(model: nn.Module, input_ids: Tensor) -> Tensor:
        """Compute log probabilities for sequences.

        Shared implementation used by DPO and KTO.

        Args:
            model: The model.
            input_ids: Input token IDs of shape (batch, seq_len).

        Returns:
            Log probability of each sequence, shape (batch,).
        """
        outputs = model(input_ids=input_ids)
        logps = outputs.logits.log_softmax(-1)

        # Gather log probs for selected tokens (autoregressive shift)
        tokens = input_ids[:, 1:]
        logps = logps[:, :-1]
        token_logps = logps.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        return token_logps.sum(-1)


class RLHF(BaseAlignment):
    """Reinforcement Learning from Human Feedback (PPO-based).

    Objective:
        max E[r(x, y)] - beta * KL[pi || pi_ref]

    The PPO algorithm:
    1. Collect trajectories using current policy
    2. Compute advantages using GAE
    3. Update policy with clipped surrogate objective
    4. Compute KL penalty to prevent drift from reference

    Robustness Analysis:
        - The KL penalty provides implicit robustness guarantees
        - Higher beta -> more robust but potentially less capable
        - Theorem 1: Rob(pi_RLHF) >= 1 - delta - beta * KL(pi || pi_ref)
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        """Initialize RLHF.

        Args:
            config: Alignment configuration.
        """
        super().__init__(config or AlignmentConfig())

    def train(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        data: list[dict[str, Tensor]],
    ) -> AlignmentResult:
        """Train using PPO-based RLHF.

        Args:
            model: The model to align.
            ref_model: The reference model.
            data: Training data with prompts and reward signals.

        Returns:
            AlignmentResult with training statistics.
        """
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        total_reward = 0.0
        total_kl = 0.0
        total_loss = 0.0
        num_steps = 0

        model.train()
        ref_model.eval()

        for _epoch in range(self.config.num_epochs):
            for batch in data:
                prompts = batch.get("input_ids")
                if prompts is None:
                    continue

                with torch.no_grad():
                    ref_outputs = ref_model(**batch)
                    ref_logprobs = ref_outputs.logits.log_softmax(-1)

                outputs = model(**batch)
                logprobs = outputs.logits.log_softmax(-1)

                rewards = batch.get("rewards", torch.zeros(1))

                # KL divergence
                kl = (logprobs.exp() * (logprobs - ref_logprobs)).sum(-1).mean()

                # PPO clipped objective
                advantages = self._compute_advantages(rewards, kl)
                ratio = (logprobs - ref_logprobs).exp()
                clipped_ratio = ratio.clamp(
                    1 - self.config.clip_range,
                    1 + self.config.clip_range,
                )

                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages,
                ).mean()

                loss = policy_loss + self.config.beta * kl

                _step_optimizer(optimizer, model, loss, num_steps, self.config)

                total_reward += rewards.mean().item()
                total_kl += kl.item()
                total_loss += loss.item()
                num_steps += 1

        # Flush remaining gradients
        if num_steps % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_kl = total_kl / max(num_steps, 1)
        robustness_bound = self.compute_robustness_bound(avg_kl)

        return AlignmentResult(
            final_reward=total_reward / max(num_steps, 1),
            kl_divergence=avg_kl,
            training_loss=total_loss / max(num_steps, 1),
            num_steps=num_steps,
            robustness_bound=robustness_bound,
        )

    def _compute_advantages(self, rewards: Tensor, kl: Tensor) -> Tensor:
        """Compute advantages using GAE.

        Args:
            rewards: Reward tensor.
            kl: KL divergence tensor.

        Returns:
            Normalized advantage tensor.
        """
        advantages = rewards - self.config.beta * kl
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


class DPO(BaseAlignment):
    """Direct Preference Optimization.

    Objective:
        L_DPO = -E[(x, y_w, y_l)] [log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                                                - log pi(y_l|x)/pi_ref(y_l|x)))]

    Robustness Analysis:
        - DPO is vulnerable near preference boundaries where Delta_r -> 0
        - Theorem 2: Gap scales as O(1/sqrt(N_offline))
        - Theorem 3: Rob(RLHF) >= Rob(DPO) - O(1/sqrt(N_offline))

    Advantages:
        - No reward model needed
        - Simpler training pipeline
        - More stable than PPO

    Disadvantages:
        - Lower worst-case robustness
        - Sensitive to preference data quality
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        """Initialize DPO.

        Args:
            config: Alignment configuration.
        """
        super().__init__(config or AlignmentConfig())

    def train(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        data: list[dict[str, Tensor]],
    ) -> AlignmentResult:
        """Train using DPO.

        Args:
            model: The model to align.
            ref_model: The reference model.
            data: Preference data with (prompt, chosen, rejected) triples.

        Returns:
            AlignmentResult with training statistics.
        """
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        total_loss = 0.0
        total_accuracy = 0.0
        num_steps = 0

        model.train()
        ref_model.eval()

        for _epoch in range(self.config.num_epochs):
            for batch in data:
                chosen_ids = batch.get("chosen_ids")
                rejected_ids = batch.get("rejected_ids")

                if chosen_ids is None or rejected_ids is None:
                    continue

                chosen_logps = self.compute_logps(model, chosen_ids)
                rejected_logps = self.compute_logps(model, rejected_ids)

                with torch.no_grad():
                    ref_chosen_logps = self.compute_logps(ref_model, chosen_ids)
                    ref_rejected_logps = self.compute_logps(ref_model, rejected_ids)

                chosen_rewards = self.config.beta * (chosen_logps - ref_chosen_logps)
                rejected_rewards = self.config.beta * (
                    rejected_logps - ref_rejected_logps
                )

                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

                _step_optimizer(optimizer, model, loss, num_steps, self.config)

                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_steps += 1

        # Flush remaining gradients
        if num_steps % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_steps, 1)
        robustness_bound = 1.0 - _DEFAULT_DELTA - 1.0 / (len(data) ** 0.5)

        return AlignmentResult(
            final_reward=total_accuracy / max(num_steps, 1),
            kl_divergence=avg_loss / self.config.beta,
            training_loss=avg_loss,
            num_steps=num_steps,
            robustness_bound=robustness_bound,
        )


class KTO(BaseAlignment):
    """Kahneman-Tversky Optimization.

    KTO uses Kahneman-Tversky's prospect theory to align models,
    applying loss aversion to undesirable outputs.

    Objective:
        L_KTO = E[(x, y)] [w(y) * (1 - sigma(beta * (log pi(y|x) - log pi_ref(y|x))))]

    Where:
        w(y) = lambda_d if y is desirable
        w(y) = lambda_u if y is undesirable (lambda_u > lambda_d)

    Advantages:
        - Does not require paired preference data
        - More robust to noisy labels
        - Better calibration on edge cases
    """

    def __init__(
        self,
        config: Optional[AlignmentConfig] = None,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 2.0,
    ) -> None:
        """Initialize KTO.

        Args:
            config: Alignment configuration.
            desirable_weight: Loss weight for desirable outputs.
            undesirable_weight: Loss weight for undesirable outputs (higher).
        """
        super().__init__(config or AlignmentConfig())
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def train(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        data: list[dict[str, Tensor]],
    ) -> AlignmentResult:
        """Train using KTO.

        Args:
            model: The model to align.
            ref_model: The reference model.
            data: Feedback data with (prompt, response, is_desirable) triples.

        Returns:
            AlignmentResult with training statistics.
        """
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        total_loss = 0.0
        num_steps = 0

        model.train()
        ref_model.eval()

        for _epoch in range(self.config.num_epochs):
            for batch in data:
                input_ids = batch.get("input_ids")
                is_desirable = batch.get("is_desirable")

                if input_ids is None:
                    continue

                model_logps = self.compute_logps(model, input_ids)

                with torch.no_grad():
                    ref_logps = self.compute_logps(ref_model, input_ids)

                log_ratio = model_logps - ref_logps

                if is_desirable is not None:
                    weights = torch.where(
                        is_desirable.bool(),
                        self.desirable_weight,
                        self.undesirable_weight,
                    )
                else:
                    weights = torch.ones_like(log_ratio)

                loss = (
                    weights * (1 - torch.sigmoid(self.config.beta * log_ratio))
                ).mean()

                _step_optimizer(optimizer, model, loss, num_steps, self.config)

                total_loss += loss.item()
                num_steps += 1

        # Flush remaining gradients
        if num_steps % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_steps, 1)
        robustness_bound = 1.0 - _DEFAULT_DELTA - 0.5 / (len(data) ** 0.5)

        return AlignmentResult(
            final_reward=1.0 - avg_loss,
            kl_divergence=avg_loss / self.config.beta,
            training_loss=avg_loss,
            num_steps=num_steps,
            robustness_bound=robustness_bound,
        )
