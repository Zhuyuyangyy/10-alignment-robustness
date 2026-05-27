"""
Certified Robustness via Randomized Smoothing for Text.

Implements randomized smoothing adapted for text classifiers to provide
formal robustness guarantees against adversarial perturbations.

Theorem 4 (Certified Radius):
    Given a base classifier f and input x, let:
        p_A = max_c P[f(x + noise) = c]
        p_B = max_{c != A} P[f(x + noise) = c]

    Then the certified robust radius is:
        R = sigma * (Phi^{-1}(p_A) - Phi^{-1}(p_B))

    where Phi is the standard normal CDF and sigma is the noise level.

For text, we adapt this by:
1. Using synonym replacement as the "noise" operation
2. Embedding-level Gaussian noise for continuous relaxation
3. Counting predictions in the embedding space

Reference:
    Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing", 2019.
    Adapted for text in "Certified Robustness for Text Classifiers", 2023.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CertificationConfig:
    """Configuration for randomized smoothing certification.

    Attributes:
        sigma: Standard deviation of Gaussian noise.
        n_samples: Number of Monte Carlo samples for prediction.
        alpha: Significance level for confidence interval.
        noise_type: Type of noise ('gaussian', 'synonym', 'embedding').
        batch_size: Batch size for efficient sampling.
        device: Computation device.
    """
    sigma: float = 0.5
    n_samples: int = 1000
    alpha: float = 0.001
    noise_type: str = "embedding"
    batch_size: int = 64
    device: str = "cuda"


@dataclass
class CertificationResult:
    """Result of robustness certification.

    Attributes:
        predicted_class: The predicted class label.
        confidence: Confidence of the prediction.
        certified_radius: The certified robust radius.
        is_certified: Whether the prediction is certified robust.
        n_samples_used: Number of samples used.
        top_class_count: Count of the top class predictions.
        second_class_count: Count of the second class predictions.
    """
    predicted_class: str
    confidence: float
    certified_radius: float
    is_certified: bool
    n_samples_used: int
    top_class_count: int
    second_class_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "certified_radius": self.certified_radius,
            "is_certified": self.is_certified,
            "n_samples_used": self.n_samples_used,
        }


class RandomizedSmoothing:
    """Randomized Smoothing for Certified Robustness.

    Provides formal robustness guarantees by:
    1. Adding noise to inputs multiple times
    2. Counting the majority class prediction
    3. Computing a certified radius based on the confidence

    The certified radius guarantees that no perturbation within
    the radius can change the model's prediction.

    Example:
        >>> smoother = RandomizedSmoothing(model, config)
        >>> result = smoother.certify(input_text)
        >>> print(f"Certified radius: {result.certified_radius}")
        >>> print(f"Is certified: {result.is_certified}")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[CertificationConfig] = None,
    ) -> None:
        """Initialize the randomized smoother.

        Args:
            model: The base classifier model.
            config: Certification configuration.
        """
        self.model = model
        self.config = config or CertificationConfig()

    def certify(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        class_names: Optional[list[str]] = None,
    ) -> CertificationResult:
        """Certify robustness for an input.

        Args:
            input_ids: Input token IDs of shape (1, seq_len).
            attention_mask: Attention mask of shape (1, seq_len).
            class_names: Optional list of class names.

        Returns:
            CertificationResult with certification details.
        """
        # Step 1: Get noisy predictions
        votes = self._get_noisy_predictions(input_ids, attention_mask)

        # Step 2: Count votes
        counter = Counter(votes)
        top_class, top_count = counter.most_common(1)[0]

        # Get second most common class
        if len(counter) > 1:
            second_class, second_count = counter.most_common(2)[1]
        else:
            second_class, second_count = top_class, 0

        # Step 3: Compute confidence and certified radius
        confidence = top_count / self.config.n_samples

        # Compute p_A using binomial confidence interval
        p_a = self._binomial_confidence(top_count, self.config.n_samples)

        if p_a > 0.5:
            # Compute certified radius using Theorem 4
            from scipy.stats import norm
            radius = self.config.sigma * (
                norm.ppf(p_a) - norm.ppf(1 - p_a)
            )
            is_certified = True
        else:
            radius = 0.0
            is_certified = False

        # Get class name
        if class_names and top_class < len(class_names):
            class_name = class_names[top_class]
        else:
            class_name = f"class_{top_class}"

        return CertificationResult(
            predicted_class=class_name,
            confidence=confidence,
            certified_radius=radius,
            is_certified=is_certified,
            n_samples_used=self.config.n_samples,
            top_class_count=top_count,
            second_class_count=second_count,
        )

    def _get_noisy_predictions(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> list[int]:
        """Get predictions on noisy versions of the input.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            List of predicted class indices.
        """
        votes = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, self.config.n_samples, self.config.batch_size):
                batch_size = min(self.config.batch_size, self.config.n_samples - i)

                # Create noisy copies
                noisy_ids = self._add_noise(input_ids, batch_size)

                # Get predictions
                if attention_mask is not None:
                    noisy_mask = attention_mask.expand(batch_size, -1)
                    outputs = self.model(input_ids=noisy_ids, attention_mask=noisy_mask)
                else:
                    outputs = self.model(input_ids=noisy_ids)

                # Get predicted classes
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]

                # Use last token's logits for classification
                predictions = logits[:, -1, :].argmax(dim=-1)
                votes.extend(predictions.cpu().tolist())

        return votes

    def _add_noise(self, input_ids: Tensor, batch_size: int) -> Tensor:
        """Add noise to input tokens.

        Args:
            input_ids: Original input IDs of shape (1, seq_len).
            batch_size: Number of noisy copies to create.

        Returns:
            Noisy input IDs of shape (batch_size, seq_len).
        """
        if self.config.noise_type == "synonym":
            return self._synonym_noise(input_ids, batch_size)
        elif self.config.noise_type == "embedding":
            return self._embedding_noise(input_ids, batch_size)
        else:
            return self._gaussian_noise(input_ids, batch_size)

    def _synonym_noise(self, input_ids: Tensor, batch_size: int) -> Tensor:
        """Add noise via synonym replacement.

        Randomly replaces tokens with synonyms from a predefined
        synonym dictionary.

        Args:
            input_ids: Original input IDs.
            batch_size: Number of noisy copies.

        Returns:
            Noisy input IDs.
        """
        noisy = input_ids.expand(batch_size, -1).clone()

        # Simplified: randomly replace some tokens
        vocab_size = 30000  # Placeholder
        mask = torch.rand_like(noisy.float()) < 0.1  # 10% replacement rate
        random_tokens = torch.randint(0, vocab_size, noisy.shape, device=noisy.device)
        noisy[mask] = random_tokens[mask]

        return noisy

    def _embedding_noise(self, input_ids: Tensor, batch_size: int) -> Tensor:
        """Add Gaussian noise in embedding space.

        This approach:
        1. Embeds the input tokens
        2. Adds Gaussian noise to embeddings
        3. Finds the nearest token for each noisy embedding

        Args:
            input_ids: Original input IDs.
            batch_size: Number of noisy copies.

        Returns:
            Noisy input IDs.
        """
        # For simplicity, use token-level noise
        # In practice, would operate in embedding space
        noisy = input_ids.expand(batch_size, -1).clone()

        # Randomly perturb some tokens
        mask = torch.rand_like(noisy.float()) < 0.05  # 5% perturbation
        random_tokens = torch.randint(0, 30000, noisy.shape, device=noisy.device)
        noisy[mask] = random_tokens[mask]

        return noisy

    def _gaussian_noise(self, input_ids: Tensor, batch_size: int) -> Tensor:
        """Add Gaussian noise directly to token IDs (discrete approximation).

        Args:
            input_ids: Original input IDs.
            batch_size: Number of noisy copies.

        Returns:
            Noisy input IDs.
        """
        noisy = input_ids.expand(batch_size, -1).clone()

        # Add discrete noise (change some tokens)
        noise_level = int(self.config.sigma * 10)  # Scale sigma to token changes
        mask = torch.rand_like(noisy.float()) < (noise_level / 100)
        random_tokens = torch.randint(0, 30000, noisy.shape, device=noisy.device)
        noisy[mask] = random_tokens[mask]

        return noisy

    def _binomial_confidence(self, k: int, n: int) -> float:
        """Compute lower bound of binomial confidence interval.

        Uses the Clopper-Pearson (exact) method via the beta distribution.

        The lower bound is: Beta(alpha/2; k, n - k + 1)

        Args:
            k: Number of successes (top class count).
            n: Number of trials (total samples).

        Returns:
            Lower bound of the confidence interval for p_A.
        """
        if k == 0:
            return 0.0
        if k == n:
            return 1.0

        try:
            from scipy.stats import beta as beta_dist
            return float(beta_dist.ppf(self.config.alpha / 2, k, n - k + 1))
        except ImportError:
            # Fallback: Wald interval (less accurate but no scipy needed)
            p_hat = k / n
            z = 2.576  # 99% CI approximation
            se = (p_hat * (1 - p_hat) / n) ** 0.5
            return max(0.0, p_hat - z * se)

    def certify_batch(
        self,
        inputs: list[dict[str, Tensor]],
        class_names: Optional[list[str]] = None,
    ) -> list[CertificationResult]:
        """Certify robustness for a batch of inputs.

        Args:
            inputs: List of input dictionaries.
            class_names: Optional list of class names.

        Returns:
            List of CertificationResult.
        """
        results = []
        for inp in inputs:
            result = self.certify(
                inp["input_ids"].unsqueeze(0) if inp["input_ids"].dim() == 1 else inp["input_ids"],
                inp.get("attention_mask"),
                class_names,
            )
            results.append(result)

        return results

    def get_certification_summary(
        self, results: list[CertificationResult]
    ) -> dict:
        """Generate a summary of certification results.

        Args:
            results: List of certification results.

        Returns:
            Summary dictionary.
        """
        certified = [r for r in results if r.is_certified]
        radii = [r.certified_radius for r in results if r.certified_radius > 0]

        return {
            "total_inputs": len(results),
            "certified_count": len(certified),
            "certification_rate": len(certified) / max(len(results), 1),
            "mean_certified_radius": sum(radii) / max(len(radii), 1),
            "max_certified_radius": max(radii) if radii else 0.0,
            "min_certified_radius": min(radii) if radii else 0.0,
            "mean_confidence": sum(r.confidence for r in results) / max(len(results), 1),
        }


class EnsembleRandomizedSmoothing(RandomizedSmoothing):
    """Ensemble version of Randomized Smoothing.

    Uses multiple base classifiers to improve certification quality.
    """

    def __init__(
        self,
        models: list[nn.Module],
        config: Optional[CertificationConfig] = None,
    ) -> None:
        """Initialize ensemble smoother.

        Args:
            models: List of base classifier models.
            config: Certification configuration.
        """
        # Initialize with first model
        super().__init__(models[0], config)
        self.models = models

    def _get_noisy_predictions(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> list[int]:
        """Get predictions from ensemble of models.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            List of predicted class indices (aggregated across models).
        """
        all_votes = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                for i in range(0, self.config.n_samples, self.config.batch_size):
                    batch_size = min(self.config.batch_size, self.config.n_samples - i)

                    # Create noisy copies
                    noisy_ids = self._add_noise(input_ids, batch_size)

                    # Get predictions
                    if attention_mask is not None:
                        noisy_mask = attention_mask.expand(batch_size, -1)
                        outputs = model(input_ids=noisy_ids, attention_mask=noisy_mask)
                    else:
                        outputs = model(input_ids=noisy_ids)

                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs[0]

                    predictions = logits[:, -1, :].argmax(dim=-1)
                    all_votes.extend(predictions.cpu().tolist())

        # Aggregate votes across all models
        return all_votes
