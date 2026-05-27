"""Certified robustness via Randomized Smoothing for text."""
import torch
from typing import Tuple


class RandomizedSmoothing:
    """Certified defense adapted for text.

    Theorem 4: r_cert = sigma * (Phi^{-1}(p_A) - Phi^{-1}(p_B))
    """

    def __init__(self, sigma: float = 0.5, n_samples: int = 1000, alpha: float = 0.001):
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha

    def certify(self, model, input_text: str) -> Tuple[float, str]:
        """Compute certified robust radius."""
        votes = [self._classify_safety(model.generate(self._add_text_noise(input_text)))
                 for _ in range(self.n_samples)]
        from collections import Counter
        counts = Counter(votes)
        top_class, top_count = counts.most_common(1)[0]
        from scipy.stats import binom
        p_A = binom.ppf(1 - self.alpha, self.n_samples, top_count / self.n_samples) / self.n_samples
        if p_A > 0.5:
            from scipy.stats import norm
            return self.sigma * (norm.ppf(p_A) - norm.ppf(1 - p_A)), top_class
        return 0.0, "uncertain"

    def _add_text_noise(self, text: str) -> str:
        return text  # TODO: synonym replacement

    def _classify_safety(self, response: str) -> str:
        return "safe"  # TODO: safety classifier
