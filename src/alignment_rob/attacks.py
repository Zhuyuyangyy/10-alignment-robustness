"""Adversarial attacks on aligned LLMs."""
from typing import List


class GCGAttack:
    """Greedy Coordinate Gradient (Zou et al., 2023)."""
    def __init__(self, suffix_length: int = 20, n_steps: int = 500):
        self.suffix_length = suffix_length
        self.n_steps = n_steps

    def evaluate(self, model, dataset: str) -> float:
        return 0.0  # TODO


class PAIRAttack:
    """PAIR: Prompt Automatic Iterative Refinement (Chao et al., 2023)."""
    def __init__(self, max_queries: int = 20):
        self.max_queries = max_queries

    def evaluate(self, model, dataset: str) -> float:
        return 0.0


class AutoDANAttack:
    """AutoDAN (Liu et al., 2024)."""
    def __init__(self, n_generations: int = 50):
        self.n_generations = n_generations

    def evaluate(self, model, dataset: str) -> float:
        return 0.0
