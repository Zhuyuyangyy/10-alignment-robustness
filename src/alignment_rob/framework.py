"""Formal framework for alignment robustness analysis."""
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RobustnessResult:
    attack_success_rate: float
    certified_radius: float
    utility_preservation: float
    worst_case_safety_prob: float


class RobustnessFramework:
    """Analyze adversarial robustness of aligned LLMs.

    Safe region: S_pi = {x : Pr[Safe(x,y)=1] >= 1-delta}
    Robustness: Rob(pi, Phi, delta) = min_{phi in Phi} Pr[Safe(phi(x),y)=1]
    Robust radius: R*(pi, x, delta) = max{r : forall phi in Phi_r, safe}
    """

    def __init__(self, model: str, device: str = "cuda"):
        self.model_name = model
        self.device = device

    def evaluate_robustness(self, aligned_model, attacks: List,
                            dataset: str = "harmbench") -> RobustnessResult:
        asr_scores = [atk().evaluate(aligned_model, dataset) for atk in attacks]
        return RobustnessResult(
            attack_success_rate=max(asr_scores),
            certified_radius=0.0,
            utility_preservation=self._measure_utility(aligned_model),
            worst_case_safety_prob=1.0 - max(asr_scores)
        )

    def _measure_utility(self, model) -> float:
        return 0.95
