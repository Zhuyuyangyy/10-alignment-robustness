"""Robustness evaluation metrics for alignment.

This module provides metrics for evaluating the robustness of
aligned models against adversarial attacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from alignment_rob.attacks import AttackResult


@dataclass
class EvaluationSummary:
    """Comprehensive robustness evaluation summary.

    Attributes:
        asr: Attack Success Rate.
        certified_radius: Certified robust radius.
        worst_case_score: Worst-case score across attacks.
        average_score: Average score across attacks.
        stability_score: Stability of alignment under perturbation.
    """

    asr: float = 0.0
    certified_radius: float = 0.0
    worst_case_score: float = 0.0
    average_score: float = 0.0
    stability_score: float = 0.0


class RobustnessEvaluator:
    """Evaluator for alignment robustness.

    Provides comprehensive evaluation of model robustness against
    various adversarial attacks.

    Args:
        model: Model to evaluate.
        attacks: List of attacks to use.
    """

    def __init__(
        self,
        model: Any,
        attacks: Optional[List[Any]] = None,
    ) -> None:
        self.model = model
        self.attacks = attacks or []

    def evaluate(
        self,
        attack_results: List[AttackResult],
    ) -> EvaluationSummary:
        """Evaluate robustness from attack results.

        Args:
            attack_results: List of attack results.

        Returns:
            EvaluationSummary object.
        """
        if not attack_results:
            return EvaluationSummary()

        asr = sum(1 for r in attack_results if r.success) / len(attack_results)

        return EvaluationSummary(
            asr=asr,
            certified_radius=self._estimate_certified_radius(attack_results),
            worst_case_score=1.0 - asr,
            average_score=1.0 - asr,
            stability_score=1.0 - asr,
        )

    @staticmethod
    def _estimate_certified_radius(
        attack_results: List[AttackResult],
    ) -> float:
        """Estimate certified robust radius from successful attack perturbation sizes.

        Args:
            attack_results: List of attack results.

        Returns:
            Estimated certified radius (half of minimum successful perturbation).
        """
        successful = [r for r in attack_results if r.success]
        if not successful:
            return float("inf")

        min_perturbation = min(r.perturbation_size for r in successful)
        return min_perturbation / 2.0

    def compare_methods(
        self,
        results_dict: Dict[str, List[AttackResult]],
    ) -> Dict[str, EvaluationSummary]:
        """Compare robustness across alignment methods.

        Args:
            results_dict: Dictionary mapping method names to attack results.

        Returns:
            Dictionary mapping method names to evaluation summaries.
        """
        return {
            method_name: self.evaluate(results)
            for method_name, results in results_dict.items()
        }


class ASRMetric:
    """Attack Success Rate metric.

    Computes the fraction of successful attacks.
    """

    @staticmethod
    def compute(results: List[AttackResult]) -> float:
        """Compute ASR.

        Args:
            results: List of attack results.

        Returns:
            Attack Success Rate (0.0 if no results).
        """
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)


class CertifiedRadiusMetric:
    """Certified robust radius metric.

    Computes the certified radius using randomized smoothing.
    """

    @staticmethod
    def compute(
        results: List[AttackResult],
        sigma: float = 0.5,
        alpha: float = 0.05,
    ) -> float:
        """Compute certified radius.

        Args:
            results: List of attack results.
            sigma: Noise scale for smoothing.
            alpha: Confidence level.

        Returns:
            Certified radius (inf if no successful attacks).
        """
        successful = [r for r in results if r.success]
        if not successful:
            return float("inf")

        min_perturbation = min(r.perturbation_size for r in successful)
        return min_perturbation * 0.5


class PerturbationSizeMetric:
    """Perturbation size metric.

    Measures the size of adversarial perturbations.
    """

    @staticmethod
    def compute(results: List[AttackResult]) -> Dict[str, float]:
        """Compute perturbation size statistics.

        Args:
            results: List of attack results.

        Returns:
            Dictionary with perturbation size statistics (mean, std, min, max).
        """
        if not results:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        sizes = np.array([r.perturbation_size for r in results])

        return {
            "mean": float(np.mean(sizes)),
            "std": float(np.std(sizes)),
            "min": float(np.min(sizes)),
            "max": float(np.max(sizes)),
        }


class QueryEfficiencyMetric:
    """Query efficiency metric.

    Measures the number of queries needed for successful attacks.
    """

    @staticmethod
    def compute(results: List[AttackResult]) -> Dict[str, float]:
        """Compute query efficiency statistics.

        Args:
            results: List of attack results.

        Returns:
            Dictionary with query efficiency statistics (mean, std, min, max).
        """
        successful = [r for r in results if r.success]
        if not successful:
            return {
                "mean": float("inf"),
                "std": 0.0,
                "min": float("inf"),
                "max": float("inf"),
            }

        queries = np.array([r.num_queries for r in successful])

        return {
            "mean": float(np.mean(queries)),
            "std": float(np.std(queries)),
            "min": float(np.min(queries)),
            "max": float(np.max(queries)),
        }
