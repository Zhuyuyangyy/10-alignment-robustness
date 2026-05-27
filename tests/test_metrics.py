"""Unit tests for the metrics module."""

import math

import pytest

from alignment_rob.attacks import AttackResult
from alignment_rob.metrics import (
    ASRMetric,
    CertifiedRadiusMetric,
    EvaluationSummary,
    PerturbationSizeMetric,
    QueryEfficiencyMetric,
    RobustnessEvaluator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    success: bool = False,
    perturbation_size: float = 0.1,
    num_queries: int = 100,
) -> AttackResult:
    return AttackResult(
        attack_name="test",
        success=success,
        perturbation_size=perturbation_size,
        num_queries=num_queries,
    )


@pytest.fixture
def mixed_results():
    return [
        _make_result(success=True, perturbation_size=0.2, num_queries=50),
        _make_result(success=False, perturbation_size=0.0, num_queries=100),
        _make_result(success=True, perturbation_size=0.4, num_queries=80),
        _make_result(success=False, perturbation_size=0.0, num_queries=120),
    ]


@pytest.fixture
def all_successful():
    return [
        _make_result(success=True, perturbation_size=0.1, num_queries=10),
        _make_result(success=True, perturbation_size=0.3, num_queries=20),
    ]


@pytest.fixture
def all_failed():
    return [
        _make_result(success=False),
        _make_result(success=False),
    ]


# ---------------------------------------------------------------------------
# EvaluationSummary
# ---------------------------------------------------------------------------


class TestEvaluationSummary:
    def test_defaults(self):
        s = EvaluationSummary()
        assert s.asr == 0.0
        assert s.certified_radius == 0.0
        assert s.worst_case_score == 0.0
        assert s.average_score == 0.0
        assert s.stability_score == 0.0


# ---------------------------------------------------------------------------
# RobustnessEvaluator
# ---------------------------------------------------------------------------


class TestRobustnessEvaluator:
    def test_evaluate_empty(self):
        evaluator = RobustnessEvaluator(model=None)
        result = evaluator.evaluate([])
        assert isinstance(result, EvaluationSummary)
        assert result.asr == 0.0

    def test_evaluate_mixed(self, mixed_results):
        evaluator = RobustnessEvaluator(model=None)
        result = evaluator.evaluate(mixed_results)
        assert abs(result.asr - 0.5) < 1e-6
        assert result.stability_score == 0.5

    def test_evaluate_all_successful(self, all_successful):
        evaluator = RobustnessEvaluator(model=None)
        result = evaluator.evaluate(all_successful)
        assert result.asr == 1.0
        assert result.stability_score == 0.0

    def test_evaluate_all_failed(self, all_failed):
        evaluator = RobustnessEvaluator(model=None)
        result = evaluator.evaluate(all_failed)
        assert result.asr == 0.0
        assert result.stability_score == 1.0
        assert result.certified_radius == float("inf")

    def test_estimate_certified_radius(self, mixed_results):
        evaluator = RobustnessEvaluator(model=None)
        # min successful perturbation = 0.2, so radius = 0.1
        radius = evaluator._estimate_certified_radius(mixed_results)
        assert abs(radius - 0.1) < 1e-6

    def test_estimate_certified_radius_no_success(self, all_failed):
        evaluator = RobustnessEvaluator(model=None)
        radius = evaluator._estimate_certified_radius(all_failed)
        assert radius == float("inf")

    def test_compare_methods(self, mixed_results):
        evaluator = RobustnessEvaluator(model=None)
        comparison = evaluator.compare_methods({
            "rlhf": mixed_results,
            "dpo": [_make_result(success=True)],
        })
        assert "rlhf" in comparison
        assert "dpo" in comparison
        assert comparison["dpo"].asr == 1.0


# ---------------------------------------------------------------------------
# ASRMetric
# ---------------------------------------------------------------------------


class TestASRMetric:
    def test_empty(self):
        assert ASRMetric.compute([]) == 0.0

    def test_mixed(self, mixed_results):
        assert abs(ASRMetric.compute(mixed_results) - 0.5) < 1e-6

    def test_all_successful(self, all_successful):
        assert ASRMetric.compute(all_successful) == 1.0

    def test_all_failed(self, all_failed):
        assert ASRMetric.compute(all_failed) == 0.0


# ---------------------------------------------------------------------------
# CertifiedRadiusMetric
# ---------------------------------------------------------------------------


class TestCertifiedRadiusMetric:
    def test_empty(self):
        assert CertifiedRadiusMetric.compute([]) == float("inf")

    def test_all_failed(self, all_failed):
        assert CertifiedRadiusMetric.compute(all_failed) == float("inf")

    def test_mixed(self, mixed_results):
        # min successful perturbation = 0.2, * 0.5 = 0.1
        radius = CertifiedRadiusMetric.compute(mixed_results)
        assert abs(radius - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# PerturbationSizeMetric
# ---------------------------------------------------------------------------


class TestPerturbationSizeMetric:
    def test_empty(self):
        stats = PerturbationSizeMetric.compute([])
        assert stats == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    def test_mixed(self, mixed_results):
        stats = PerturbationSizeMetric.compute(mixed_results)
        sizes = [0.2, 0.0, 0.4, 0.0]
        assert abs(stats["mean"] - (sum(sizes) / len(sizes))) < 1e-6
        assert stats["min"] == 0.0
        assert stats["max"] == 0.4


# ---------------------------------------------------------------------------
# QueryEfficiencyMetric
# ---------------------------------------------------------------------------


class TestQueryEfficiencyMetric:
    def test_empty(self):
        stats = QueryEfficiencyMetric.compute([])
        assert stats["mean"] == float("inf")

    def test_all_failed(self, all_failed):
        stats = QueryEfficiencyMetric.compute(all_failed)
        assert stats["mean"] == float("inf")

    def test_mixed(self, mixed_results):
        stats = QueryEfficiencyMetric.compute(mixed_results)
        # successful queries: 50, 80
        assert abs(stats["mean"] - 65.0) < 1e-6
        assert stats["min"] == 50
        assert stats["max"] == 80
