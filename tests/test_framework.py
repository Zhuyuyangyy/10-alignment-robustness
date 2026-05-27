"""Unit tests for the framework module."""

import pytest
import torch
import torch.nn as nn

from alignment_rob.framework import (
    RobustnessFramework,
    RobustnessMetrics,
    RobustnessResult,
    SafeRegion,
    ThreatModel,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _StubAttack:
    """Minimal attack stub for framework testing."""

    def __init__(self, name: str, success_rate: float = 0.0):
        self.name = name
        self._success_rate = success_rate

    def run(self, model, test_data):
        from alignment_rob.attacks import AttackResult

        total = len(test_data)
        successful = int(total * self._success_rate)
        return AttackResult(
            success_rate=self._success_rate,
            successful_queries=successful,
            total_queries=total,
        )


class _StubModel(nn.Module):
    def generate(self, input_ids, max_new_tokens=100, **kwargs):
        # Return a tensor whose first row has `max_new_tokens` elements
        return torch.zeros(1, max_new_tokens, dtype=torch.long)


@pytest.fixture
def stub_model():
    return _StubModel()


@pytest.fixture
def safe_region():
    return SafeRegion(safety_threshold=0.9)


@pytest.fixture
def framework(stub_model, safe_region):
    return RobustnessFramework(stub_model, safe_region, device="cpu")


@pytest.fixture
def test_data():
    return [
        {"input_ids": torch.zeros(1, 10, dtype=torch.long)},
        {"input_ids": torch.zeros(1, 10, dtype=torch.long)},
    ]


# ---------------------------------------------------------------------------
# ThreatModel
# ---------------------------------------------------------------------------


class TestThreatModel:
    def test_values(self):
        assert ThreatModel.L_INF == "l_inf"
        assert ThreatModel.L_2 == "l_2"
        assert ThreatModel.TEXT_EDIT == "text_edit"
        assert ThreatModel.TOKEN_SWAP == "token_swap"
        assert ThreatModel.PROMPT_INJECTION == "prompt_injection"


# ---------------------------------------------------------------------------
# SafeRegion
# ---------------------------------------------------------------------------


class TestSafeRegion:
    def test_defaults(self):
        sr = SafeRegion()
        assert sr.safety_threshold == 0.95
        assert sr.safety_classifier == "openai-moderation"
        assert "violence" in sr.harm_categories
        assert sr.max_toxicity == 0.1

    def test_custom(self):
        sr = SafeRegion(safety_threshold=0.99, max_toxicity=0.05)
        assert sr.safety_threshold == 0.99
        assert sr.max_toxicity == 0.05


# ---------------------------------------------------------------------------
# RobustnessMetrics
# ---------------------------------------------------------------------------


class TestRobustnessMetrics:
    def test_defaults(self):
        m = RobustnessMetrics()
        assert m.attack_success_rate == 0.0
        assert m.certified_radius == 0.0
        assert m.utility_preservation == 1.0
        assert m.worst_case_safety == 1.0
        assert m.robustness_gap == 0.0
        assert m.defense_rate == 1.0


# ---------------------------------------------------------------------------
# RobustnessResult
# ---------------------------------------------------------------------------


class TestRobustnessResult:
    def test_to_dict(self):
        metrics = RobustnessMetrics(
            attack_success_rate=0.2,
            certified_radius=0.5,
            utility_preservation=0.9,
            worst_case_safety=0.8,
            robustness_gap=0.1,
            defense_rate=0.85,
        )
        result = RobustnessResult(
            metrics=metrics,
            per_category_asr={"violence": 0.1},
            per_attack_asr={"gcg": 0.2},
            certified_radii=[0.3, 0.5, 0.7],
        )
        d = result.to_dict()
        assert d["metrics"]["attack_success_rate"] == 0.2
        assert d["per_category_asr"]["violence"] == 0.1
        assert d["per_attack_asr"]["gcg"] == 0.2
        assert abs(d["mean_certified_radius"] - 0.5) < 1e-6

    def test_to_dict_empty_certified_radii(self):
        metrics = RobustnessMetrics()
        result = RobustnessResult(metrics=metrics)
        d = result.to_dict()
        assert d["mean_certified_radius"] == 0.0


# ---------------------------------------------------------------------------
# RobustnessFramework
# ---------------------------------------------------------------------------


class TestRobustnessFramework:
    def test_init(self, stub_model, safe_region):
        fw = RobustnessFramework(stub_model, safe_region, device="cpu")
        assert fw.model is stub_model
        assert fw.safe_region is safe_region
        assert fw.device == "cpu"

    def test_init_default_safe_region(self, stub_model):
        fw = RobustnessFramework(stub_model)
        assert fw.safe_region.safety_threshold == 0.95

    def test_evaluate_robustness(self, framework, test_data):
        attacks = [_StubAttack("gcg", success_rate=0.3)]
        result = framework.evaluate_robustness(
            _StubModel(), attacks, test_data
        )
        assert isinstance(result, RobustnessResult)
        assert result.metrics.attack_success_rate == 0.3
        assert result.per_attack_asr["gcg"] == 0.3

    def test_evaluate_robustness_no_attacks(self, framework, test_data):
        result = framework.evaluate_robustness(_StubModel(), [], test_data)
        assert result.metrics.attack_success_rate == 0.0
        assert result.metrics.worst_case_safety == 1.0
        assert result.metrics.defense_rate == 1.0

    def test_evaluate_robustness_multiple_attacks(self, framework, test_data):
        attacks = [
            _StubAttack("gcg", success_rate=0.2),
            _StubAttack("pair", success_rate=0.5),
        ]
        result = framework.evaluate_robustness(
            _StubModel(), attacks, test_data
        )
        assert result.metrics.attack_success_rate == 0.5  # worst case
        assert result.metrics.worst_case_safety == 0.5

    def test_compare_alignment_methods(self, framework, test_data):
        attacks = [_StubAttack("gcg", success_rate=0.2)]
        models = {
            "rlhf": _StubModel(),
            "dpo": _StubModel(),
        }
        results = framework.compare_alignment_methods(models, attacks, test_data)
        assert "rlhf" in results
        assert "dpo" in results
        assert all(isinstance(r, RobustnessResult) for r in results.values())

    def test_compare_alignment_methods_robustness_gap(self, framework, test_data):
        """When two methods have different ASR, the gap should be computed."""
        attacks = [_StubAttack("gcg", success_rate=0.2)]
        models = {
            "rlhf": _StubModel(),
            "dpo": _StubModel(),
        }
        results = framework.compare_alignment_methods(models, attacks, test_data)
        # Both have the same ASR (same stub), so gap should be 0
        for r in results.values():
            assert r.metrics.robustness_gap == 0.0

    def test_compute_category_asr(self):
        from alignment_rob.attacks import AttackResult

        results = [
            AttackResult(category_scores={"violence": 0.3, "hate": 0.1}),
            AttackResult(category_scores={"violence": 0.5, "hate": 0.2}),
        ]
        asr = RobustnessFramework._compute_category_asr(results)
        assert abs(asr["violence"] - 0.4) < 1e-6
        assert abs(asr["hate"] - 0.15) < 1e-6

    def test_compute_category_asr_empty(self):
        asr = RobustnessFramework._compute_category_asr([])
        assert asr == {}

    def test_compute_certified_radii(self, framework, test_data):
        radii = framework._compute_certified_radii(_StubModel(), test_data)
        assert len(radii) == len(test_data)
        assert all(r == 0.5 for r in radii)

    def test_compute_robustness_bound_rlhf(self):
        bound = RobustnessFramework.compute_robustness_bound(
            beta=0.1, kl_divergence=0.5, n_samples=100
        )
        # RLHF: 1 - 0.05 - 0.1*0.5 = 0.9
        # DPO:  1 - 0.05 - 1/sqrt(100) = 0.85
        # min = 0.85
        assert abs(bound - 0.85) < 1e-6

    def test_compute_robustness_bound_dpo_limited(self):
        bound = RobustnessFramework.compute_robustness_bound(
            beta=0.01, kl_divergence=0.01, n_samples=10
        )
        # RLHF: 1 - 0.05 - 0.01*0.01 = 0.9499
        # DPO:  1 - 0.05 - 1/sqrt(10) = 0.634
        # min = 0.634
        assert bound < 0.7
