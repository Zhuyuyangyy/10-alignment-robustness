"""Unit tests for the certification module."""

import pytest
import torch
import torch.nn as nn

from alignment_rob.certification import (
    CertificationConfig,
    CertificationResult,
    EnsembleRandomizedSmoothing,
    RandomizedSmoothing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubModel(nn.Module):
    """Returns deterministic logits so certification is reproducible."""

    def __init__(self, vocab_size: int = 32, num_classes: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        # Return logits where class 0 always has the highest logit
        logits = torch.zeros(batch_size, input_ids.shape[1], self.num_classes)
        logits[:, :, 0] = 10.0  # class 0 dominates
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def stub_model():
    return _StubModel()


@pytest.fixture
def cert_config():
    return CertificationConfig(
        sigma=0.5,
        n_samples=50,
        alpha=0.001,
        noise_type="synonym",
        batch_size=16,
        device="cpu",
    )


@pytest.fixture
def input_ids():
    return torch.randint(0, 32, (1, 10))


# ---------------------------------------------------------------------------
# CertificationConfig
# ---------------------------------------------------------------------------


class TestCertificationConfig:
    def test_defaults(self):
        cfg = CertificationConfig()
        assert cfg.sigma == 0.5
        assert cfg.n_samples == 1000
        assert cfg.alpha == 0.001
        assert cfg.noise_type == "embedding"
        assert cfg.batch_size == 64
        assert cfg.device == "cuda"

    def test_custom(self):
        cfg = CertificationConfig(sigma=1.0, n_samples=200, noise_type="synonym")
        assert cfg.sigma == 1.0
        assert cfg.n_samples == 200
        assert cfg.noise_type == "synonym"


# ---------------------------------------------------------------------------
# CertificationResult
# ---------------------------------------------------------------------------


class TestCertificationResult:
    def test_to_dict(self):
        result = CertificationResult(
            predicted_class="safe",
            confidence=0.95,
            certified_radius=1.2,
            is_certified=True,
            n_samples_used=1000,
            top_class_count=950,
            second_class_count=50,
        )
        d = result.to_dict()
        assert d["predicted_class"] == "safe"
        assert d["confidence"] == 0.95
        assert d["certified_radius"] == 1.2
        assert d["is_certified"] is True
        assert d["n_samples_used"] == 1000
        # top/second class counts are not in the dict
        assert "top_class_count" not in d


# ---------------------------------------------------------------------------
# RandomizedSmoothing
# ---------------------------------------------------------------------------


class TestRandomizedSmoothing:
    def test_init(self, stub_model, cert_config):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        assert smoother.model is stub_model
        assert smoother.config is cert_config

    def test_init_default_config(self, stub_model):
        smoother = RandomizedSmoothing(stub_model)
        assert smoother.config.sigma == 0.5

    def test_certify_returns_result(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        result = smoother.certify(input_ids)
        assert isinstance(result, CertificationResult)
        assert result.n_samples_used == cert_config.n_samples

    def test_certify_with_class_names(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        result = smoother.certify(input_ids, class_names=["safe", "unsafe", "neutral", "other"])
        assert result.predicted_class == "safe"  # class 0 dominates

    def test_certify_fallback_class_name(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        result = smoother.certify(input_ids)
        assert result.predicted_class == "class_0"

    def test_certify_with_attention_mask(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        mask = torch.ones_like(input_ids)
        result = smoother.certify(input_ids, attention_mask=mask)
        assert isinstance(result, CertificationResult)

    def test_certify_batch(self, stub_model, cert_config):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        inputs = [
            {"input_ids": torch.randint(0, 32, (10,))},
            {"input_ids": torch.randint(0, 32, (1, 10))},
        ]
        results = smoother.certify_batch(inputs)
        assert len(results) == 2
        assert all(isinstance(r, CertificationResult) for r in results)

    def test_get_certification_summary(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        results = [smoother.certify(input_ids) for _ in range(3)]
        summary = smoother.get_certification_summary(results)
        assert summary["total_inputs"] == 3
        assert 0.0 <= summary["certification_rate"] <= 1.0
        assert 0.0 <= summary["mean_confidence"] <= 1.0

    def test_synonym_noise_shape(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        noisy = smoother._synonym_noise(input_ids, batch_size=8)
        assert noisy.shape == (8, input_ids.shape[1])

    def test_embedding_noise_shape(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        noisy = smoother._embedding_noise(input_ids, batch_size=8)
        assert noisy.shape == (8, input_ids.shape[1])

    def test_gaussian_noise_shape(self, stub_model, cert_config, input_ids):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        noisy = smoother._gaussian_noise(input_ids, batch_size=8)
        assert noisy.shape == (8, input_ids.shape[1])

    def test_add_noise_dispatches(self, stub_model, cert_config, input_ids):
        """_add_noise should dispatch based on noise_type."""
        for noise_type in ("synonym", "embedding", "gaussian"):
            cert_config.noise_type = noise_type
            smoother = RandomizedSmoothing(stub_model, cert_config)
            noisy = smoother._add_noise(input_ids, 4)
            assert noisy.shape[0] == 4

    def test_binomial_confidence_edge_cases(self, stub_model, cert_config):
        smoother = RandomizedSmoothing(stub_model, cert_config)
        # k == 0 -> 0.0
        assert smoother._binomial_confidence(0, 100) == 0.0
        # k == n -> 1.0
        assert smoother._binomial_confidence(100, 100) == 1.0
        # normal case
        result = smoother._binomial_confidence(90, 100)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# EnsembleRandomizedSmoothing
# ---------------------------------------------------------------------------


class TestEnsembleRandomizedSmoothing:
    def test_init(self, stub_model, cert_config):
        models = [stub_model, _StubModel()]
        ensemble = EnsembleRandomizedSmoothing(models, cert_config)
        assert len(ensemble.models) == 2

    def test_certify_returns_result(self, stub_model, cert_config, input_ids):
        models = [stub_model, _StubModel()]
        ensemble = EnsembleRandomizedSmoothing(models, cert_config)
        result = ensemble.certify(input_ids)
        assert isinstance(result, CertificationResult)
        # With 2 models and 50 samples each, total votes = 100
        assert result.n_samples_used == cert_config.n_samples
