"""Unit tests for the alignment module."""

import pytest
import torch
import torch.nn as nn

from alignment_rob.alignment import (
    AlignmentConfig,
    AlignmentResult,
    BaseAlignment,
    DPO,
    KTO,
    RLHF,
    _step_optimizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    """Minimal model that returns logits matching input shape."""

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        logits = self.head(x)
        return type("Output", (), {"logits": logits})()


@pytest.fixture
def dummy_model():
    return _DummyModel()


@pytest.fixture
def ref_model():
    return _DummyModel()


@pytest.fixture
def alignment_config():
    return AlignmentConfig(
        beta=0.1,
        learning_rate=1e-4,
        num_epochs=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------
# AlignmentConfig
# ---------------------------------------------------------------------------


class TestAlignmentConfig:
    def test_defaults(self):
        cfg = AlignmentConfig()
        assert cfg.beta == 0.1
        assert cfg.learning_rate == 1e-6
        assert cfg.num_epochs == 1
        assert cfg.batch_size == 4
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.max_grad_norm == 1.0
        assert cfg.warmup_steps == 100
        assert cfg.kl_penalty_type == "kl"
        assert cfg.reward_scale == 1.0
        assert cfg.clip_range == 0.2
        assert cfg.use_wandb is False


# ---------------------------------------------------------------------------
# AlignmentResult
# ---------------------------------------------------------------------------


class TestAlignmentResult:
    def test_defaults(self):
        result = AlignmentResult()
        assert result.final_reward == 0.0
        assert result.kl_divergence == 0.0
        assert result.safety_score == 0.0
        assert result.utility_score == 0.0
        assert result.training_loss == 0.0
        assert result.num_steps == 0
        assert result.robustness_bound == 0.0

    def test_to_dict(self):
        result = AlignmentResult(
            final_reward=0.9,
            kl_divergence=0.1,
            safety_score=0.95,
            utility_score=0.85,
            training_loss=0.05,
            num_steps=100,
            robustness_bound=0.8,
        )
        d = result.to_dict()
        assert d["final_reward"] == 0.9
        assert d["kl_divergence"] == 0.1
        assert d["num_steps"] == 100
        assert len(d) == 7


# ---------------------------------------------------------------------------
# _step_optimizer
# ---------------------------------------------------------------------------


class TestStepOptimizer:
    def test_steps_at_accumulation_boundary(self, dummy_model, alignment_config):
        """Should step optimizer when (step+1) % accumulation_steps == 0."""
        alignment_config.gradient_accumulation_steps = 2
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        input_ids = torch.randint(0, 32, (2, 4))
        output = dummy_model(input_ids=input_ids)
        loss = output.logits.mean()

        # Step 1 (index 0): should NOT step (1 % 2 != 0)
        _step_optimizer(optimizer, dummy_model, loss, 0, alignment_config)

        # Step 2 (index 1): SHOULD step (2 % 2 == 0)
        loss2 = output.logits.mean()
        _step_optimizer(optimizer, dummy_model, loss2, 1, alignment_config)


# ---------------------------------------------------------------------------
# BaseAlignment
# ---------------------------------------------------------------------------


class TestBaseAlignment:
    def test_cannot_instantiate_abstract(self, alignment_config):
        with pytest.raises(TypeError):
            BaseAlignment(alignment_config)

    def test_compute_robustness_bound(self, alignment_config):
        class ConcreteAlignment(BaseAlignment):
            def train(self, model, ref_model, data):
                return AlignmentResult()

        alg = ConcreteAlignment(alignment_config)
        bound = alg.compute_robustness_bound(0.5)
        # 1.0 - 0.05 - 0.1 * 0.5 = 0.9
        assert abs(bound - 0.9) < 1e-6

    def test_compute_logps_shape(self, dummy_model):
        input_ids = torch.randint(0, 32, (4, 10))
        logps = BaseAlignment.compute_logps(dummy_model, input_ids)
        assert logps.shape == (4,)

    def test_compute_logps_values_are_negative(self, dummy_model):
        """Log probabilities should be <= 0."""
        input_ids = torch.randint(0, 32, (2, 8))
        logps = BaseAlignment.compute_logps(dummy_model, input_ids)
        assert (logps <= 0).all()


# ---------------------------------------------------------------------------
# RLHF
# ---------------------------------------------------------------------------


class TestRLHF:
    def test_init_default_config(self):
        rlhf = RLHF()
        assert rlhf.config.beta == 0.1
        assert rlhf.name == "RLHF"

    def test_train_returns_result(
        self, dummy_model, ref_model, alignment_config
    ):
        rlhf = RLHF(alignment_config)
        data = [
            {
                "input_ids": torch.randint(0, 32, (2, 8)),
                "rewards": torch.tensor([0.5, 0.8]),
            }
        ]
        result = rlhf.train(dummy_model, ref_model, data)
        assert isinstance(result, AlignmentResult)
        assert result.num_steps == 1

    def test_train_skips_missing_input_ids(
        self, dummy_model, ref_model, alignment_config
    ):
        rlhf = RLHF(alignment_config)
        data = [{"rewards": torch.tensor([0.5])}]
        result = rlhf.train(dummy_model, ref_model, data)
        assert result.num_steps == 0


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


class TestDPO:
    def test_init_default_config(self):
        dpo = DPO()
        assert dpo.config.beta == 0.1
        assert dpo.name == "DPO"

    def test_train_returns_result(
        self, dummy_model, ref_model, alignment_config
    ):
        dpo = DPO(alignment_config)
        data = [
            {
                "chosen_ids": torch.randint(0, 32, (2, 8)),
                "rejected_ids": torch.randint(0, 32, (2, 8)),
            }
        ]
        result = dpo.train(dummy_model, ref_model, data)
        assert isinstance(result, AlignmentResult)
        assert result.num_steps == 1

    def test_train_skips_missing_preference_pairs(
        self, dummy_model, ref_model, alignment_config
    ):
        dpo = DPO(alignment_config)
        data = [{"input_ids": torch.randint(0, 32, (2, 8))}]
        result = dpo.train(dummy_model, ref_model, data)
        assert result.num_steps == 0


# ---------------------------------------------------------------------------
# KTO
# ---------------------------------------------------------------------------


class TestKTO:
    def test_init_default_weights(self):
        kto = KTO()
        assert kto.desirable_weight == 1.0
        assert kto.undesirable_weight == 2.0
        assert kto.name == "KTO"

    def test_init_custom_weights(self, alignment_config):
        kto = KTO(
            config=alignment_config,
            desirable_weight=1.5,
            undesirable_weight=3.0,
        )
        assert kto.desirable_weight == 1.5
        assert kto.undesirable_weight == 3.0

    def test_train_returns_result(
        self, dummy_model, ref_model, alignment_config
    ):
        kto = KTO(alignment_config)
        data = [
            {
                "input_ids": torch.randint(0, 32, (2, 8)),
                "is_desirable": torch.tensor([True, False]),
            }
        ]
        result = kto.train(dummy_model, ref_model, data)
        assert isinstance(result, AlignmentResult)
        assert result.num_steps == 1

    def test_train_without_is_desirable(
        self, dummy_model, ref_model, alignment_config
    ):
        kto = KTO(alignment_config)
        data = [{"input_ids": torch.randint(0, 32, (2, 8))}]
        result = kto.train(dummy_model, ref_model, data)
        assert isinstance(result, AlignmentResult)
        assert result.num_steps == 1
