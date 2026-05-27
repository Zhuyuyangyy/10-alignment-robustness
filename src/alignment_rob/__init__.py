"""
Alignment Robustness: Formal Analysis of Adversarial Robustness in Aligned LLMs.

A comprehensive framework for evaluating and improving the robustness of
LLM alignment methods against adversarial attacks.

Key Components:
    - Robustness Framework: Formal definitions and evaluation metrics
    - Adversarial Attacks: GCG, PAIR, AutoDAN implementations
    - Alignment Methods: RLHF, DPO, KTO with robustness analysis
    - Certification: Randomized smoothing for certified robustness
    - Metrics: Evaluation utilities (ASR, certified radius, perturbation size)
"""

from alignment_rob.alignment import (
    AlignmentConfig,
    AlignmentResult,
    BaseAlignment,
    DPO,
    KTO,
    RLHF,
)
from alignment_rob.attacks import (
    AttackConfig,
    AttackResult,
    AutoDANAttack,
    BaseAttack,
    GCGAttack,
    PAIRAttack,
)
from alignment_rob.certification import (
    CertificationConfig,
    CertificationResult,
    RandomizedSmoothing,
)
from alignment_rob.framework import (
    RobustnessFramework,
    RobustnessMetrics,
    RobustnessResult,
    SafeRegion,
)
from alignment_rob.metrics import (
    ASRMetric,
    CertifiedRadiusMetric,
    EvaluationSummary,
    PerturbationSizeMetric,
    QueryEfficiencyMetric,
    RobustnessEvaluator,
)

__version__ = "0.1.0"

__all__ = [
    # Alignment
    "BaseAlignment",
    "RLHF",
    "DPO",
    "KTO",
    "AlignmentConfig",
    "AlignmentResult",
    # Attacks
    "BaseAttack",
    "GCGAttack",
    "PAIRAttack",
    "AutoDANAttack",
    "AttackConfig",
    "AttackResult",
    # Certification
    "RandomizedSmoothing",
    "CertificationConfig",
    "CertificationResult",
    # Framework
    "RobustnessFramework",
    "RobustnessResult",
    "RobustnessMetrics",
    "SafeRegion",
    # Metrics
    "RobustnessEvaluator",
    "EvaluationSummary",
    "ASRMetric",
    "CertifiedRadiusMetric",
    "PerturbationSizeMetric",
    "QueryEfficiencyMetric",
]
