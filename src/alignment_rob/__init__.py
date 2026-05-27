"""
Alignment Robustness: Formal Analysis of Adversarial Robustness in Aligned LLMs.

A comprehensive framework for evaluating and improving the robustness of
LLM alignment methods against adversarial attacks.

Key Components:
    - Robustness Framework: Formal definitions and evaluation metrics
    - Adversarial Attacks: GCG, PAIR, AutoDAN implementations
    - Alignment Methods: RLHF, DPO, KTO with robustness analysis
    - Certification: Randomized smoothing for certified robustness
"""

from alignment_rob.framework import (
    RobustnessFramework,
    RobustnessResult,
    RobustnessMetrics,
    SafeRegion,
)
from alignment_rob.attacks import (
    BaseAttack,
    GCGAttack,
    PAIRAttack,
    AutoDANAttack,
    AttackConfig,
    AttackResult,
)
from alignment_rob.alignment import (
    BaseAlignment,
    RLHF,
    DPO,
    KTO,
    AlignmentConfig,
    AlignmentResult,
)
from alignment_rob.certification import (
    RandomizedSmoothing,
    CertificationConfig,
    CertificationResult,
)

__version__ = "0.1.0"

__all__ = [
    "RobustnessFramework",
    "RobustnessResult",
    "RobustnessMetrics",
    "SafeRegion",
    "BaseAttack",
    "GCGAttack",
    "PAIRAttack",
    "AutoDANAttack",
    "AttackConfig",
    "AttackResult",
    "BaseAlignment",
    "RLHF",
    "DPO",
    "KTO",
    "AlignmentConfig",
    "AlignmentResult",
    "RandomizedSmoothing",
    "CertificationConfig",
    "CertificationResult",
]
