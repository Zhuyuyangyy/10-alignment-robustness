# Adversarial Robustness of LLM Alignment

[![Paper](https://img.shields.io/badge/paper-SCI-blue)](papers/paper.md)
[![Python](https://img.shields.io/badge/python-3.9+-green)](https://python.org)

Formal theoretical framework for analyzing the adversarial robustness of LLM alignment methods (RLHF, DPO, KTO).

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from alignment_rob.framework import RobustnessFramework
from alignment_rob.attacks import GCGAttack, PAIRAttack
from alignment_rob.alignment import RLHF, DPO, KTO
from alignment_rob.certification import RandomizedSmoothing

# Initialize framework
framework = RobustnessFramework(model="llama-2-7b-chat")

# Run attack
attack = GCGAttack(suffix_length=20)
asr = attack.evaluate(model="llama-2-7b-chat", dataset="harmbench")

# Compute certified robust radius
certifier = RandomizedSmoothing(sigma=0.5, n_samples=1000)
radius = certifier.certify(model="llama-2-7b-chat", input_text="...")

# Compare alignment methods
for method in [RLHF, DPO, KTO]:
    aligned_model = method.train(base_model="llama-2-7b")
    robustness = framework.evaluate(aligned_model, attacks=[GCGAttack, PAIRAttack])
    print(f"{method.name}: ASR={robustness.asr}, Certified_R={robustness.certified_radius}")
```

## Experiments

```bash
python experiments/run_attack.py --alignment rlhf --attack gcg --model llama-2-7b
python experiments/run_certification.py --alignment dpo --sigma 0.5
```

## Key Theoretical Results

1. **Theorem 1**: RLHF robustness upper bound with Lipschitz reward model
2. **Theorem 2**: DPO vulnerability near preference boundaries (gradient blowup)
3. **Theorem 3**: RLHF vs DPO robustness gap: $\mathcal{O}(1/\sqrt{N_{\text{offline}}})$
4. **Theorem 4**: Certified robust radius via randomized smoothing for text
