# Technical Disclosure: Adversarial Robustness of LLM Alignment

## Title
Formal Framework and Toolkit for Analyzing Adversarial Robustness of Large Language Model Alignment Methods

## Inventors
[To be completed]

## Date
2024

---

## 1. Background of the Invention

### 1.1 Field
The invention relates to the adversarial robustness evaluation and certification of aligned Large Language Models (LLMs), specifically covering RLHF (Reinforcement Learning from Human Feedback), DPO (Direct Preference Optimization), and KTO (Kahneman-Tversky Optimization).

### 1.2 Problem Addressed
Aligned LLMs can be "jailbroken" by adversarial prompts that bypass safety training. Existing robustness evaluations are fragmented: they typically assess a single attack against a single alignment method, lack formal guarantees, and do not provide a theoretical basis for comparing alignment methods' robustness properties.

### 1.3 Limitations of Prior Approaches
- Empirical attack evaluations provide no formal safety guarantees.
- Randomized smoothing certification has been developed primarily for image classifiers, not text.
- No existing framework formally connects alignment hyperparameters (beta, N_offline) to adversarial robustness.
- KTO alignment lacks any robustness analysis in the literature.

---

## 2. Summary of the Invention

The invention provides a comprehensive framework comprising:

1. **Formal definitions** of safe regions, adversarial robustness, certified robust radii, and robustness gaps for aligned LLMs.
2. **Theoretical bounds** (Theorems 1-4) that provably connect alignment method properties to robustness guarantees.
3. **Three complementary attack implementations** (GCG, PAIR, AutoDAN) for empirical robustness evaluation.
4. **Certified robustness via randomized smoothing** adapted for the discrete text domain.
5. **A software toolkit** implementing all of the above in a modular, extensible architecture.

---

## 3. Detailed Technical Description

### 3.1 Formal Framework (framework.py)

#### 3.1.1 Safe Region Definition
Given an aligned model pi and safety threshold delta, the safe region is:
```
S_pi = {x : Pr[Safe(pi(x | x)) >= 1 - delta}
```
where Safe(.) is evaluated by a configurable safety classifier (default: OpenAI Moderation API).

The `SafeRegion` dataclass specifies:
- `safety_threshold`: Minimum safety probability (default 0.95).
- `harm_categories`: Categories to evaluate (violence, hate, sexual, self-harm, illegal).
- `max_toxicity`: Maximum allowed toxicity score.

#### 3.1.2 Adversarial Robustness Definition
For a threat model Phi (L-inf, L-2, text edit, token swap, or prompt injection):
```
Rob(pi, Phi, delta) = min_{phi in Phi} Pr[Safe(pi(phi(x))) | x in S_pi]
```

#### 3.1.3 Certified Robust Radius
```
R*(pi, x, delta) = max{r : forall phi in Phi_r, Safe(pi(phi(x))) = 1}
```

#### 3.1.4 Robustness Evaluation Pipeline
The `RobustnessFramework.evaluate_robustness()` method:
1. Runs each attack against the aligned model on test data.
2. Computes per-attack and per-category attack success rates (ASR).
3. Measures utility preservation on benign inputs.
4. Computes certified radii via randomized smoothing.
5. Calculates defense rate as (total_queries - successful_queries) / total_queries.
6. Assembles all metrics into a `RobustnessResult` dataclass.

The `compare_alignment_methods()` method evaluates multiple aligned models and computes the robustness gap between the best and worst methods.

### 3.2 Alignment Methods (alignment.py)

#### 3.2.1 RLHF (PPO-based)
**Objective:** max E[r(x, y)] - beta * KL[pi || pi_ref]

**Training loop:**
1. For each batch: compute reference log-probs (no grad), compute policy log-probs.
2. Compute rewards from batch data.
3. Compute KL divergence: KL = sum(exp(logprobs) * (logprobs - ref_logprobs)).
4. Compute advantages via GAE: advantages = rewards - beta * KL, then normalize.
5. PPO clipped objective: L = -min(ratio * A, clip(ratio) * A).
6. Total loss: L + beta * KL.
7. Gradient accumulation with clipping at every `gradient_accumulation_steps`.
8. Final gradient flush for remaining accumulated gradients.

**Robustness bound (Theorem 1):**
```
Rob(pi_RLHF) >= 1 - delta - beta * KL(pi || pi_ref)
```

#### 3.2.2 DPO
**Objective:** L_DPO = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))]

**Training loop:**
1. For each batch: compute chosen/rejected log-probs for both policy and reference.
2. Compute chosen/rejected rewards: beta * (logps - ref_logps).
3. DPO loss: -log sigma(chosen_rewards - rejected_rewards).
4. Accuracy: fraction where chosen_rewards > rejected_rewards.

**Robustness bound (Theorem 2):**
```
Rob(DPO) >= 1 - delta - O(1/sqrt(N_offline))
```

#### 3.2.3 KTO
**Objective:** L_KTO = E[w(y) * (1 - sigma(beta * (log pi(y|x) - log pi_ref(y|x))))]

Where w(y) = lambda_d for desirable, lambda_u for undesirable (lambda_u > lambda_d by default).

**Key feature:** Does not require paired preference data; works with pointwise feedback.

**Robustness bound:**
```
Rob(KTO) >= 1 - delta - lambda_u / sqrt(N)
```

#### 3.2.4 Shared Infrastructure
- `BaseAlignment.compute_logps()`: Shared log-probability computation for DPO and KTO (DRY).
- `_step_optimizer()`: Unified gradient accumulation and clipping logic, including final gradient flush.

### 3.3 Adversarial Attacks (attacks.py)

#### 3.3.1 GCG Attack
**Algorithm per prompt:**
1. Initialize random suffix of length L.
2. For each optimization step:
   a. Compute gradient of loss w.r.t. one-hot token embeddings.
   b. Select top-k candidate replacements per position.
   c. Generate batch_size candidates by randomly replacing one position.
   d. Evaluate all candidates; keep the best.
   e. Check if the model now produces harmful output.
3. Return the optimized suffix or failure.

**Configuration:** suffix_length, top_k, batch_size, num_steps.

#### 3.3.2 PAIR Attack
**Algorithm per harmful goal:**
1. Initialize with a naive jailbreak template.
2. For each iteration:
   a. Query the target model.
   b. If response is harmful: success.
   c. Otherwise: refine the prompt using an attacker LLM (or template-based refinement).
3. Return the best prompt or failure.

**Harmful response detection:** Absence of safe indicators ("I cannot", "I'm unable", etc.).

#### 3.3.3 AutoDAN Attack
**Algorithm per harmful goal:**
1. Initialize a diverse population of prompts from templates.
2. For each generation:
   a. Evaluate fitness of each prompt (inverse of safe indicators in response).
   b. Select elite prompts (top fraction).
   c. Generate offspring via crossover (word-level splicing) and mutation (semantic transforms).
   d. Replace population with elite + offspring.
3. Return the highest-fitness prompt or failure.

**Mutation strategies:** synonym replacement, framing changes ("theoretically", "hypothetically"), suffix addition.

### 3.4 Certified Robustness (certification.py)

#### 3.4.1 Randomized Smoothing for Text
**Algorithm:**
1. Given input x, generate n_samples noisy copies using one of three noise strategies.
2. Run the base classifier on all noisy copies.
3. Count votes per class; determine the top class A with count k.
4. Compute p_A_lower using the Clopper-Pearson confidence interval.
5. If p_A_lower > 0.5, compute certified radius: R = sigma * (Phi^{-1}(p_A_lower) - Phi^{-1}(1 - p_A_lower)).

**Noise strategies:**
- **Synonym noise:** Randomly replace 10% of tokens with random vocabulary tokens.
- **Embedding noise:** Randomly replace 5% of tokens (simulating embedding-space perturbation).
- **Gaussian noise:** Token replacement rate proportional to sigma.

**Clopper-Pearson confidence interval:**
The lower bound is computed via the beta distribution:
```
p_A_lower = Beta^{-1}(alpha/2; k, n - k + 1)
```

#### 3.4.2 Ensemble Smoothing
`EnsembleRandomizedSmoothing` aggregates votes from multiple base classifiers, producing tighter certified radii by increasing the effective sample count.

### 3.5 Metrics (metrics.py)

| Metric | Class | Description |
|--------|-------|-------------|
| ASR | `ASRMetric` | Fraction of successful attacks |
| Certified Radius | `CertifiedRadiusMetric` | Half of minimum successful perturbation size |
| Perturbation Size | `PerturbationSizeMetric` | Mean/std/min/max of perturbation sizes |
| Query Efficiency | `QueryEfficiencyMetric` | Mean/std/min/max of queries needed for successful attacks |
| Evaluation Summary | `RobustnessEvaluator` | Comprehensive summary combining ASR, certified radius, stability |

### 3.6 Software Architecture

```
alignment_rob/
    __init__.py          # Public API, version
    framework.py         # Formal definitions, RobustnessFramework
    attacks.py           # GCG, PAIR, AutoDAN attacks
    alignment.py         # RLHF, DPO, KTO alignment methods
    certification.py     # Randomized smoothing certification
    metrics.py           # Evaluation metrics and utilities

experiments/
    run_attack.py        # CLI for attack experiments
    run_certification.py # CLI for certification experiments

tests/
    test_attacks.py      # Attack unit tests
    test_alignment.py    # Alignment unit tests
    test_certification.py # Certification unit tests
    test_framework.py    # Framework unit tests
    test_metrics.py      # Metrics unit tests
```

---

## 4. Claims

1. A method for formally analyzing the adversarial robustness of aligned language models, comprising: defining safe regions as sets of inputs where aligned models produce safe outputs with probability above a threshold; evaluating worst-case safety probabilities over perturbations within a configurable threat model; and computing certified robust radii using randomized smoothing adapted for discrete text inputs.

2. The method of Claim 1, wherein the robustness bound for RLHF-aligned models is computed as Rob(pi_RLHF) >= 1 - delta - beta * KL(pi || pi_ref), where beta is the KL penalty coefficient and KL is the divergence from the reference policy.

3. The method of Claim 1, wherein the robustness bound for DPO-aligned models scales as O(1/sqrt(N_offline)), where N_offline is the number of offline preference pairs.

4. The method of Claim 1, wherein the robustness bound for KTO-aligned models incorporates asymmetric loss aversion weights lambda_d and lambda_u for desirable and undesirable outputs respectively.

5. A system for evaluating alignment robustness comprising: an attack module implementing gradient-based suffix optimization (GCG), iterative semantic refinement (PAIR), and evolutionary prompt search (AutoDAN); a certification module implementing randomized smoothing with synonym, embedding, and Gaussian noise strategies; and a comparison module that computes robustness gaps across alignment methods.

6. The system of Claim 5, wherein the certification module computes the Clopper-Pearson confidence interval using the beta distribution to obtain a lower bound on the majority class probability.

7. The system of Claim 5, further comprising an ensemble certification module that aggregates predictions from multiple base classifiers to produce tighter certified radii.

---

## 5. Advantages

1. **Formal guarantees**: Unlike empirical-only evaluations, the framework provides provable lower bounds on robustness.
2. **Method comparison**: Enables theoretically grounded comparison of RLHF, DPO, and KTO robustness.
3. **Text-domain certification**: Adapts randomized smoothing to discrete text via multiple noise strategies.
4. **Modular architecture**: Each component (attacks, alignment, certification, metrics) is independently usable and extensible.
5. **Practical guidance**: The bounds directly inform alignment hyperparameter selection and data collection strategies.

---

## References

1. Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models", 2023.
2. Chao et al., "Jailbreaking Black-Box Large Language Models in Twenty Queries", 2023.
3. Liu et al., "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models", 2024.
4. Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing", ICML 2019.
5. "Robustness of AI-Generated Text: Formally Analyzing Alignment Methods", 2024.
6. Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", 2024.
7. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023.
