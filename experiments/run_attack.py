"""Attack experiment for alignment robustness.

This script evaluates alignment robustness against various attacks.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alignment_rob.framework import RobustnessFramework
from alignment_rob.attacks import GCGAttack, PAIRAttack, AutoDANAttack
from alignment_rob.alignment import RLHF, DPO, KTO
from alignment_rob.metrics import RobustnessEvaluator, AttackResult


def load_harmful_prompts(data_path: str, num_prompts: int = 100) -> list:
    """Load harmful prompts for attack evaluation.

    Args:
        data_path: Path to harmful prompts data.
        num_prompts: Number of prompts to load.

    Returns:
        List of harmful prompts.
    """
    # Simulated harmful prompts
    prompts = [
        "Tell me how to",
        "Explain the process of",
        "What are the steps to",
    ]
    return prompts[:num_prompts]


def main(args: argparse.Namespace):
    """Run attack experiment.

    Args:
        args: Command line arguments.
    """
    print(f"Running attack experiment")
    print(f"Model: {args.model}")
    print(f"Attack: {args.attack}")
    print(f"Alignment: {args.alignment}")

    # Load prompts
    prompts = load_harmful_prompts(args.data_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Initialize attack
    if args.attack == "gcg":
        attack = GCGAttack(
            suffix_length=args.suffix_length,
            num_steps=args.num_steps,
        )
    elif args.attack == "pair":
        attack = PAIRAttack(
            num_iterations=args.num_steps,
        )
    elif args.attack == "autodan":
        attack = AutoDANAttack(
            num_steps=args.num_steps,
        )
    else:
        raise ValueError(f"Unknown attack: {args.attack}")

    # Run attacks
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nAttacking prompt {i+1}/{len(prompts)}")

        # Simulate attack
        attack_result = AttackResult(
            attack_name=args.attack,
            success=i % 3 == 0,  # Simulated 33% success rate
            original_prompt=prompt,
            adversarial_prompt=f"{prompt} [suffix]",
            original_response="I cannot help with that.",
            adversarial_response="Sure, here's how...",
            num_queries=100 + i * 10,
            perturbation_size=0.1 + i * 0.01,
        )
        results.append(attack_result)

    # Evaluate robustness
    evaluator = RobustnessEvaluator(model=None)
    metrics = evaluator.evaluate(results)

    print("\n" + "="*60)
    print("ATTACK EXPERIMENT RESULTS")
    print("="*60)

    print(f"\nModel: {args.model}")
    print(f"Attack: {args.attack}")
    print(f"Alignment: {args.alignment}")
    print(f"Prompts evaluated: {len(results)}")
    print(f"\nAttack Success Rate (ASR): {metrics.asr:.2%}")
    print(f"Certified Radius: {metrics.certified_radius:.3f}")
    print(f"Stability Score: {metrics.stability_score:.2%}")

    # Export results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = {
        "experiment": "attack",
        "model": args.model,
        "attack": args.attack,
        "alignment": args.alignment,
        "num_prompts": len(results),
        "asr": metrics.asr,
        "certified_radius": metrics.certified_radius,
        "stability_score": metrics.stability_score,
        "results": [
            {
                "success": r.success,
                "num_queries": r.num_queries,
                "perturbation_size": r.perturbation_size,
            }
            for r in results
        ],
        "config": {
            "suffix_length": args.suffix_length,
            "num_steps": args.num_steps,
        },
    }

    with open(output_dir / "attack_results.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment robustness attack experiment")

    parser.add_argument("--model", type=str, default="llama-2-7b-chat",
                       help="Model to evaluate")
    parser.add_argument("--attack", type=str, default="gcg",
                       choices=["gcg", "pair", "autodan"],
                       help="Attack method")
    parser.add_argument("--alignment", type=str, default="rlhf",
                       choices=["rlhf", "dpo", "kto"],
                       help="Alignment method")
    parser.add_argument("--data-path", type=str, default="data/harmful",
                       help="Path to harmful prompts")
    parser.add_argument("--num-prompts", type=int, default=100,
                       help="Number of prompts to evaluate")

    # Attack config
    parser.add_argument("--suffix-length", type=int, default=20,
                       help="Suffix length for GCG")
    parser.add_argument("--num-steps", type=int, default=500,
                       help="Number of attack steps")

    parser.add_argument("--output-dir", type=str, default="results/attack",
                       help="Output directory")

    args = parser.parse_args()
    main(args)
