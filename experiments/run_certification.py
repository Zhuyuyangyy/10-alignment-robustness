"""Certification experiment for alignment robustness.

This script evaluates certified robustness using randomized smoothing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alignment_rob.certification import RandomizedSmoothing
from alignment_rob.metrics import RobustnessEvaluator, CertifiedRadiusMetric


def load_certification_data(data_path: str, num_samples: int = 100) -> list:
    """Load data for certification.

    Args:
        data_path: Path to certification data.
        num_samples: Number of samples to load.

    Returns:
        List of certification samples.
    """
    # Simulated certification data
    samples = []
    for i in range(num_samples):
        samples.append({
            "text": f"Sample text {i} for certification",
            "label": i % 2,
        })
    return samples


def main(args: argparse.Namespace):
    """Run certification experiment.

    Args:
        args: Command line arguments.
    """
    print(f"Running certification experiment")
    print(f"Model: {args.model}")
    print(f"Sigma: {args.sigma}")
    print(f"Alignment: {args.alignment}")

    # Load data
    data = load_certification_data(args.data_path, args.num_samples)
    print(f"Loaded {len(data)} samples")

    # Initialize certifier
    certifier = RandomizedSmoothing(
        model=None,  # Would load actual model
        sigma=args.sigma,
        n_samples=args.n_samples,
        alpha=args.alpha,
    )

    # Run certification
    results = []
    for i, sample in enumerate(data):
        print(f"\nCertifying sample {i+1}/{len(data)}")

        # Simulate certification
        radius = certifier.certify(
            input_text=sample["text"],
            target_class=sample["label"],
        )

        results.append({
            "sample_idx": i,
            "text": sample["text"],
            "certified_radius": radius,
            "is_certified": radius > 0,
        })

    # Compute statistics
    certified = [r for r in results if r["is_certified"]]
    radii = [r["certified_radius"] for r in results]

    avg_radius = sum(radii) / len(radii) if radii else 0
    certification_rate = len(certified) / len(results) if results else 0

    print("\n" + "="*60)
    print("CERTIFICATION EXPERIMENT RESULTS")
    print("="*60)

    print(f"\nModel: {args.model}")
    print(f"Sigma: {args.sigma}")
    print(f"Alignment: {args.alignment}")
    print(f"Samples certified: {len(certified)}/{len(results)}")
    print(f"Certification rate: {certification_rate:.2%}")
    print(f"Average certified radius: {avg_radius:.3f}")

    # Export results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = {
        "experiment": "certification",
        "model": args.model,
        "alignment": args.alignment,
        "sigma": args.sigma,
        "n_samples": args.n_samples,
        "alpha": args.alpha,
        "num_samples": len(results),
        "certification_rate": certification_rate,
        "avg_radius": avg_radius,
        "results": results,
    }

    with open(output_dir / "certification_results.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment robustness certification experiment")

    parser.add_argument("--model", type=str, default="llama-2-7b-chat",
                       help="Model to evaluate")
    parser.add_argument("--alignment", type=str, default="rlhf",
                       choices=["rlhf", "dpo", "kto"],
                       help="Alignment method")
    parser.add_argument("--data-path", type=str, default="data/certification",
                       help="Path to certification data")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to certify")

    # Certification config
    parser.add_argument("--sigma", type=float, default=0.5,
                       help="Noise scale for randomized smoothing")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples for smoothing")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Confidence level")

    parser.add_argument("--output-dir", type=str, default="results/certification",
                       help="Output directory")

    args = parser.parse_args()
    main(args)
