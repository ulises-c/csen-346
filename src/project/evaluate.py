"""
Evaluation runner — computes metrics after a batch evaluation completes.

Usage:
    python3 -m src.project.evaluate results/baseline
    python3 -m src.project.evaluate results/gemma4
    python3 -m src.project.evaluate --compare results/baseline results/gemma4
"""

import argparse
import json
from pathlib import Path

from src.project.metrics import compute_all_metrics, format_metrics_table


def evaluate_run(results_dir: Path) -> dict:
    """Compute and save metrics for a single evaluation run."""
    dialogues_dir = results_dir / "dialogues"

    if not dialogues_dir.exists():
        print(f"ERROR: No dialogues directory at {dialogues_dir}")
        return {}

    n_files = len(list(dialogues_dir.glob("*.json")))
    print(f"Computing metrics for {n_files} dialogues in {results_dir}...")

    metrics = compute_all_metrics(dialogues_dir)

    # Save metrics
    metrics_file = results_dir / "metrics_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {metrics_file}")

    # Print
    print(format_metrics_table(metrics))
    return metrics


def compare_runs(dirs: list[Path]) -> None:
    """Side-by-side comparison of multiple evaluation runs."""
    results = {}
    for d in dirs:
        metrics_file = d / "metrics_summary.json"
        if metrics_file.exists():
            results[d.name] = json.loads(metrics_file.read_text())
        else:
            print(f"No metrics found for {d}, computing...")
            results[d.name] = evaluate_run(d)

    if len(results) < 2:
        print("Need at least 2 runs to compare.")
        return

    # Build comparison table
    metric_keys = ["rouge1", "rouge2", "rougeL", "bleu4"]
    header = f"{'Metric':<15}" + "".join(f"{name:>15}" for name in results.keys())
    print("\n" + "=" * (15 + 15 * len(results)))
    print("COMPARISON")
    print("=" * (15 + 15 * len(results)))
    print(header)
    print("-" * (15 + 15 * len(results)))

    for key in metric_keys:
        row = f"{key:<15}"
        for name, metrics in results.items():
            val = metrics.get(key, "N/A")
            row += f"{val:>15}"
        print(row)

    # State accuracy
    row = f"{'state_acc':<15}"
    for name, metrics in results.items():
        val = metrics.get("state_accuracy", {}).get("overall", "N/A")
        row += f"{val:>15}"
    print(row)
    print("=" * (15 + 15 * len(results)))

    # Save comparison
    comparison_file = dirs[0].parent / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {comparison_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KELE run results")
    parser.add_argument("dirs", type=Path, nargs="+", help="Results directories to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs side-by-side")
    args = parser.parse_args()

    if args.compare and len(args.dirs) >= 2:
        compare_runs(args.dirs)
    else:
        for d in args.dirs:
            evaluate_run(d)
