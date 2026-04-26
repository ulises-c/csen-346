"""
KELE Socratic Teaching System — working copy extended from the original.

This module wraps the KELE SocraticTeachingSystem with our config layer
and adds batch evaluation for running against the SocratDataset.
"""

import json
import time
from pathlib import Path

from src.project.config import load_config
from src.project.socratic_teaching_system import SocraticTeachingSystem

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "references" / "KELE"


def create_system(
    debug: bool | None = None, experiment: str | None = None
) -> SocraticTeachingSystem:
    """Create a SocraticTeachingSystem from environment config."""
    cfg = load_config(experiment=experiment)
    return SocraticTeachingSystem(
        consultant_api_key=cfg.consultant.api_key,
        consultant_base_url=cfg.consultant.base_url,
        consultant_model_name=cfg.consultant.model_name,
        teacher_api_key=cfg.teacher.api_key,
        teacher_base_url=cfg.teacher.base_url,
        teacher_model_name=cfg.teacher.model_name,
        debug_mode=debug if debug is not None else cfg.debug_mode,
        max_teaching_rounds=cfg.max_teaching_rounds,
        consultant_max_tokens=cfg.consultant.max_tokens,
        consultant_disable_thinking=cfg.consultant.disable_thinking,
        consultant_thinking_budget=cfg.consultant.thinking_budget,
        consultant_num_ctx=cfg.consultant.num_ctx,
    )


def load_dataset(path: Path | None = None, split: str = "test", seed: int = 42) -> list[dict]:
    """Load the SocratDataset with train/test split.

    The paper uses a 90/10 train/test split. We evaluate on the test set (~680 dialogues).
    Args:
        split: "test" (10%, for evaluation), "train" (90%), or "all" (full dataset).
        seed: Random seed for reproducible splits.
    """
    if path is None:
        path = RESOURCES_DIR / "SocratDataset.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if split == "all":
        return data

    import random

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    split_point = int(len(data) * 0.9)
    if split == "train":
        return [data[i] for i in sorted(indices[:split_point])]
    else:  # test
        return [data[i] for i in sorted(indices[split_point:])]


def run_single_dialogue(system: SocraticTeachingSystem, item: dict) -> dict:
    """Replay a single dataset item through the system and collect outputs.

    Uses the student turns from the ground-truth dialogue as input,
    but the teacher and consultant responses come from our live system.
    """
    system.reset_session()
    ground_truth = item["dialogue"]
    generated_turns = []
    start = time.time()

    for turn in ground_truth:
        student_input = turn["student"]
        teacher_response = system.process_student_input(student_input)

        generated_turns.append(
            {
                "student": student_input,
                "state": system.current_state,
                "teacher_response": teacher_response,
                "ground_truth_teacher": turn["teacher"],
                "ground_truth_state": turn["state"],
            }
        )

        # If we hit the summary stage, stop
        if system.current_state == "e34":
            break

    elapsed = time.time() - start

    return {
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "num_turns_ground_truth": len(ground_truth),
        "num_turns_generated": len(generated_turns),
        "dialogue": generated_turns,
        "elapsed_seconds": round(elapsed, 2),
    }


def run_batch_evaluation(
    output_dir: Path,
    dataset_path: Path | None = None,
    start_id: int = 1,
    limit: int | None = None,
    experiment: str | None = None,
    split: str = "test",
) -> None:
    """Run the full evaluation pipeline on the dataset.

    Saves each dialogue result individually (crash-safe) and writes
    a progress log for monitoring.
    """
    dataset = load_dataset(dataset_path, split=split)
    total = len(dataset)

    # Filter to start_id and apply limit
    dataset = [d for d in dataset if d["id"] >= start_id]
    if limit is not None:
        dataset = dataset[:limit]

    system = create_system(debug=False, experiment=experiment)

    output_dir.mkdir(parents=True, exist_ok=True)
    dialogues_dir = output_dir / "dialogues"
    dialogues_dir.mkdir(exist_ok=True)
    progress_log = output_dir / "progress.log"
    completed = 0
    start_time = time.time()

    print(f"Starting evaluation: {len(dataset)} dialogues (of {total} in {split} split)")
    print(f"Output: {output_dir}")
    print(f"Teacher model: {system.teacher_model_name}")
    print(f"Consultant model: {system.consultant_model_name}")
    print("-" * 60)
    print(
        f"  {'#':>9}  {'id':<8}  {'turns':>5}  {'time':>5}  {'%':>5}  {'dlg/s':>6}  {'ETA':>5}  status"
    )
    print("-" * 60)

    CLR = "\r\033[K"  # carriage return + erase to end of line

    for item in dataset:
        item_id = item["id"]
        out_file = dialogues_dir / f"{item_id:04d}.json"

        # Skip if already completed (crash recovery)
        if out_file.exists():
            completed += 1
            continue

        pos = completed + 1
        print(f"{CLR}▶ {pos:>4}/{len(dataset)}  id={item_id:04d}", end="", flush=True)

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0

        try:
            result = run_single_dialogue(system, item)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            completed += 1
            turns = result.get("num_turns_generated", "?")
            secs = result.get("elapsed_seconds", 0)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(dataset) - completed) / rate if rate > 0 else 0
            print(
                f"{CLR}  {pos:>4}/{len(dataset)}  id={item_id:04d}"
                f"  {turns:>5} turns  {secs:>4.0f}s"
                f"  {completed / len(dataset) * 100:>4.1f}%"
                f"  {rate:>5.2f}  {remaining / 60:>4.0f}m  ✓"
            )
        except Exception as e:
            error_result = {"id": item_id, "error": str(e)}
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(dataset) - completed) / rate if rate > 0 else 0
            print(
                f"{CLR}  {pos:>4}/{len(dataset)}  id={item_id:04d}"
                f"  {'?':>5} turns  {'?':>4}s"
                f"  {completed / len(dataset) * 100:>4.1f}%"
                f"  {rate:>5.2f}  {remaining / 60:>4.0f}m  ERROR: {e}"
            )

        with open(progress_log, "w") as f:
            f.write(
                f"{completed}/{len(dataset)} {completed / len(dataset) * 100:.1f}%"
                f" {rate:.2f} dlg/s ETA {remaining / 60:.0f}m elapsed {elapsed / 60:.0f}m\n"
            )

    print(f"\nDone. {completed} dialogues saved to {dialogues_dir}")

    # Save run config
    cfg = load_config()
    run_config = {
        "experiment": output_dir.name,
        "teacher_model": cfg.teacher.model_name,
        "teacher_base_url": cfg.teacher.base_url,
        "consultant_model": cfg.consultant.model_name,
        "consultant_base_url": cfg.consultant.base_url,
        "max_teaching_rounds": cfg.max_teaching_rounds,
        "total_dialogues": len(dataset),
        "completed": completed,
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Auto-compute metrics after run completes
    print("\nComputing metrics...")
    from src.project.metrics import compute_all_metrics, format_metrics_table

    metrics = compute_all_metrics(dialogues_dir)
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(format_metrics_table(metrics))


def interactive() -> None:
    """Launch an interactive Socratic teaching session."""
    system = create_system()
    system.start_conversation()


def main() -> None:
    """CLI entry point for the KELE runner."""
    import argparse

    parser = argparse.ArgumentParser(description="KELE Socratic Teaching System")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment config name (loads configs/<name>.env). E.g.: baseline, gemma4",
    )
    sub = parser.add_subparsers(dest="command")

    # Interactive mode
    sub.add_parser("interactive", help="Start an interactive teaching session")

    # Batch evaluation mode
    eval_parser = sub.add_parser("evaluate", help="Run batch evaluation on the dataset")
    eval_parser.add_argument("--output", type=Path, default=None)
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Dataset split: test (10%%, default), train (90%%), all",
    )
    eval_parser.add_argument("--start-id", type=int, default=1, help="Resume from this dialogue ID")
    eval_parser.add_argument("--limit", type=int, default=None, help="Max dialogues to process")

    # Quick test mode — run on a handful of dialogues
    test_parser = sub.add_parser("test", help="Quick test with a few dialogues")
    test_parser.add_argument("--n", type=int, default=3, help="Number of dialogues to test")
    test_parser.add_argument("--output", type=Path, default=Path("results/test"))

    args = parser.parse_args()

    if args.command == "interactive":
        interactive()
    elif args.command == "evaluate":
        output = args.output or Path(f"results/{args.experiment or 'baseline'}")
        run_batch_evaluation(
            output,
            start_id=args.start_id,
            limit=args.limit,
            experiment=args.experiment,
            split=args.split,
        )
    elif args.command == "test":
        run_batch_evaluation(args.output, limit=args.n, experiment=args.experiment)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
