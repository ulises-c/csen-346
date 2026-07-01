# TODO(MELE rename): rename this module to MELE.py and change imports across the
# board — KELE→MELE is a major architectural change touching every importer.
"""
KELE Socratic Teaching System — working copy extended from the original.

This module wraps the KELE SocraticTeachingSystem with our config layer
and adds batch evaluation for running against the SocratDataset.
"""

import json
import os
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.project.config import load_config
from src.project.socratic_teaching_system import SocraticTeachingSystem
from src.project.wandb_tracking import EvalTracker

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "references" / "KELE"


def create_system(
    debug: bool | None = None,
    experiment: str | None = None,
    unified: bool = False,
    bert_consultant: str | None = None,
) -> SocraticTeachingSystem:
    """Create a SocraticTeachingSystem from environment config.

    If unified=True, instantiates SocraticTeachingSystemUnified — the
    single-call variant that fuses consultant + teacher into one
    structured-output LLM call. See docs/SOCRATIC_FUSION_PLAN.md.

    If bert_consultant is a path to a trained 34-state classifier checkpoint,
    use SocraticTeachingSystemBertConsultant: BERT predicts the state and
    the LLM only generates the teacher response (two-call style). Mutually
    exclusive with unified=True.
    """
    cfg = load_config(experiment=experiment)
    if bert_consultant and unified:
        raise ValueError("--unified and --bert-consultant are mutually exclusive")

    if bert_consultant:
        from src.project.socratic_teaching_bert_consultant import (
            SocraticTeachingSystemBertConsultant,
        )

        cls: type[SocraticTeachingSystem] = SocraticTeachingSystemBertConsultant
        extra_kwargs: dict = {"bert_ckpt": bert_consultant}
    elif unified:
        from src.project.socratic_teaching_unified import SocraticTeachingSystemUnified

        cls = SocraticTeachingSystemUnified
        extra_kwargs = {}
    else:
        cls = SocraticTeachingSystem
        extra_kwargs = {}
    return cls(
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
        **extra_kwargs,
    )


def load_dataset(
    path: Path | None = None,
    split: str = "test",
    seed: int = 42,
    source: str = "hf",
    hf_repo: str | list[str] = [  # noqa: B006
        # "ulises-c/SocratDataset-EN",  # uncomment to eval the bilingual set (zh+en, ~1361 test)
        "ulises-c/SocratDataset",
    ],
) -> list[dict]:
    """Load the SocratDataset with train/test split.

    The paper uses a 90/10 train/test split. We evaluate on the test set (~680 dialogues).
    Args:
        split: "test" (10%, for evaluation), "train" (90%), or "all" (full dataset).
        seed: Random seed for reproducible splits.
        source: "hf" to load from HuggingFace (default), "local" to read from a JSON file.
            Passing an explicit path implies source="local" regardless of this argument.
        hf_repo: One or more HuggingFace repo IDs loaded and concatenated when source="hf".
            Each repo must have dialogue records with {id, question, answer, dialogue} fields
            including state annotations (only SocratDataset / SocratDataset-EN qualify).
    """
    if source == "hf" and path is None:
        from datasets import load_dataset as hf_load

        repos = [hf_repo] if isinstance(hf_repo, str) else list(hf_repo)
        data: list[dict] = []
        for repo in repos:
            data.extend(dict(r) for r in hf_load(repo, split="train"))
    else:
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

        turn_record: dict = {
            "student": student_input,
            "state": system.current_state,
            "teacher_response": teacher_response,
            "ground_truth_teacher": turn["teacher"],
            "ground_truth_state": turn["state"],
        }
        thinking = getattr(system, "_last_thinking_content", None)
        if thinking:
            turn_record["thinking_content"] = thinking

        generated_turns.append(turn_record)

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


def _format_progress_line(
    pos: int,
    total: int,
    item_id: int,
    result: dict,
    elapsed: float,
    rate: float,
    remaining: float,
) -> str:
    """One-line progress record used by both sequential and parallel paths."""
    if "error" in result:
        status = f"ERROR: {result['error']}"
        turns: str | int = "?"
        secs: float = 0
    else:
        status = "✓"
        turns = result.get("num_turns_generated", "?")
        secs = result.get("elapsed_seconds", 0) or 0
    return (
        f"  {pos:>4}/{total}  id={item_id:04d}"
        f"  {turns:>5} turns  {secs:>4.0f}s"
        f"  {pos / total * 100:>4.1f}%"
        f"  {rate * 3600:>6.1f}  {remaining / 60:>4.0f}m  {status}"
    )


def _write_progress_log(
    progress_log: Path,
    completed: int,
    total: int,
    rate: float,
    remaining: float,
    elapsed: float,
) -> None:
    with open(progress_log, "w") as f:
        f.write(
            f"{completed}/{total} {completed / total * 100:.1f}%"
            f" {rate * 3600:.1f} dlg/hr ETA {remaining / 60:.0f}m elapsed {elapsed / 60:.0f}m\n"
        )


def _make_incremental_metrics_logger(
    tracker: EvalTracker, dialogues_dir: Path
) -> Callable[[int], None] | None:
    """Per-N-dialogues W&B logging (WANDB_EVAL_LOG_EVERY, default 10): recompute
    metrics over everything on disk and log at step=completed, so eval metrics
    form a convergence curve instead of a single end-of-run point."""
    every = int(os.environ.get("WANDB_EVAL_LOG_EVERY", "10"))
    if every < 1:
        return None
    from src.project.metrics import compute_metrics_from_records, load_dialogue_records

    def on_result(completed: int) -> None:
        if completed % every:
            return
        records = load_dialogue_records(dialogues_dir, skip_invalid=True)
        metrics = compute_metrics_from_records(records)
        if "error" not in metrics:
            tracker.log_step(metrics, step=completed)

    return on_result


def _run_sequential(
    pending: list[dict],
    dialogues_dir: Path,
    progress_log: Path,
    system: SocraticTeachingSystem,
    total_dataset: int,
    completed_initial: int,
    start_time: float,
    on_result: Callable[[int], None] | None = None,
) -> int:
    completed = completed_initial
    for item in pending:
        item_id = item["id"]
        out_file = dialogues_dir / f"{item_id:04d}.json"
        try:
            result = run_single_dialogue(system, item)
        except Exception as e:
            result = {"id": item_id, "error": str(e)}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        completed += 1
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (total_dataset - completed) / rate if rate > 0 else 0
        print(
            _format_progress_line(
                completed, total_dataset, item_id, result, elapsed, rate, remaining
            ),
            flush=True,
        )
        _write_progress_log(progress_log, completed, total_dataset, rate, remaining, elapsed)
        if on_result is not None:
            on_result(completed)
    return completed


def _run_parallel(
    pending: list[dict],
    dialogues_dir: Path,
    progress_log: Path,
    total_dataset: int,
    completed_initial: int,
    start_time: float,
    *,
    workers: int,
    experiment: str | None,
    unified: bool,
    bert_consultant: str | None,
    on_result: Callable[[int], None] | None = None,
) -> tuple[int, list[SocraticTeachingSystem]]:
    """Run pending dialogues concurrently using a ThreadPoolExecutor.

    Each worker thread owns its own SocraticTeachingSystem (session state
    on the system object makes per-worker isolation the clean choice).
    BERT-consultant variant: each worker loads its own ~100 MB classifier
    onto the GPU — small price for full thread isolation.

    Returns (final completed count, list of worker systems for end-of-run
    metadata aggregation).
    """
    thread_local = threading.local()
    worker_systems: list[SocraticTeachingSystem] = []
    worker_systems_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_box = [completed_initial]

    def get_or_create_system() -> SocraticTeachingSystem:
        sys_ = getattr(thread_local, "system", None)
        if sys_ is None:
            sys_ = create_system(
                debug=False,
                experiment=experiment,
                unified=unified,
                bert_consultant=bert_consultant,
            )
            thread_local.system = sys_
            with worker_systems_lock:
                worker_systems.append(sys_)
        return sys_

    def process_one(item: dict) -> tuple[dict, dict]:
        item_id = item["id"]
        out_file = dialogues_dir / f"{item_id:04d}.json"
        sys_ = get_or_create_system()
        try:
            result = run_single_dialogue(sys_, item)
        except Exception as e:
            result = {"id": item_id, "error": str(e)}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return item, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, item) for item in pending]
        for fut in as_completed(futures):
            item, result = fut.result()
            with progress_lock:
                completed_box[0] += 1
                completed = completed_box[0]
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_dataset - completed) / rate if rate > 0 else 0
                print(
                    _format_progress_line(
                        completed, total_dataset, item["id"], result, elapsed, rate, remaining
                    ),
                    flush=True,
                )
                _write_progress_log(
                    progress_log, completed, total_dataset, rate, remaining, elapsed
                )
                if on_result is not None:
                    on_result(completed)

    return completed_box[0], worker_systems


def run_batch_evaluation(
    output_dir: Path,
    dataset_path: Path | None = None,
    start_id: int = 1,
    limit: int | None = None,
    experiment: str | None = None,
    split: str = "test",
    unified: bool = False,
    fresh: bool = False,
    worker_id: int = 0,
    num_workers: int = 1,
    bert_consultant: str | None = None,
    workers: int | None = None,
    sample_seed: int | None = None,
    hf_repo: str | list[str] | None = None,
) -> None:
    """Run the full evaluation pipeline on the dataset.

    Saves each dialogue result individually (crash-safe) and writes
    a progress log for monitoring.

    If unified=True, uses the single-call fusion architecture
    (see docs/SOCRATIC_FUSION_PLAN.md).

    If workers > 1 (or env var KELE_PARALLEL_WORKERS > 1), dialogues are
    processed concurrently against the llama.cpp server's parallel KV slots.
    Each worker thread owns its own SocraticTeachingSystem; per-dialogue
    file output remains crash-safe (each dialogue → own JSON file).
    """
    if workers is None:
        workers = int(os.environ.get("KELE_PARALLEL_WORKERS", "1"))
    if workers < 1:
        workers = 1

    if hf_repo:
        dataset = load_dataset(dataset_path, split=split, hf_repo=hf_repo)
    else:
        dataset = load_dataset(dataset_path, split=split)
    total = len(dataset)

    # Filter to start_id, optionally random-subsample, then apply limit.
    # sample_seed is needed when --limit << len(split): without it, --limit
    # picks first-N-by-sorted-ID, which collides with the convergence
    # analysis's random-subsample assumption (ε≤2pp at n=400 requires
    # random draw, not the first 400 by ID).
    dataset = [d for d in dataset if d["id"] >= start_id]
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(dataset)
    if limit is not None:
        dataset = dataset[:limit]
    if num_workers > 1:
        dataset = [d for i, d in enumerate(dataset) if i % num_workers == worker_id]

    # Always create one "probe" system for header info + run config metadata.
    # In sequential mode this is also the workhorse; in parallel mode each
    # worker thread creates its own via thread-local storage.
    system = create_system(
        debug=False, experiment=experiment, unified=unified, bert_consultant=bert_consultant
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dialogues_dir = output_dir / "dialogues"
    dialogues_dir.mkdir(exist_ok=True)

    if fresh:
        # Archive previous run_config + metrics into run{N}_{timestamp}/
        prev_config = output_dir / "run_config.json"
        prev_metrics = output_dir / "metrics_summary.json"
        if prev_config.exists() or prev_metrics.exists():
            existing = sorted(output_dir.glob("run[0-9]*_*/"))
            next_n = len(existing) + 1
            try:
                ts_raw = (
                    json.loads(prev_config.read_text())["started_at"]
                    if prev_config.exists()
                    else None
                )
            except Exception:
                ts_raw = None
            ts = (ts_raw or datetime.now().astimezone().isoformat(timespec="seconds")).replace(
                ":", "-"
            )
            archive_dir = output_dir / f"run{next_n}_{ts}"
            archive_dir.mkdir()
            for src in (prev_config, prev_metrics):
                if src.exists():
                    src.rename(archive_dir / src.name)

        # Clear dialogues and all progress logs
        for f in dialogues_dir.glob("*.json"):
            f.unlink()
        for p in output_dir.glob("progress*.log"):
            p.unlink()

    progress_log_name = f"progress_worker{worker_id + 1}.log" if num_workers > 1 else "progress.log"
    progress_log = output_dir / progress_log_name
    start_time = time.time()
    started_at = datetime.now().astimezone()

    fresh_tag = "  [NEW — previous results cleared]" if fresh else ""
    worker_tag = f"  [worker {worker_id + 1}/{num_workers}]" if num_workers > 1 else ""
    print(
        f"Starting evaluation: {len(dataset)} dialogues (of {total} in {split} split)"
        f"{worker_tag}{fresh_tag}"
    )
    print(f"Started: {started_at.isoformat(timespec='seconds')}")
    print(f"Output: {output_dir}")
    print(f"Teacher model: {system.teacher_model_name}")
    print(f"Consultant model: {system.consultant_model_name}")
    print(f"Parallel workers: {workers}")
    print("-" * 60)
    print(
        f"  {'#':>9}  {'id':<8}  {'turns':>5}  {'time':>5}  {'%':>5}  {'dlg/hr':>6}  {'ETA':>5}  status"
    )
    print("-" * 60)

    # Bucket already-done dialogues into the completed counter; only pending
    # items get dispatched. Same crash-recovery semantics as the prior loop.
    # Empty files (zero bytes) are treated as missing — a crash mid-write leaves
    # 0-byte placeholders that would otherwise be skipped and later break
    # metrics computation when json.loads sees an empty string.
    pending: list[dict] = []
    completed = 0
    for item in dataset:
        out_file = dialogues_dir / f"{item['id']:04d}.json"
        if out_file.exists() and out_file.stat().st_size > 0:
            completed += 1
        else:
            pending.append(item)

    if completed and pending:
        print(f"  (resuming: {completed} already on disk, {len(pending)} to run)")

    # WANDB_EVAL_RUN_NAME lets a caller distinguish runs that share an
    # --experiment but differ otherwise (e.g. the same teacher served with
    # MTP off vs on → distinct output dirs, same experiment config).
    run_name = os.environ.get("WANDB_EVAL_RUN_NAME") or experiment or output_dir.name
    tracker = EvalTracker()
    on_result = None
    if num_workers == 1:
        tracker.start(run_name)
        if tracker.active:
            on_result = _make_incremental_metrics_logger(tracker, dialogues_dir)

    if workers == 1:
        completed = _run_sequential(
            pending,
            dialogues_dir,
            progress_log,
            system,
            len(dataset),
            completed,
            start_time,
            on_result=on_result,
        )
    else:
        completed, worker_systems = _run_parallel(
            pending,
            dialogues_dir,
            progress_log,
            len(dataset),
            completed,
            start_time,
            workers=workers,
            experiment=experiment,
            unified=unified,
            bert_consultant=bert_consultant,
            on_result=on_result,
        )
        # Aggregate fallback counts across worker systems for unified mode.
        if unified:
            total_fb = 0
            for ws in [system, *worker_systems]:
                fb = getattr(ws, "_unified_fallback_count", 0)
                if fb:
                    total_fb += fb
            setattr(system, "_unified_fallback_count", total_fb)

    print(f"\nDone. {completed} dialogues saved to {dialogues_dir}")

    cfg = load_config()
    run_config = {
        "experiment": output_dir.name,
        "teacher_model": cfg.teacher.model_name,
        "teacher_base_url": cfg.teacher.base_url,
        "consultant_model": cfg.consultant.model_name,
        "consultant_base_url": cfg.consultant.base_url,
        # --bert-consultant replaces the LLM consultant entirely; without this
        # field run_config misattributes state decisions to consultant_model.
        "bert_consultant": bert_consultant,
        "thinking_budget": cfg.consultant.thinking_budget,
        "max_teaching_rounds": cfg.max_teaching_rounds,
        "unified": unified,
        "workers": workers,
        "total_dialogues": len(dataset),
        "completed": completed,
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if unified and hasattr(system, "_unified_fallback_count"):
        run_config["unified_fallback_count"] = system._unified_fallback_count  # type: ignore[reportAttributeAccessIssue]

    if num_workers > 1:
        # Multi-worker: write a per-worker config; orchestrator merges and computes metrics
        config_name = f"run_config_worker{worker_id + 1}.json"
        with open(output_dir / config_name, "w") as f:
            json.dump(run_config, f, indent=2)
        print(f"Worker {worker_id + 1} done. Config saved to {config_name}")
    else:
        with open(output_dir / "run_config.json", "w") as f:
            json.dump(run_config, f, indent=2)

        print("\nComputing metrics...")
        from src.project.metrics import compute_all_metrics, format_metrics_table

        metrics = compute_all_metrics(dialogues_dir)
        with open(output_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(format_metrics_table(metrics))
        tracker.finish(metrics, step=completed)


def replay_wandb(
    output_dir: Path,
    every: int = 10,
    run_name: str | None = None,
    order: str = "completion",
) -> None:
    """Re-log a finished eval to W&B as a per-dialogue metric curve.

    Recomputes metrics over growing prefixes of the saved dialogues and logs
    one point per ``every`` dialogues, turning a run that produced a single
    end-of-run data point into a convergence curve — no re-evaluation needed.

    order='completion' replays in the order dialogues actually finished
    (file mtime); order='id' replays sorted by dialogue id.
    """
    from src.project.metrics import compute_metrics_from_records

    dialogues_dir = output_dir / "dialogues"
    files = sorted(dialogues_dir.glob("*.json"))
    if order == "completion":
        files.sort(key=lambda f: f.stat().st_mtime)
    records = []
    for f in files:
        data = json.loads(f.read_text())
        if "error" not in data:
            records.append(data)
    if not records:
        print(f"No valid dialogues in {dialogues_dir}")
        return

    # Explicit replay request implies wandb logging — bypass the WANDB_EVAL gate.
    os.environ.setdefault("WANDB_EVAL", "1")
    tracker = EvalTracker()
    tracker.start(run_name or os.environ.get("WANDB_EVAL_RUN_NAME") or f"{output_dir.name}-curve")
    if not tracker.active:
        return

    print(f"Replaying {len(records)} dialogues from {dialogues_dir} (every {every}, {order} order)")
    for n in [*range(every, len(records), every), len(records)]:
        metrics = compute_metrics_from_records(records[:n])
        tracker.log_step(metrics, step=n)
        print(
            f"  n={n:>4}  rouge1={metrics['rouge1']:>6.2f}  rougeL={metrics['rougeL']:>6.2f}"
            f"  bleu4={metrics['bleu4']:>6.2f}"
            f"  state_acc={metrics['state_accuracy']['overall']:>6.2f}"
        )
    tracker.finish()


def interactive(
    experiment: str | None = None,
    bert_consultant: str | None = None,
    unified: bool = False,
) -> None:
    """Launch an interactive Socratic teaching session."""
    system = create_system(
        experiment=experiment,
        bert_consultant=bert_consultant,
        unified=unified,
    )
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
    interactive_parser = sub.add_parser("interactive", help="Start an interactive teaching session")
    interactive_parser.add_argument(
        "--bert-consultant",
        type=str,
        default=None,
        help="Path to a trained 34-state classifier checkpoint dir. Replaces the "
        "LLM consultant with the classifier (the headline architecture).",
    )
    interactive_parser.add_argument(
        "--unified",
        action="store_true",
        help="Use the single-call fusion architecture (mutually exclusive with --bert-consultant).",
    )

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
    eval_parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="If set with --limit, randomly subsample N dialogues using this "
        "seed (instead of taking the first N by sorted ID). Required for "
        "the n=400 convergence-budget runs.",
    )
    eval_parser.add_argument(
        "--unified",
        action="store_true",
        help="Use single-call fusion architecture (consultant + teacher in one LLM call). "
        "See docs/SOCRATIC_FUSION_PLAN.md.",
    )
    eval_parser.add_argument(
        "--new", action="store_true", help="Start fresh — clear any existing results before running"
    )
    eval_parser.add_argument(
        "--worker-id", type=int, default=0, help="0-indexed worker number (for parallel runs)"
    )
    eval_parser.add_argument(
        "--num-workers", type=int, default=1, help="Total number of parallel workers"
    )
    eval_parser.add_argument(
        "--bert-consultant",
        type=str,
        default=None,
        help="Path to a trained 34-state BERT classifier checkpoint dir. "
        "Replaces the LLM consultant with the BERT classifier; LLM only "
        "generates the teacher response (two-call style).",
    )
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent dialogue workers. Each worker hits one "
        "of the llama-server's --parallel KV slots. Default 1 (or env "
        "KELE_PARALLEL_WORKERS). Recommended: 4. Server-side -np must be "
        "at least this value.",
    )
    eval_parser.add_argument(
        "--hf-repo",
        type=str,
        nargs="+",
        default=None,
        help="One or more HuggingFace dataset repo IDs to evaluate (concatenated). "
        "Default: ulises-c/SocratDataset (ZH). Use e.g. ulises-c/SocratDataset-EN "
        "(held-out EN) or ulises-c/SocratDataset-SYNTHETIC{,-EN} with --split all "
        "(synthetic sets are tiny, never-trained OOD probes).",
    )
    eval_parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to a SocratDataset JSON file. Defaults to references/KELE/"
        "SocratDataset.json (the original Chinese dataset). Use "
        "references/KELE-EN/SocratDataset.json for the English translation.",
    )

    # Replay a finished eval into W&B as a per-dialogue metric curve
    replay_parser = sub.add_parser(
        "wandb-replay",
        help="Re-log a finished eval's results dir to W&B as a per-dialogue metric curve",
    )
    replay_parser.add_argument(
        "--output", type=Path, required=True, help="Results dir containing dialogues/"
    )
    replay_parser.add_argument(
        "--every", type=int, default=10, help="Log one point per N dialogues"
    )
    replay_parser.add_argument(
        "--run-name", type=str, default=None, help="W&B run name (default: <dir>-curve)"
    )
    replay_parser.add_argument(
        "--order",
        choices=["completion", "id"],
        default="completion",
        help="Prefix order: completion (file mtime, the order dialogues finished) or id",
    )

    # Quick test mode — run on a handful of dialogues
    test_parser = sub.add_parser("test", help="Quick test with a few dialogues")
    test_parser.add_argument("--n", type=int, default=3, help="Number of dialogues to test")
    test_parser.add_argument("--output", type=Path, default=Path("results/test"))
    test_parser.add_argument(
        "--unified",
        action="store_true",
        help="Use single-call fusion architecture (consultant + teacher in one LLM call). "
        "See docs/SOCRATIC_FUSION_PLAN.md.",
    )
    test_parser.add_argument(
        "--bert-consultant",
        type=str,
        default=None,
        help="Path to a trained 34-state BERT classifier checkpoint dir.",
    )
    test_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        dest="dataset_path",
        help="Path to a SocratDataset-format JSON file. Defaults to the standard dataset.",
    )

    args = parser.parse_args()

    # weave.init auto-patches the openai clients in socratic_teaching_system, so a
    # single gated call here traces the whole consultant→teacher loop. Off unless set.
    weave_project = os.getenv("WEAVE_PROJECT")
    if weave_project:
        import weave  # pyright: ignore[reportMissingImports]

        weave.init(weave_project)

    if args.command == "interactive":
        interactive(
            experiment=args.experiment,
            bert_consultant=args.bert_consultant,
            unified=args.unified,
        )
    elif args.command == "evaluate":
        output = args.output or Path(f"results/{args.experiment or 'baseline'}")
        run_batch_evaluation(
            output,
            dataset_path=args.dataset_path,
            start_id=args.start_id,
            limit=args.limit,
            experiment=args.experiment,
            split=args.split,
            unified=args.unified,
            fresh=args.new,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            bert_consultant=args.bert_consultant,
            workers=args.workers,
            sample_seed=args.sample_seed,
            hf_repo=args.hf_repo,
        )
    elif args.command == "wandb-replay":
        replay_wandb(
            args.output,
            every=args.every,
            run_name=args.run_name,
            order=args.order,
        )
    elif args.command == "test":
        run_batch_evaluation(
            args.output,
            dataset_path=args.dataset_path,
            limit=args.n,
            experiment=args.experiment,
            unified=args.unified,
            bert_consultant=args.bert_consultant,
            split="all" if args.dataset_path else "test",
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
