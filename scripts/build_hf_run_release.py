#!/usr/bin/env python3
"""Stage the per-run outputs of the journal reruns as a Hugging Face dataset folder.

This script only writes a local folder. It never talks to the Hub.

    uv run python scripts/build_hf_run_release.py \
        --archive-dir ~/csen346-archive [--overlay DIR ...] --out STAGE_DIR

Input: one tarball per run in the archive directory (``r<digit>*.tgz``), each unpacking
to ``RUN/`` with ``dialogues/NNNN.json``, ``judge_scores/<stem>/NNNN.json``,
``metrics_summary.json``, ``judge_summary*.json`` and ``run_config.json``. Every
``--overlay`` directory holds ``RUN/`` subfolders whose files are laid over the tarball
contents (later overlays win); that is how a second judge's scores are added.

Output (the same bytes for the same input):

    STAGE_DIR/README.md                  dataset card with the Hub configs
    STAGE_DIR/runs.csv                   one row per run
    STAGE_DIR/data/RUN/turns.jsonl       one row per generated turn, one schema for all runs
    STAGE_DIR/data/RUN/dialogues.jsonl   one row per dialogue
    STAGE_DIR/summaries/RUN/*.json       run summaries with host names replaced

The build ends with a self-check and exits non-zero when it finds an internal identifier
anywhere in the staged tree, a run without turns, or row counts that disagree with the
run's own summaries.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

GITHUB_URL = "https://github.com/ulises-c/csen-346"
RUN_NAME_RE = re.compile(r"^r\d[A-Za-z0-9._-]*$")

# Host names are replaced by a description of the hardware, by exact string replacement,
# in this order (the longer name first, because it contains the shorter one).
HOST_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("lz-hw-0047", "rtx-a6000-node"),
    ("hw-0047", "rtx-a6000-node"),
    ("hw-0304", "rtx-4000-ada-node"),
    ("ollie-cachy", "rx-9700-node"),
    # Run 1 records the same machine by its short name.
    ("host cachy", "host rx-9700-node"),
)
HARDWARE_NAMES: tuple[str, ...] = ("rtx-a6000-node", "rtx-4000-ada-node", "rx-9700-node")

# Nothing in the staged tree may contain these, in any letter case.
FORBIDDEN_IDENTIFIERS: tuple[str, ...] = (
    "lz-hw-0047",
    "hw-0047",
    "hw-0304",
    "hw-0457",
    "lz-lazalab",
    "ollie-cachy",
    "cachy",
    "work-laptop",
    "m4-mini",
    "uchavarria",
    "ollie",
    "/home/",
    "/Users/",
    "lazamedical",
)

KNOWN_JUDGE_SHORT_NAMES = {
    "judge_summary": "qwen36",
    "judge_summary_bedrock-claude-sonnet-4-6": "sonnet46_bedrock",
}
# The model each known stem is named after. A summary written by another model under the
# same file name (run 1 was judged by Qwen3.8-27B) must not be listed under this name.
EXPECTED_JUDGE_MODEL_PREFIX = {
    "judge_summary": "Qwen3.6-27B",
    "judge_summary_bedrock-claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}
JUDGE_SCORE_FIELDS = (
    "total",
    "socratic_validity",
    "advancement",
    "age_appropriateness",
    "question_form",
)

TURN_BASE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "string"),
    ("dialogue_id", "int64"),
    ("dialogue_file", "string"),
    ("turn_idx", "int64"),
    ("question", "string"),
    ("student", "string"),
    ("predicted_state", "string"),
    ("ground_truth_state", "string"),
    ("teacher_response", "string"),
    ("ground_truth_teacher", "string"),
)
DIALOGUE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "string"),
    ("id", "int64"),
    ("dialogue_file", "string"),
    ("question", "string"),
    ("answer", "string"),
    ("num_turns_ground_truth", "int64"),
    ("num_turns_generated", "int64"),
    ("elapsed_seconds", "float64"),
)

# sha256 of the classifier that was trained on 607 of the 681 test dialogues (run 1).
LEAKING_CLASSIFIER_SHA256 = "9183119d77b79f5d8a60f5facd436dc59afe391aa97064ae83e1b2ae0be62220"
CLASSIFIER_NAMES = {
    "state-clf-qwen3.5-0.8b-lora-zh": "Qwen3.5-0.8B LoRA 34-state classifier (clean retrain)",
    "state-clf-qwen3.5-0.8b-lora-wandb": (
        "Qwen3.5-0.8B LoRA 34-state classifier (earlier checkpoint, trained on test dialogues)"
    ),
    "state-clf-bge-small-zh": "bge-small-zh 34-state classifier (clean retrain)",
}
TEACHER_NAMES = {
    "Gemma 4 12B SFT": "Gemma 4 12B fine-tune (SocratesLM-12B)",
    "GLM-4-9B-Chat-base": "GLM-4-9B-Chat (base)",
}
DATASETS = {
    "test-split-681": ("ulises-c/SocratDataset", 681),
    "synthetic-75": ("ulises-c/SocratDataset-SYNTHETIC", 75),
}

STATUS_REPORTABLE = "reportable"
STATUS_LEAK = "leak-documentation-only"
STATUS_CONTAMINATED = "contaminated-test-split"
STATUS_TEACHER_UNKNOWN = "reportable-teacher-training-overlap-unknown"


class BuildError(Exception):
    """The input cannot be staged."""


# --------------------------------------------------------------------------- input


def _is_hidden(rel: PurePosixPath) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _read_tarball(path: Path) -> dict[str, bytes]:
    """Files of one run tarball, keyed by their path below ``RUN/``."""
    run = path.name[: -len(".tgz")]
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if parts and parts[0] == ".":
                parts = parts[1:]
            if len(parts) < 2 or _is_hidden(PurePosixPath(*parts)):
                continue
            if parts[0] != run:
                raise BuildError(f"{path.name}: member {member.name!r} is outside {run}/")
            handle = tar.extractfile(member)
            if handle is not None:
                files[str(PurePosixPath(*parts[1:]))] = handle.read()
    return files


def _read_overlay_run(run_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(run_dir.rglob("*")):
        rel = PurePosixPath(path.relative_to(run_dir).as_posix())
        if path.is_file() and not _is_hidden(rel):
            files[str(rel)] = path.read_bytes()
    return files


def load_runs(archive_dir: Path, overlays: list[Path]) -> dict[str, dict[str, bytes]]:
    """Every run's files: the tarballs first, then each overlay directory on top."""
    runs: dict[str, dict[str, bytes]] = {}
    for tarball in sorted(archive_dir.glob("*.tgz")):
        run = tarball.name[: -len(".tgz")]
        if tarball.name.startswith(".") or not RUN_NAME_RE.match(run):
            continue
        runs[run] = _read_tarball(tarball)
    for overlay in overlays:
        if not overlay.is_dir():
            raise BuildError(f"overlay {overlay} is not a directory")
        for run_dir in sorted(overlay.iterdir()):
            if run_dir.is_dir() and RUN_NAME_RE.match(run_dir.name):
                runs.setdefault(run_dir.name, {}).update(_read_overlay_run(run_dir))
    if not runs:
        raise BuildError(f"no run tarballs (r<digit>*.tgz) in {archive_dir}")
    return dict(sorted(runs.items()))


# --------------------------------------------------------------------------- judges


def judge_short_name(stem: str) -> str:
    """Column-safe short name of a judge, from the stem of its summary file."""
    if stem in KNOWN_JUDGE_SHORT_NAMES:
        return KNOWN_JUDGE_SHORT_NAMES[stem]
    rest = stem[len("judge_summary") :] if stem.startswith("judge_summary") else stem
    short = re.sub(r"[^a-z0-9]+", "_", rest.lower()).strip("_")
    return short or "unnamed"


def judge_stems(files: dict[str, bytes]) -> set[str]:
    stems: set[str] = set()
    for rel in files:
        parts = PurePosixPath(rel).parts
        if len(parts) == 3 and parts[0] == "judge_scores" and rel.endswith(".json"):
            stems.add(parts[1])
        elif len(parts) == 1 and parts[0].startswith("judge_summary") and rel.endswith(".json"):
            stems.add(parts[0][: -len(".json")])
    return stems


def short_names_for(stems: set[str]) -> dict[str, str]:
    """Stem to short name, in the column order used everywhere; duplicates are an error."""
    mapping = {stem: judge_short_name(stem) for stem in sorted(stems)}
    counts = Counter(mapping.values())
    clashes = sorted(short for short, n in counts.items() if n > 1)
    if clashes:
        raise BuildError(f"two judge stems map to the same short name: {clashes}")
    return mapping


def judge_columns(short: str) -> list[str]:
    return [f"judge_{short}_{name}" for name in (*JUDGE_SCORE_FIELDS, "comment")]


# --------------------------------------------------------------------------- redaction


def substitute_hosts(data: bytes) -> bytes:
    text = data.decode("utf-8")
    for old, new in HOST_SUBSTITUTIONS:
        text = text.replace(old, new)
    return text.encode("utf-8")


def is_substitution_only(original: bytes, staged: bytes) -> bool:
    """True when ``staged`` differs from ``original`` only by the host substitutions.

    Checked without calling :func:`substitute_hosts`: every source name in the original
    and every hardware name in the staged copy is masked, and what is left must be the
    same text. A hardware name that already occurs in the original would make the
    comparison meaningless, so that case counts as a failure.
    """
    before = original.decode("utf-8")
    after = staged.decode("utf-8")
    if any(name in before for name in HARDWARE_NAMES):
        return False
    if any(old in after for old, _ in HOST_SUBSTITUTIONS):
        return False
    mask = "\x00"
    for old, new in HOST_SUBSTITUTIONS:
        for name in HARDWARE_NAMES:
            new = new.replace(name, mask)
        before = before.replace(old, new)
    for name in HARDWARE_NAMES:
        after = after.replace(name, mask)
    return before == after


def find_forbidden(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name in FORBIDDEN_IDENTIFIERS if name.lower() in lowered]


# --------------------------------------------------------------------------- run facts


def describe_consultant(config: dict) -> str:
    if config.get("oracle_consultant"):
        return "oracle (ground-truth states)"
    checkpoint = config.get("bert_consultant")
    if not checkpoint:
        model = str(config.get("consultant_model", "unknown"))
        same = model == config.get("teacher_model")
        return f"{model} (LLM consultant{', same model as the teacher' if same else ''})"
    path = PurePosixPath(str(checkpoint))
    name = path.parent.name if path.name == "final" else path.name
    return CLASSIFIER_NAMES.get(name, f"classifier checkpoint {name}")


def describe_teacher(config: dict) -> str:
    model = str(config.get("teacher_model", "unknown"))
    return TEACHER_NAMES.get(model, model)


def detect_dataset(run: str, n_dialogues: int) -> str:
    """The evaluation set, from the run name; the dialogue count has to agree with it."""
    name = "synthetic-75" if "synth75" in run else "test-split-681" if "n681" in run else None
    if name is None:
        raise BuildError(f"{run}: the run name does not say which dataset it used")
    if DATASETS[name][1] != n_dialogues:
        raise BuildError(
            f"{run}: {n_dialogues} dialogues, but {name} has {DATASETS[name][1]} dialogues"
        )
    return name


def run_status(run: str, config: dict, dataset: str) -> tuple[str, list[str]]:
    """Status label and the reasons behind it, following the rerun handoff."""
    notes: list[str] = []
    teacher = str(config.get("teacher_model", ""))
    fine_tuned_gemma = "gemma12bsft" in run or teacher == "Gemma 4 12B SFT"
    status = STATUS_REPORTABLE
    if (
        run.startswith("r1-")
        or config.get("bert_consultant_sha256") == LEAKING_CLASSIFIER_SHA256
        or "state-clf-qwen3.5-0.8b-lora-wandb" in str(config.get("bert_consultant"))
    ):
        status = STATUS_LEAK
        notes.append(
            "The consultant classifier was trained on 607 of the 681 test dialogues; "
            "kept only to document that leak."
        )
    elif fine_tuned_gemma and dataset == "test-split-681":
        status = STATUS_CONTAMINATED
        notes.append(
            "The Gemma 4 12B fine-tune was trained on turns from all 681 test dialogues; "
            "its synthetic-75 run is the only clean comparison."
        )
    elif teacher == "SocratTeachLLM" and dataset == "test-split-681":
        status = STATUS_TEACHER_UNKNOWN
        notes.append("SocratTeachLLM's training data is not known to exclude the test split.")
    if dataset == "synthetic-75":
        notes.append(
            "State accuracy on the synthetic set reflects a different sub-state labelling "
            "convention; do not report it."
        )
    return status, notes


@dataclass
class JudgeFacts:
    model: str = ""
    mean: float | None = None
    complete: bool | None = None
    turns_judged: int | None = None
    per_turn_scores: int = 0
    listed: bool = True


@dataclass
class RunResult:
    run: str
    row: dict[str, object]
    n_turns: int
    n_dialogues: int
    judges: dict[str, JudgeFacts] = field(default_factory=dict)


def _load_json(files: dict[str, bytes], rel: str, run: str) -> dict:
    if rel not in files:
        raise BuildError(f"{run}: {rel} is missing")
    return json.loads(files[rel].decode("utf-8"))


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def stage_run(run: str, files: dict[str, bytes], shorts: dict[str, str], out: Path) -> RunResult:
    """Write one run's data and summaries; return its index row and counts."""
    config = _load_json(files, "run_config.json", run)
    metrics = _load_json(files, "metrics_summary.json", run)

    dialogue_files = sorted(
        rel for rel in files if rel.startswith("dialogues/") and rel.endswith(".json")
    )
    scores: dict[str, dict[tuple[str, int], dict]] = {}
    for stem in shorts:
        per_turn: dict[tuple[str, int], dict] = {}
        prefix = f"judge_scores/{stem}/"
        for rel in sorted(files):
            if not (rel.startswith(prefix) and rel.endswith(".json")):
                continue
            name = rel[len(prefix) :]
            for score in json.loads(files[rel].decode("utf-8")).get("scores", []):
                per_turn[(name, int(score["turn_idx"]))] = score
        scores[stem] = per_turn

    turn_rows: list[dict] = []
    dialogue_rows: list[dict] = []
    seen_turns: set[tuple[str, int]] = set()
    ground_truth_turns = correct = cut_short = 0
    for rel in dialogue_files:
        name = PurePosixPath(rel).name
        dialogue = json.loads(files[rel].decode("utf-8"))
        answer = dialogue.get("answer")
        dialogue_rows.append(
            {
                "run": run,
                "id": dialogue["id"],
                "dialogue_file": name,
                "question": dialogue.get("question"),
                "answer": None if answer is None else str(answer),
                "num_turns_ground_truth": dialogue.get("num_turns_ground_truth"),
                "num_turns_generated": dialogue.get("num_turns_generated"),
                "elapsed_seconds": dialogue.get("elapsed_seconds"),
            }
        )
        turns = dialogue.get("dialogue", [])
        ground_truth_turns += int(dialogue.get("num_turns_ground_truth") or 0)
        cut_short += len(turns) < int(dialogue.get("num_turns_ground_truth") or 0)
        for turn_idx, turn in enumerate(turns):
            correct += turn.get("state") == turn.get("ground_truth_state")
            seen_turns.add((name, turn_idx))
            row: dict[str, object] = {
                "run": run,
                "dialogue_id": dialogue["id"],
                "dialogue_file": name,
                "turn_idx": turn_idx,
                "question": dialogue.get("question"),
                "student": turn.get("student"),
                "predicted_state": turn.get("state"),
                "ground_truth_state": turn.get("ground_truth_state"),
                "teacher_response": turn.get("teacher_response"),
                "ground_truth_teacher": turn.get("ground_truth_teacher"),
            }
            for stem, short in shorts.items():
                score = scores[stem].get((name, turn_idx), {})
                for key in (*JUDGE_SCORE_FIELDS, "comment"):
                    row[f"judge_{short}_{key}"] = score.get(key)
            turn_rows.append(row)

    for stem, per_turn in scores.items():
        orphans = sorted(set(per_turn) - seen_turns)
        if orphans:
            raise BuildError(
                f"{run}: {len(orphans)} {stem} scores have no matching turn (first {orphans[0]})"
            )

    data_dir = out / "data" / run
    data_dir.mkdir(parents=True)
    (data_dir / "turns.jsonl").write_bytes(_jsonl(turn_rows))
    (data_dir / "dialogues.jsonl").write_bytes(_jsonl(dialogue_rows))

    summary_dir = out / "summaries" / run
    summary_dir.mkdir(parents=True)
    for rel in sorted(files):
        top_level = "/" not in rel and rel.endswith(".json")
        if top_level and (
            rel in ("metrics_summary.json", "run_config.json") or rel.startswith("judge_summary")
        ):
            staged = substitute_hosts(files[rel])
            if not is_substitution_only(files[rel], staged):
                raise BuildError(f"{run}: {rel} changed by more than the host substitution")
            json.loads(staged.decode("utf-8"))
            (summary_dir / rel).write_bytes(staged)

    n_dialogues = len(dialogue_rows)
    dataset = detect_dataset(run, n_dialogues)
    status, notes = run_status(run, config, dataset)
    state_accuracy = metrics.get("state_accuracy", {})
    run_notes = substitute_hosts(
        str(config.get("env", {}).get("KELE_RUN_NOTES", "")).encode("utf-8")
    ).decode("utf-8")
    weights = re.search(r"teacher (\S+\.gguf)", run_notes)

    row_out: dict[str, object] = {
        "run": run,
        "status": status,
        "consultant": describe_consultant(config),
        "teacher": describe_teacher(config),
        "teacher_weights_file": weights.group(1) if weights else None,
        "dataset": dataset,
        "source_dataset": DATASETS[dataset][0],
        "n_dialogues": n_dialogues,
        "n_turns": len(turn_rows),
        "n_ground_truth_turns": ground_truth_turns,
        "state_accuracy": state_accuracy.get("overall"),
        "state_accuracy_all_ground_truth_turns": (
            round(100 * correct / ground_truth_turns, 2) if ground_truth_turns else None
        ),
        "state_accuracy_reportable": dataset != "synthetic-75"
        and status in (STATUS_REPORTABLE, STATUS_TEACHER_UNKNOWN),
        "pct_dialogues_cut_short": round(100 * cut_short / n_dialogues, 1) if n_dialogues else None,
        "rouge1": metrics.get("rouge1"),
        "rouge2": metrics.get("rouge2"),
        "rougeL": metrics.get("rougeL"),
        "bleu4": metrics.get("bleu4"),
    }

    result = RunResult(run=run, row=row_out, n_turns=len(turn_rows), n_dialogues=n_dialogues)
    for stem, short in shorts.items():
        facts = JudgeFacts(per_turn_scores=len(scores[stem]))
        if f"{stem}.json" in files:
            summary = json.loads(files[f"{stem}.json"].decode("utf-8"))
            facts.model = str(summary.get("judge_model", ""))
            facts.mean = summary.get("overall_avg")
            facts.complete = summary.get("complete")
            facts.turns_judged = summary.get("n_turns_judged")
            expected = EXPECTED_JUDGE_MODEL_PREFIX.get(stem)
            if expected and not facts.model.startswith(expected):
                facts.listed = False
                notes.append(
                    f"{stem}.json was written by {facts.model}, a different judge from the "
                    f"other runs; its mean is left out of this table (see summaries/)."
                )
            if not facts.per_turn_scores:
                notes.append(f"No per-turn scores were kept for {stem}.json.")
        result.judges[stem] = facts
        row_out[f"judge_{short}_model"] = facts.model or None
        row_out[f"judge_{short}_mean"] = facts.mean if facts.listed else None
        row_out[f"judge_{short}_complete"] = facts.complete
        row_out[f"judge_{short}_turns_judged"] = facts.turns_judged
        row_out[f"judge_{short}_per_turn_scores"] = facts.per_turn_scores

    row_out["git_commit"] = config.get("git_commit")
    row_out["git_dirty"] = config.get("git_dirty")
    row_out["run_notes"] = run_notes or None
    row_out["notes"] = " ".join(notes) or None
    return result


# --------------------------------------------------------------------------- card


def _yaml_features(columns: list[tuple[str, str]]) -> str:
    return "".join(f"  - name: {name}\n    dtype: {dtype}\n" for name, dtype in columns)


def turn_columns(shorts: dict[str, str]) -> list[tuple[str, str]]:
    columns = list(TURN_BASE_COLUMNS)
    for short in shorts.values():
        columns += [(f"judge_{short}_{name}", "int64") for name in JUDGE_SCORE_FIELDS]
        columns.append((f"judge_{short}_comment", "string"))
    return columns


def _size_category(n: int) -> str:
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return label
    return "100K<n<1M"


def _num(value: object, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"  # type: ignore[arg-type]


def build_readme(results: list[RunResult], shorts: dict[str, str]) -> str:
    total_turns = sum(r.n_turns for r in results)
    total_dialogues = sum(r.n_dialogues for r in results)
    judge_models: dict[str, str] = {}
    for stem in shorts:
        models = Counter(
            r.judges[stem].model for r in results if r.judges[stem].model and r.judges[stem].listed
        )
        judge_models[stem] = models.most_common(1)[0][0] if models else "unknown"

    front = (
        "---\n"
        "# TODO(owner): add a `license:` field before publishing. It is left out on purpose;\n"
        "# see the Licence section below. Both source datasets are cc-by-4.0 on the Hub.\n"
        "language:\n- zh\n- en\n"
        "pretty_name: KELE reproduction, per-turn run outputs (November 2026 reruns)\n"
        "task_categories:\n- text-generation\n"
        "tags:\n- education\n- socratic-teaching\n- dialogue\n- evaluation\n- llm-judge\n- kele\n"
        f"size_categories:\n- {_size_category(total_turns)}\n"
        "configs:\n"
        "- config_name: turns\n  default: true\n  data_files:\n"
        "  - split: train\n    path: data/*/turns.jsonl\n"
        "- config_name: dialogues\n  data_files:\n"
        "  - split: train\n    path: data/*/dialogues.jsonl\n"
        "dataset_info:\n"
        "- config_name: turns\n  features:\n"
        f"{_yaml_features(turn_columns(shorts))}"
        "- config_name: dialogues\n  features:\n"
        f"{_yaml_features(list(DIALOGUE_COLUMNS))}"
        "---\n"
    )

    header = ["Run", "Status", "Consultant", "Teacher", "Set", "Dialogues", "Turns"]
    header += ["State acc. (scored turns)", "ROUGE-1", "BLEU-4"]
    header += [f"Judge `{short}` mean (complete)" for short in shorts.values()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        row = r.row
        accuracy = _num(row["state_accuracy"])
        if row["dataset"] == "synthetic-75":
            accuracy = "not reported"
        cells = [
            f"`{r.run}`",
            str(row["status"]),
            str(row["consultant"]),
            str(row["teacher"]),
            str(row["dataset"]),
            str(r.n_dialogues),
            str(r.n_turns),
            accuracy,
            _num(row["rouge1"]),
            _num(row["bleu4"]),
        ]
        for stem in shorts:
            facts = r.judges[stem]
            if facts.mean is None:
                cells.append("")
            elif not facts.listed:
                cells.append(f"other judge ({facts.model})")
            else:
                done = "yes" if facts.complete else f"no, {facts.turns_judged} turns"
                if facts.per_turn_scores != facts.turns_judged:
                    done += f", {facts.per_turn_scores} per-turn scores here"
                cells.append(f"{_num(facts.mean)} ({done})")
        lines.append("| " + " | ".join(cells) + " |")
    run_table = "\n".join(lines)

    judge_lines = "\n".join(
        f"- `{short}`: summary file `{stem}.json`, per-turn scores from "
        f"`judge_scores/{stem}/`, judge model `{judge_models[stem]}`."
        for stem, short in shorts.items()
    )
    judge_column_lines = "\n".join(
        f"| `judge_{short}_total` | Sum of the four axes for judge `{short}`, 0 to 10. |\n"
        f"| `judge_{short}_socratic_validity`, `judge_{short}_advancement` | Axis scores, 0 to 3. |\n"
        f"| `judge_{short}_age_appropriateness`, `judge_{short}_question_form` | Axis scores, "
        f"0 to 2. |\n"
        f"| `judge_{short}_comment` | The judge's short reason, in English. |"
        for short in shorts.values()
    )
    substitutions = ", ".join(f"`{name}`" for name in HARDWARE_NAMES)

    body = f"""
# KELE reproduction: per-turn outputs of the November 2026 reruns

This dataset holds the outputs of {len(results)} evaluation runs of a Socratic-teaching
pipeline, from the project "Reproducing and Extending KELE" ([code]({GITHUB_URL})). In the
pipeline a *consultant* picks one of 34 teaching states for each student turn and a *teacher*
language model writes the reply to an elementary-school student. Each run pairs one consultant
with one teacher and replays the student turns of a set of reference dialogues.

It contains {total_turns:,} generated turns from {total_dialogues:,} dialogue replays: the
student turn, the state the consultant predicted, the reference state, the teacher's reply,
the reference reply, and the scores an LLM judge gave the reply. The run summaries and the
run settings are included. The dialogue text is Chinese. The judge comments are English;
some quote the Chinese text.

It is published so that the numbers in the paper can be recomputed. It is not a training set,
and several runs are included only to document a problem. Read the Caveats section before
using any number.

## Loading

```python
from datasets import load_dataset

turns = load_dataset("REPO_ID", "turns", split="train")          # one row per turn
dialogues = load_dataset("REPO_ID", "dialogues", split="train")  # one row per dialogue
one_run = turns.filter(lambda row: row["run"] == "{results[min(1, len(results) - 1)].run}")
```

Replace `REPO_ID` with this repository's id. Every run has the same columns. `runs.csv` is
a plain CSV file and is not a config: download it with
`huggingface_hub.hf_hub_download("REPO_ID", "runs.csv", repo_type="dataset")` and read it
with any CSV reader.

The column types are declared in this card. If you read the JSONL files without the card
(for example `load_dataset("json", data_files=...)`), pass the types yourself: the first
run in file order has no per-turn judge scores, so its judge columns are all null and the
types cannot be inferred from it.

## Runs

The same table, with more columns, is in `runs.csv`.

{run_table}

Status values:

- `{STATUS_REPORTABLE}`: a result, subject to the caveats below.
- `{STATUS_TEACHER_UNKNOWN}`: a result, but the teacher is SocratTeachLLM on the test split,
  and its training data is not known to exclude that split.
- `{STATUS_CONTAMINATED}`: the teacher was trained on the dialogues it is evaluated on. Not a
  clean measurement.
- `{STATUS_LEAK}`: the consultant was trained on most of the evaluation dialogues. Kept to
  document the leak, not as a result.

Sets: `test-split-681` is the 681-dialogue test part of a 90/10 split by dialogue (seed 42) of
[ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset);
`synthetic-75` is all 75 dialogues of
[ulises-c/SocratDataset-SYNTHETIC](https://huggingface.co/datasets/ulises-c/SocratDataset-SYNTHETIC).
All runs use a 10-shot teacher prompt.

## Files

| Path | Content |
|---|---|
| `data/<run>/turns.jsonl` | One row per generated turn. |
| `data/<run>/dialogues.jsonl` | One row per dialogue. |
| `summaries/<run>/metrics_summary.json` | Turn count, ROUGE, BLEU-4 and state accuracy of the run. |
| `summaries/<run>/judge_summary*.json` | One file per judge: means, per-stage means, per-dialogue means, and whether the pass is complete. |
| `summaries/<run>/run_config.json` | Models, settings, git commit, start and end time. |
| `runs.csv` | One row per run. |

### Columns of `turns`

| Column | Meaning |
|---|---|
| `run` | Run name. |
| `dialogue_id` | Id of the dialogue in the source dataset. |
| `dialogue_file` | File name of the dialogue in the run's output folder. |
| `turn_idx` | Position of the turn in the dialogue, from 0. |
| `question` | The science question the dialogue is about. |
| `student` | The student's turn, taken from the reference dialogue. |
| `predicted_state` | The teaching state the consultant chose (`a1`, `b2` to `b7`, `c8` to `c29`, `d30` to `d33`, `e34`). |
| `ground_truth_state` | The state recorded in the source dataset for this turn. |
| `teacher_response` | The reply the teacher model wrote. |
| `ground_truth_teacher` | The reference reply in the source dataset. |
{judge_column_lines}

Judge columns are null where that judge did not score the turn. Judges present:

{judge_lines}

### Columns of `dialogues`

| Column | Meaning |
|---|---|
| `run`, `id`, `dialogue_file` | Run name, dialogue id in the source dataset, and file name. |
| `question`, `answer` | The question and its answer as stored in the source dataset. `answer` is written as text here because the two source datasets store it in different types. |
| `num_turns_ground_truth` | Turns in the reference dialogue. |
| `num_turns_generated` | Turns the run produced. A dialogue stops when the consultant predicts the closing state `e34`, so this can be smaller. |
| `elapsed_seconds` | Wall-clock time of the dialogue. |

### Columns of `runs.csv`

| Column | Meaning |
|---|---|
| `run`, `status`, `consultant`, `teacher`, `teacher_weights_file`, `dataset`, `source_dataset` | What the run is. `consultant` and `teacher` come from `run_config.json`. |
| `n_dialogues`, `n_turns`, `n_ground_truth_turns` | Dialogues, generated turns, and turns in the reference dialogues. |
| `state_accuracy` | Percent of generated turns whose predicted state equals the reference state (from `metrics_summary.json`). |
| `state_accuracy_all_ground_truth_turns` | The same count of correct turns divided by all reference turns, so turns that were never generated count as wrong. Computed by the build script from the turn rows. |
| `state_accuracy_reportable` | `false` where the caveats below say the figure must not be reported. |
| `pct_dialogues_cut_short` | Percent of dialogues with fewer generated turns than reference turns. |
| `rouge1`, `rouge2`, `rougeL`, `bleu4` | Text overlap between the teacher's replies and the reference replies, over generated turns. |
| `judge_<name>_model`, `_mean`, `_complete`, `_turns_judged`, `_per_turn_scores` | The judge model, its mean total score, whether it scored every turn of the run, how many turns it scored, and how many per-turn scores are in `turns`. |
| `git_commit`, `git_dirty` | The code version recorded by the run. |
| `run_notes` | Serving settings recorded by the run, with host names replaced. |
| `notes` | The reasons for the status and other warnings. |

## How the numbers were produced

The code is at {GITHUB_URL}; `git_commit` in `runs.csv` gives the commit of each run.

- The evaluation pipeline writes one JSON file per dialogue and `metrics_summary.json`.
  ROUGE and BLEU-4 are computed on generated turns only.
- `scripts/llm_judge_eval.py` scores every teacher reply on four axes (Socratic validity 0 to
  3, advancement 0 to 3, age appropriateness 0 to 2, question form 0 to 2) and writes the
  per-turn scores and `judge_summary*.json`.
- `scripts/state_accuracy_bootstrap.py` gives confidence intervals for state accuracy, and
  `scripts/judge_bootstrap.py` gives them for the judge score and ROUGE-1. Both resample
  dialogues and both can compare two runs paired by dialogue. Intervals are not stored here.
- This folder is built from the run archives by `scripts/build_hf_run_release.py`, which
  checks that the row counts match each run's `metrics_summary.json`.

## Caveats

1. **Run `r1-…` used a classifier that had been trained on 607 of the 681 test dialogues.**
   It is included only to document that leak. Its state accuracy is not a result. Its judge
   summary was also written by a different judge model from the other runs, and its per-turn
   judge scores were not kept, so its judge columns are null. The run named `r1b-…` repeats
   it with a classifier retrained on the training part only.
2. **The Gemma 4 12B fine-tune (runs named `gemma12bsft`) was trained on turns from all 681
   test dialogues.** Its test-split run is contaminated. The 75-dialogue synthetic set is its
   only clean comparison with the base model.
3. **SocratTeachLLM's training data is not known to exclude the test split.** Treat its
   test-split scores, text-overlap scores above all, with that in mind.
4. **State accuracy on the synthetic set should not be reported.** The synthetic set's
   sub-state labels follow a different convention from SocratDataset, so the figure measures
   the labelling convention more than the consultant. The value stays in the files for
   completeness and is marked `state_accuracy_reportable = false`. The analysis is in
   `docs/SYNTHETIC_STATE_ACCURACY_NOTE.md` in the code repository.
5. **The judge is a language model.** The judge of record is Qwen3.6-27B (Q5_K_XL
   quantisation, served locally with llama.cpp); where present, Claude Sonnet 4.6 through
   Amazon Bedrock is a second judge. The judge has been checked only against deliberately
   broken control replies and against a second judge. A check against human scores is in
   progress and is not part of this release. Judge scores from different judge models are
   not comparable with each other.
6. **The runs were made on shared machines.** `run_config.json` records the serving settings
   in `env.KELE_RUN_NOTES`. Host names in that field were replaced, by exact string
   replacement and nothing else, with a description of the hardware: {substitutions}. Some
   runs were made in one-hour job slices and resumed from disk, so
   `total_elapsed_seconds`, `started_at` and `finished_at` in their `run_config.json` cover
   the last slice only. The machines ran different llama.cpp builds, recorded in the notes.

Further limits:

- Each run is scored on the turns it generated, and a dialogue stops when the consultant
  predicts `e34`. Runs therefore cover different turns. Compare runs by dialogue, and give
  the turn count beside any turn-level figure.
- The teacher samples without a fixed temperature, and the server's sampler seed was random
  unless the notes say otherwise. The two `r8-…` runs repeat the `r1b-…` configuration to
  show run-to-run variation; they are repeats, not reproducible seeded runs.
- The reference replies and reference states come from the source datasets. According to
  the source dataset cards, both sets of dialogues were generated with language models, so
  the references are not human gold labels.

## Source data

- [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset): Chinese
  elementary-school science dialogues from the KELE paper. Licence on its Hub card:
  `cc-by-4.0`.
- [ulises-c/SocratDataset-SYNTHETIC](https://huggingface.co/datasets/ulises-c/SocratDataset-SYNTHETIC):
  75 synthetic dialogues generated with Claude Sonnet 4.6 for this project. Licence on its
  Hub card: `cc-by-4.0`.

The questions, student turns, reference replies and reference states in this dataset are
copied from those two datasets.

## Licence

**TODO (owner): choose the licence of this dataset before publishing.** It has not been
decided. This dataset copies text from the two source datasets above, both listed as
`cc-by-4.0` on the Hub, so their attribution terms apply to that text. SocratDataset was
first distributed by the KELE authors in their GitHub repository
(https://github.com/yuanpan1020/KELE), which had no licence file when checked on 2026-11-20;
the `cc-by-4.0` label was set on the Hub copy. Cite the KELE paper (Peng et al., Findings of
EMNLP 2025, https://aclanthology.org/2025.findings-emnlp.888/) when using the reference
dialogues.

## Contact

Questions and corrections: {GITHUB_URL}/issues
"""
    return front + body


# --------------------------------------------------------------------------- output


def write_runs_csv(results: list[RunResult], path: Path) -> None:
    columns = list(results[0].row)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for result in results:
        writer.writerow([_fmt(result.row.get(column)) for column in columns])
    path.write_bytes(buffer.getvalue().encode("utf-8"))


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def self_check(
    out: Path, results: list[RunResult], shorts: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Problems and warnings found in the staged tree, read back from disk.

    Any problem fails the build. A judge pass whose summary is marked incomplete may
    cover turns whose per-turn scores were never saved; that is reported as a warning.
    """
    problems: list[str] = []
    warnings: list[str] = []
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out).as_posix()
        for name in find_forbidden(rel):
            problems.append(f"forbidden identifier {name!r} in the file name {rel}")
        if path.is_file():
            for name in find_forbidden(path.read_text(encoding="utf-8")):
                problems.append(f"forbidden identifier {name!r} in {rel}")

    expected_columns = [name for name, _ in turn_columns(shorts)]
    for result in results:
        run = result.run
        turns_path = out / "data" / run / "turns.jsonl"
        dialogues_path = out / "data" / run / "dialogues.jsonl"
        metrics = json.loads((out / "summaries" / run / "metrics_summary.json").read_text("utf-8"))
        config = json.loads((out / "summaries" / run / "run_config.json").read_text("utf-8"))
        rows = [json.loads(line) for line in turns_path.read_text("utf-8").splitlines()]
        if not rows:
            problems.append(f"{run}: no turns")
            continue
        if len(rows) != metrics.get("n_turns"):
            problems.append(
                f"{run}: {len(rows)} turn rows, metrics_summary.json says {metrics.get('n_turns')}"
            )
        if any(list(row) != expected_columns for row in rows):
            problems.append(f"{run}: turn rows do not all have the shared columns")
        n_dialogues = _count_lines(dialogues_path)
        if "completed" in config and n_dialogues != config["completed"]:
            problems.append(
                f"{run}: {n_dialogues} dialogue rows, run_config.json says {config['completed']}"
            )
        if len({(row["dialogue_file"], row["turn_idx"]) for row in rows}) != len(rows):
            problems.append(f"{run}: duplicate turn rows")
        for stem, short in shorts.items():
            totals = [row[f"judge_{short}_total"] for row in rows]
            scored = [total for total in totals if total is not None]
            summary_path = out / "summaries" / run / f"{stem}.json"
            if not scored:
                continue
            if not summary_path.exists():
                problems.append(f"{run}: per-turn {stem} scores without {stem}.json")
                continue
            summary = json.loads(summary_path.read_text("utf-8"))
            if len(scored) != summary.get("n_turns_judged"):
                message = (
                    f"{run}: {len(scored)} per-turn {stem} scores, its summary says "
                    f"{summary.get('n_turns_judged')} turns judged"
                )
                (problems if summary.get("complete") else warnings).append(message)
            elif abs(sum(scored) / len(scored) - float(summary.get("overall_avg", -1))) > 1e-9:
                problems.append(
                    f"{run}: mean of {stem} scores is {sum(scored) / len(scored):.6f}, "
                    f"its summary says {summary.get('overall_avg')}"
                )
    return problems, warnings


def build(
    archive_dir: Path, overlays: list[Path], out: Path
) -> tuple[list[RunResult], list[str], list[str]]:
    """Stage everything under ``out``; return the run results, problems and warnings."""
    runs = load_runs(archive_dir, overlays)
    shorts = short_names_for(set().union(*(judge_stems(files) for files in runs.values())))
    out.mkdir(parents=True, exist_ok=True)
    results = [stage_run(run, files, shorts, out) for run, files in runs.items()]
    write_runs_csv(results, out / "runs.csv")
    (out / "README.md").write_text(build_readme(results, shorts), encoding="utf-8")
    problems, warnings = self_check(out, results, shorts)
    return results, problems, warnings


def tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--archive-dir", type=Path, required=True, help="folder of run tarballs")
    parser.add_argument(
        "--overlay",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="folder of RUN/ subfolders laid over the tarballs; may be given more than once",
    )
    parser.add_argument("--out", type=Path, required=True, help="folder to write the stage to")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace --out when it is not empty"
    )
    args = parser.parse_args(argv)

    out: Path = args.out.expanduser()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            print(f"error: {out} is not empty; pass --overwrite to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    try:
        results, problems, warnings = build(
            args.archive_dir.expanduser(), [p.expanduser() for p in args.overlay], out
        )
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for result in results:
        print(
            f"  {result.run}: {result.n_dialogues} dialogues, {result.n_turns} turns, "
            f"{result.row['status']}"
        )
    total_turns = sum(result.n_turns for result in results)
    size = tree_size(out)
    print(f"{len(results)} runs, {total_turns} turns, {size} bytes ({size / 1e6:.1f} MB) in {out}")
    for warning in warnings:
        print(f"warning: {warning}")
    sys.stdout.flush()
    if problems:
        print(f"SELF-CHECK FAILED ({len(problems)} problems):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("self-check passed: no internal identifiers, row counts match the run summaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
