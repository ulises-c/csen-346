"""Tests for scripts/build_hf_run_release.py on small made-up run archives (no network)."""

import csv
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "build_hf_run_release",
    Path(__file__).resolve().parents[1] / "scripts" / "build_hf_run_release.py",
)
assert _spec is not None and _spec.loader is not None
rel = importlib.util.module_from_spec(_spec)
sys.modules["build_hf_run_release"] = rel
_spec.loader.exec_module(rel)

CLEAN = "r1b-qwen35clfzh-gemma31b-fewshot10-n681-seed42"
LEAK = "r1-qwen35clf-gemma31b-fewshot10-n681-seed42"
SFT_TEST = "r6-qwen35clfzh-gemma12bsft-fewshot10-n681-seed42"
SFT_SYNTH = "r6-synth75-qwen35clfzh-gemma12bsft-fewshot10"
STL_TEST = "r4-qwen35clfzh-socratteachllm-fewshot10-n681-seed42"
SONNET = "judge_summary_bedrock-claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def _small_datasets(monkeypatch):
    """Let a two-dialogue run stand in for the 681 and 75 dialogue sets."""
    monkeypatch.setattr(
        rel,
        "DATASETS",
        {
            "test-split-681": ("ulises-c/SocratDataset", 2),
            "synthetic-75": ("ulises-c/SocratDataset-SYNTHETIC", 2),
        },
    )


def _dialogue(idx: int, n_turns: int, answer: object) -> dict:
    return {
        "id": idx,
        "question": f"题目 {idx}",
        "answer": answer,
        "num_turns_ground_truth": n_turns + (idx % 2),
        "num_turns_generated": n_turns,
        "dialogue": [
            {
                "student": f"学生 {idx}-{t}",
                "state": "a1" if t == 0 else "c12",
                "teacher_response": f"老师 {idx}-{t}？",
                "ground_truth_teacher": f"参考 {idx}-{t}？",
                "ground_truth_state": "a1" if t == 0 else "c16",
            }
            for t in range(n_turns)
        ],
        "elapsed_seconds": 1.5 * idx,
    }


def _scores(dialogue: dict, model: str, base: int) -> dict:
    return {
        "judge_model": model,
        "id": dialogue["id"],
        "scores": [
            {
                "turn_idx": t,
                "total": base + t,
                "socratic_validity": 3,
                "advancement": 2,
                "age_appropriateness": 2,
                "question_form": base + t - 7,
                "comment": f"comment {t}",
                "ground_truth_state": turn["ground_truth_state"],
                "usage_in": 500,
                "usage_out": 90,
            }
            for t, turn in enumerate(dialogue["dialogue"])
        ],
    }


def _judge_summary(run: str, model: str, score_files: list[dict], n_turns: int) -> dict:
    totals = [s["total"] for f in score_files for s in f["scores"]]
    return {
        "results_dir": f"results/{run}",
        "judge_model": model,
        "n_turns_judged": len(totals),
        "n_turns_total": n_turns,
        "complete": len(totals) == n_turns,
        "overall_avg": sum(totals) / len(totals),
    }


def _run_files(
    run: str,
    *,
    teacher: str = "Gemma 4 31B",
    checkpoint: str | None = "results/state-clf-qwen3.5-0.8b-lora-zh/final",
    host: str = "lz-hw-0047",
    judge_model: str = "Qwen3.6-27B-Q5_K_XL",
    per_turn_scores: bool = True,
    answer: object = 2,
) -> dict[str, bytes]:
    dialogues = [_dialogue(4, 3, answer), _dialogue(15, 2, answer)]
    n_turns = sum(len(d["dialogue"]) for d in dialogues)
    files: dict[str, object] = {}
    score_files = []
    for dialogue in dialogues:
        name = f"{dialogue['id']:04d}.json"
        files[f"dialogues/{name}"] = dialogue
        score_files.append(_scores(dialogue, judge_model, 7))
        if per_turn_scores:
            files[f"judge_scores/judge_summary/{name}"] = score_files[-1]
    files["judge_summary.json"] = _judge_summary(run, judge_model, score_files, n_turns)
    files["metrics_summary.json"] = {
        "n_turns": n_turns,
        "rouge1": 36.86,
        "rouge2": 16.49,
        "rougeL": 28.08,
        "bleu4": 9.01,
        "state_accuracy": {"overall": 40.0, "total_turns": n_turns},
    }
    files["run_config.json"] = {
        "experiment": run,
        "teacher_model": teacher,
        "consultant_model": teacher,
        "bert_consultant": checkpoint,
        "oracle_consultant": False,
        "total_dialogues": 2,
        "completed": 2,
        "git_commit": "31949d330ce804126b45c62620ba48d04a583766",
        "git_dirty": False,
        "env": {
            "KELE_RUN_NOTES": f"llama.cpp build 12138; teacher model-Q5.gguf; host {host}",
        },
    }
    encoded = {
        name: json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        for name, value in files.items()
    }
    encoded["progress.log"] = b"started on lz-hw-0047 by uchavarria\n"
    return encoded


def _write_tarball(archive: Path, run: str, files: dict[str, bytes]) -> None:
    archive.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive / f"{run}.tgz", "w:gz") as tar:
        # macOS tar leaves "._" companions beside real files; the build must skip them.
        strays = {"dialogues/._0004.json": b"\x00\x05junk", "._run_config.json": b"\x00junk"}
        for name, data in {**files, **strays}.items():
            info = tarfile.TarInfo(f"{run}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _write_overlay(overlay: Path, run: str, base: dict[str, bytes], n_dialogues: int = 2) -> None:
    dialogues = [json.loads(base[k]) for k in sorted(base) if k.startswith("dialogues/")]
    n_turns = sum(len(d["dialogue"]) for d in dialogues)
    score_files = []
    for dialogue in dialogues[:n_dialogues]:
        score_files.append(_scores(dialogue, "us.anthropic.claude-sonnet-4-6", 6))
        path = overlay / run / "judge_scores" / SONNET / f"{dialogue['id']:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(score_files[-1]), encoding="utf-8")
        (path.parent / f"._{path.name}").write_bytes(b"\x00junk")
    summary = _judge_summary(run, "us.anthropic.claude-sonnet-4-6", score_files, n_turns)
    (overlay / run / f"{SONNET}.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    _write_tarball(root, CLEAN, _run_files(CLEAN))
    _write_tarball(
        root,
        LEAK,
        _run_files(
            LEAK,
            checkpoint="results/state-clf-qwen3.5-0.8b-lora-wandb/final",
            host="cachy",
            judge_model="Qwen3.8-27B-Q5_K_M",
            per_turn_scores=False,
        ),
    )
    _write_tarball(root, SFT_TEST, _run_files(SFT_TEST, teacher="Gemma 4 12B SFT"))
    _write_tarball(
        root, SFT_SYNTH, _run_files(SFT_SYNTH, teacher="Gemma 4 12B SFT", answer="会生锈")
    )
    _write_tarball(root, STL_TEST, _run_files(STL_TEST, teacher="SocratTeachLLM", host="hw-0304"))
    # Not runs: other tarballs in the same folder are left alone.
    (root / "state-clf-bge-small-zh-20261118.tgz").write_bytes(b"not a tarball")
    (root / "._r9-stray.tgz").write_bytes(b"\x00junk")
    return root


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _runs_csv(out: Path) -> dict[str, dict[str, str]]:
    with (out / "runs.csv").open(encoding="utf-8", newline="") as handle:
        return {row["run"]: row for row in csv.DictReader(handle)}


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_judge_short_names():
    assert rel.judge_short_name("judge_summary") == "qwen36"
    assert rel.judge_short_name(SONNET) == "sonnet46_bedrock"
    assert rel.judge_short_name("judge_summary_claude-sonnet-4-6") == "claude_sonnet_4_6"
    assert rel.judge_short_name("judge_summary_GPT 5.5 (high)") == "gpt_5_5_high"
    with pytest.raises(rel.BuildError):
        rel.short_names_for({"judge_summary_a-b", "judge_summary_a.b"})


def test_turns_have_one_row_per_turn_and_one_schema(archive: Path, tmp_path: Path):
    overlay = tmp_path / "overlay"
    _write_overlay(overlay, CLEAN, _run_files(CLEAN), n_dialogues=1)
    out = tmp_path / "stage"
    results, problems, warnings = rel.build(archive, [overlay], out)
    assert problems == [] and warnings == []
    assert [r.run for r in results] == sorted([CLEAN, LEAK, SFT_TEST, SFT_SYNTH, STL_TEST])

    expected = [
        "run",
        "dialogue_id",
        "dialogue_file",
        "turn_idx",
        "question",
        "student",
        "predicted_state",
        "ground_truth_state",
        "teacher_response",
        "ground_truth_teacher",
        *rel.judge_columns("qwen36"),
        *rel.judge_columns("sonnet46_bedrock"),
    ]
    for result in results:
        rows = _rows(out / "data" / result.run / "turns.jsonl")
        assert len(rows) == 5
        assert all(list(row) == expected for row in rows)

    rows = _rows(out / "data" / CLEAN / "turns.jsonl")
    first = rows[0]
    assert (first["dialogue_id"], first["dialogue_file"], first["turn_idx"]) == (4, "0004.json", 0)
    assert first["student"] == "学生 4-0" and first["teacher_response"] == "老师 4-0？"
    assert first["predicted_state"] == "a1" and rows[1]["ground_truth_state"] == "c16"
    assert [row["judge_qwen36_total"] for row in rows] == [7, 8, 9, 7, 8]
    assert rows[1]["judge_qwen36_comment"] == "comment 1"
    # The second judge scored the first dialogue only: null elsewhere.
    assert [row["judge_sonnet46_bedrock_total"] for row in rows] == [6, 7, 8, None, None]
    # A run without that judge, or without per-turn scores, has the columns as nulls.
    leak_rows = _rows(out / "data" / LEAK / "turns.jsonl")
    assert all(row["judge_qwen36_total"] is None for row in leak_rows)
    assert all(row["judge_sonnet46_bedrock_comment"] is None for row in leak_rows)

    dialogues = _rows(out / "data" / SFT_SYNTH / "dialogues.jsonl")
    assert dialogues[0] == {
        "run": SFT_SYNTH,
        "id": 4,
        "dialogue_file": "0004.json",
        "question": "题目 4",
        "answer": "会生锈",
        "num_turns_ground_truth": 3,
        "num_turns_generated": 3,
        "elapsed_seconds": 6.0,
    }
    # Integer answers of the other source dataset are written as text too.
    assert _rows(out / "data" / CLEAN / "dialogues.jsonl")[0]["answer"] == "2"


def test_runs_csv_statuses_and_numbers(archive: Path, tmp_path: Path):
    overlay = tmp_path / "overlay"
    _write_overlay(overlay, CLEAN, _run_files(CLEAN), n_dialogues=1)
    out = tmp_path / "stage"
    rel.build(archive, [overlay], out)
    rows = _runs_csv(out)

    assert rows[CLEAN]["status"] == "reportable"
    assert rows[LEAK]["status"] == "leak-documentation-only"
    assert rows[SFT_TEST]["status"] == "contaminated-test-split"
    assert rows[SFT_SYNTH]["status"] == "reportable"
    assert rows[STL_TEST]["status"] == "reportable-teacher-training-overlap-unknown"
    assert rows[SFT_SYNTH]["state_accuracy_reportable"] == "false"
    assert rows[SFT_TEST]["state_accuracy_reportable"] == "false"
    assert rows[LEAK]["state_accuracy_reportable"] == "false"
    assert rows[CLEAN]["state_accuracy_reportable"] == "true"

    clean = rows[CLEAN]
    assert "Qwen3.5-0.8B LoRA" in clean["consultant"] and "clean" in clean["consultant"]
    assert clean["teacher"] == "Gemma 4 31B"
    assert clean["teacher_weights_file"] == "model-Q5.gguf"
    assert (clean["dataset"], clean["source_dataset"]) == (
        "test-split-681",
        "ulises-c/SocratDataset",
    )
    assert rows[SFT_SYNTH]["dataset"] == "synthetic-75"
    assert (clean["n_dialogues"], clean["n_turns"], clean["n_ground_truth_turns"]) == (
        "2",
        "5",
        "6",
    )
    # 2 of 5 generated turns are right (the two opening turns); 6 reference turns.
    assert clean["state_accuracy"] == "40.0"
    assert clean["state_accuracy_all_ground_truth_turns"] == "33.33"
    assert clean["pct_dialogues_cut_short"] == "50.0"
    assert (clean["rouge1"], clean["bleu4"]) == ("36.86", "9.01")
    assert clean["git_commit"] == "31949d330ce804126b45c62620ba48d04a583766"
    assert float(clean["judge_qwen36_mean"]) == pytest.approx(7.8)
    assert clean["judge_qwen36_complete"] == "true"
    assert float(clean["judge_sonnet46_bedrock_mean"]) == pytest.approx(7.0)
    assert clean["judge_sonnet46_bedrock_complete"] == "false"
    assert clean["judge_sonnet46_bedrock_per_turn_scores"] == "3"
    assert clean["run_notes"].endswith("host rtx-a6000-node")
    assert rows[SFT_TEST]["judge_sonnet46_bedrock_mean"] == ""

    # Run 1 was judged by another model under the same file name: its mean is not listed.
    leak = rows[LEAK]
    assert leak["judge_qwen36_model"] == "Qwen3.8-27B-Q5_K_M"
    assert leak["judge_qwen36_mean"] == ""
    assert "607 of the 681" in leak["notes"] and "Qwen3.8-27B" in leak["notes"]
    assert "trained on test dialogues" in leak["consultant"]
    assert leak["run_notes"].endswith("host rx-9700-node")


def test_consultant_and_dataset_rules():
    config = {"teacher_model": "Gemma 4 31B", "consultant_model": "Gemma 4 31B"}
    assert "LLM consultant, same model" in rel.describe_consultant(config)
    assert rel.describe_consultant({**config, "oracle_consultant": True}).startswith("oracle")
    assert "bge-small" in rel.describe_consultant(
        {"bert_consultant": "results/state-clf-bge-small-zh/final"}
    )
    assert rel.describe_consultant({"bert_consultant": "results/other/final"}) == (
        "classifier checkpoint other"
    )
    with pytest.raises(rel.BuildError):
        rel.detect_dataset("r7-synth75-x", 5)
    with pytest.raises(rel.BuildError):
        rel.detect_dataset("r9-something", 2)


def test_summaries_differ_from_the_archive_only_by_host_names(archive: Path, tmp_path: Path):
    out = tmp_path / "stage"
    rel.build(archive, [], out)
    originals = {
        CLEAN: _run_files(CLEAN),
        STL_TEST: _run_files(STL_TEST, teacher="SocratTeachLLM", host="hw-0304"),
    }
    for run, files in originals.items():
        staged_names = sorted(p.name for p in (out / "summaries" / run).iterdir())
        assert staged_names == ["judge_summary.json", "metrics_summary.json", "run_config.json"]
        for name in staged_names:
            staged = (out / "summaries" / run / name).read_bytes()
            assert rel.is_substitution_only(files[name], staged)
            if name != "run_config.json":
                assert staged == files[name]

    # Reversing the one replacement by hand gives the archived bytes back.
    staged = (out / "summaries" / CLEAN / "run_config.json").read_bytes()
    assert staged != originals[CLEAN]["run_config.json"]
    assert staged.replace(b"rtx-a6000-node", b"lz-hw-0047") == originals[CLEAN]["run_config.json"]
    notes = json.loads(staged)["env"]["KELE_RUN_NOTES"]
    assert notes == "llama.cpp build 12138; teacher model-Q5.gguf; host rtx-a6000-node"
    stl = json.loads((out / "summaries" / STL_TEST / "run_config.json").read_text("utf-8"))
    assert stl["env"]["KELE_RUN_NOTES"].endswith("host rtx-4000-ada-node")


def test_substitution_check_rejects_any_other_change():
    original = b'{"notes": "host lz-hw-0047, also hw-0047 and ollie-cachy", "n": 4120}'
    staged = rel.substitute_hosts(original)
    assert staged == (
        b'{"notes": "host rtx-a6000-node, also rtx-a6000-node and rx-9700-node", "n": 4120}'
    )
    assert rel.is_substitution_only(original, staged)
    assert rel.is_substitution_only(b"host cachy", rel.substitute_hosts(b"host cachy"))
    assert rel.is_substitution_only(b'{"n": 1}', b'{"n": 1}')
    assert not rel.is_substitution_only(original, staged.replace(b"4120", b"4121"))
    assert not rel.is_substitution_only(original, staged + b"\n")
    assert not rel.is_substitution_only(original, original)  # a host name is left
    # A hardware name already in the original cannot be told from a replacement.
    assert not rel.is_substitution_only(b"rtx-a6000-node x", b"rtx-a6000-node x")


def test_readme_front_matter_defines_configs_that_match_the_files(archive: Path, tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "stage"
    results, _, _ = rel.build(archive, [], out)
    text = (out / "README.md").read_text("utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n")[1])
    assert "license" not in front
    configs = {c["config_name"]: c["data_files"][0] for c in front["configs"]}
    assert configs["turns"] == {"split": "train", "path": "data/*/turns.jsonl"}
    assert configs["dialogues"] == {"split": "train", "path": "data/*/dialogues.jsonl"}
    # One repository loads with one builder, so the CSV index is not a config.
    assert sorted(configs) == ["dialogues", "turns"]
    for name in ("turns", "dialogues"):
        assert len(list(out.glob(configs[name]["path"]))) == len(results)
    features = {c["config_name"]: [f["name"] for f in c["features"]] for c in front["dataset_info"]}
    assert features["turns"] == list(_rows(out / "data" / CLEAN / "turns.jsonl")[0])
    assert features["dialogues"] == list(_rows(out / "data" / CLEAN / "dialogues.jsonl")[0])

    body = text.split("---\n", 2)[2]
    for run in (CLEAN, LEAK, SFT_TEST, SFT_SYNTH, STL_TEST):
        assert f"| `{run}` |" in body
    for needle in (
        "607 of the 681",
        "gemma12bsft",
        "not known to exclude",
        "SYNTHETIC_STATE_ACCURACY_NOTE.md",
        "human scores is in",
        "rtx-a6000-node",
        "cc-by-4.0",
        "TODO (owner)",
        "https://github.com/ulises-c/csen-346/issues",
        "scripts/state_accuracy_bootstrap.py",
        "scripts/judge_bootstrap.py",
        "scripts/llm_judge_eval.py",
        "ulises-c/SocratDataset-SYNTHETIC",
    ):
        assert needle in body, needle
    # The synthetic run's state accuracy is not shown in the card.
    synth_line = next(line for line in body.splitlines() if line.startswith(f"| `{SFT_SYNTH}`"))
    assert "not reported" in synth_line and "40.00" not in synth_line


def test_build_is_deterministic_and_ignores_dot_files(archive: Path, tmp_path: Path):
    rel.build(archive, [], tmp_path / "one")
    rel.build(archive, [], tmp_path / "two")
    assert _tree_hash(tmp_path / "one") == _tree_hash(tmp_path / "two")
    names = [p.name for p in (tmp_path / "one").rglob("*")]
    assert not any(name.startswith(".") for name in names)
    assert "progress.log" not in names
    assert sorted(p.name for p in (tmp_path / "one" / "data").iterdir()) == sorted(
        [CLEAN, LEAK, SFT_TEST, SFT_SYNTH, STL_TEST]
    )


def test_staged_tree_has_no_forbidden_identifier(archive: Path, tmp_path: Path):
    out = tmp_path / "stage"
    _, problems, _ = rel.build(archive, [], out)
    assert problems == []
    for path in out.rglob("*"):
        if path.is_file():
            assert rel.find_forbidden(path.read_text("utf-8")) == [], path


@pytest.mark.parametrize(
    "leak",
    ["/home/someone/run", "/Users/someone/x", "work-laptop", "M4-MINI", "lazamedical", "hw-0457"],
)
def test_self_check_fails_on_an_identifier_in_the_data(tmp_path: Path, leak: str, capsys):
    files = _run_files(CLEAN)
    scores = json.loads(files["judge_scores/judge_summary/0004.json"])
    scores["scores"][0]["comment"] = f"see {leak}"
    files["judge_scores/judge_summary/0004.json"] = json.dumps(scores).encode("utf-8")
    _write_tarball(tmp_path / "archive", CLEAN, files)
    code = rel.main(["--archive-dir", str(tmp_path / "archive"), "--out", str(tmp_path / "stage")])
    assert code == 1
    assert "forbidden identifier" in capsys.readouterr().err


def test_self_check_fails_on_an_identifier_in_a_file_name(archive: Path, tmp_path: Path):
    out = tmp_path / "stage"
    results, problems, _ = rel.build(archive, [], out)
    assert problems == []
    (out / "summaries" / CLEAN / "notes-ollie.json").write_text("{}", encoding="utf-8")
    shorts = rel.short_names_for({"judge_summary"})
    problems, _ = rel.self_check(out, results, shorts)
    assert any("file name" in problem and "ollie" in problem for problem in problems)


def test_self_check_fails_when_rows_disagree_with_metrics_summary(tmp_path: Path, capsys):
    files = _run_files(CLEAN)
    metrics = json.loads(files["metrics_summary.json"])
    metrics["n_turns"] = 6
    files["metrics_summary.json"] = json.dumps(metrics).encode("utf-8")
    _write_tarball(tmp_path / "archive", CLEAN, files)
    code = rel.main(["--archive-dir", str(tmp_path / "archive"), "--out", str(tmp_path / "stage")])
    assert code == 1
    assert "5 turn rows, metrics_summary.json says 6" in capsys.readouterr().err


def test_self_check_fails_on_a_run_without_turns(tmp_path: Path, capsys):
    files = _run_files(CLEAN)
    for name in ("dialogues/0004.json", "dialogues/0015.json"):
        dialogue = json.loads(files[name])
        dialogue["dialogue"] = []
        files[name] = json.dumps(dialogue).encode("utf-8")
    files = {k: v for k, v in files.items() if not k.startswith("judge_scores/")}
    _write_tarball(tmp_path / "archive", CLEAN, files)
    code = rel.main(["--archive-dir", str(tmp_path / "archive"), "--out", str(tmp_path / "stage")])
    assert code == 1
    assert "no turns" in capsys.readouterr().err


def test_self_check_compares_per_turn_scores_with_the_judge_summary(tmp_path: Path, capsys):
    files = _run_files(CLEAN)
    summary = json.loads(files["judge_summary.json"])
    summary["overall_avg"] = 9.0
    files["judge_summary.json"] = json.dumps(summary).encode("utf-8")
    _write_tarball(tmp_path / "archive", CLEAN, files)
    code = rel.main(["--archive-dir", str(tmp_path / "archive"), "--out", str(tmp_path / "stage")])
    assert code == 1
    assert "mean of judge_summary scores is 7.8" in capsys.readouterr().err


def test_incomplete_judge_pass_with_fewer_saved_scores_is_a_warning(archive: Path, tmp_path: Path):
    overlay = tmp_path / "overlay"
    _write_overlay(overlay, CLEAN, _run_files(CLEAN), n_dialogues=1)
    summary_path = overlay / CLEAN / f"{SONNET}.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["n_turns_judged"] = 4  # one more than the per-turn scores that were saved
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _, problems, warnings = rel.build(archive, [overlay], tmp_path / "stage")
    assert problems == []
    assert len(warnings) == 1 and "3 per-turn" in warnings[0] and "4 turns judged" in warnings[0]
    readme = (tmp_path / "stage" / "README.md").read_text("utf-8")
    assert "no, 4 turns, 3 per-turn scores here" in readme


def test_judge_score_for_a_missing_turn_stops_the_build(archive: Path, tmp_path: Path):
    overlay = tmp_path / "overlay"
    _write_overlay(overlay, CLEAN, _run_files(CLEAN))
    path = overlay / CLEAN / "judge_scores" / SONNET / "0015.json"
    scores = json.loads(path.read_text("utf-8"))
    scores["scores"][0]["turn_idx"] = 9
    path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(rel.BuildError, match="no matching turn"):
        rel.build(archive, [overlay], tmp_path / "stage")


def test_main_prints_a_summary_and_refuses_a_non_empty_folder(
    archive: Path, tmp_path: Path, capsys
):
    out = tmp_path / "stage"
    assert rel.main(["--archive-dir", str(archive), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "5 runs, 25 turns" in printed and "self-check passed" in printed
    first = _tree_hash(out)
    assert rel.main(["--archive-dir", str(archive), "--out", str(out)]) == 2
    assert rel.main(["--archive-dir", str(archive), "--out", str(out), "--overwrite"]) == 0
    assert _tree_hash(out) == first
    assert rel.main(["--archive-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "x")]) == 1
