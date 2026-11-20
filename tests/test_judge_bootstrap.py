"""Tests for scripts/judge_bootstrap.py on small synthetic runs (no model calls)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.project.metrics import compute_rouge, extract_predictions_and_references

_spec = importlib.util.spec_from_file_location(
    "judge_bootstrap",
    Path(__file__).resolve().parents[1] / "scripts" / "judge_bootstrap.py",
)
assert _spec is not None and _spec.loader is not None
jb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jb)

REPLIES = [
    "你觉得为什么会这样呢？",
    "我们先看看题目给了什么条件。",
    "What do you notice first?",
    "很好！",
]
REFS = [
    "你觉得原因是什么？",
    "题目告诉了我们哪些条件？",
    "What do you see in the picture?",
    "答案是化学变化。",
]


def _write_run(root: Path, scores: dict[str, list[float]], seed: int = 0) -> Path:
    """One dialogue per entry of scores; each score is one judged turn with a reply."""
    rng = np.random.default_rng(seed)
    (root / "dialogues").mkdir(parents=True)
    judge = root / "judge_scores" / "judge_summary"
    judge.mkdir(parents=True)
    for name, totals in scores.items():
        turns = [
            {
                "student": "s",
                "state": "a1",
                "teacher_response": REPLIES[int(rng.integers(len(REPLIES)))],
                "ground_truth_teacher": REFS[int(rng.integers(len(REFS)))],
                "ground_truth_state": "a1",
            }
            for _ in totals
        ]
        (root / "dialogues" / name).write_text(json.dumps({"id": name, "dialogue": turns}))
        entries = [{"turn_idx": i, "total": t} for i, t in enumerate(totals)]
        (judge / name).write_text(json.dumps({"id": name, "scores": entries}))
    return root


def _random_run(root: Path, n: int = 40, seed: int = 1, shift: float = 0.0) -> Path:
    rng = np.random.default_rng(seed)
    scores = {
        f"{d:04d}.json": [
            float(np.clip(rng.integers(4, 11) + shift, 0, 10)) for _ in range(rng.integers(2, 8))
        ]
        for d in range(n)
    }
    return _write_run(root, scores, seed)


def test_judge_point_estimate_is_the_turn_weighted_mean(tmp_path):
    run = _write_run(tmp_path / "run", {"0001.json": [10, 10, 10, 10], "0002.json": [0]})
    names, sums, counts = jb.load_judge(run)

    assert names == ["0001.json", "0002.json"]
    assert sums.tolist() == [40.0, 0.0]
    assert counts.tolist() == [4, 1]
    # 40 / 5 turns, not the mean of the two dialogue means (5.0).
    assert jb.bootstrap_run((names, sums, counts), n_boot=200)["estimate"] == pytest.approx(8.0)


def test_dot_files_and_entries_without_total_are_ignored(tmp_path):
    run = _write_run(tmp_path / "run", {"0001.json": [8, 6]})
    judge = run / "judge_scores" / "judge_summary"
    (judge / "._0001.json").write_bytes(b"\x00\x05\x16\x07not json")
    (run / "dialogues" / "._0001.json").write_bytes(b"\x00\x05\x16\x07not json")
    (judge / "0002.json").write_text(
        json.dumps({"id": 2, "scores": [{"turn_idx": 0, "total": None}, {"turn_idx": 1}]})
    )

    names, sums, counts = jb.load_judge(run)
    assert names == ["0001.json", "0002.json"]
    assert sums.tolist() == [14.0, 0.0]
    assert counts.tolist() == [2, 0]
    assert jb.load_rouge1(run)[0] == ["0001.json"]


def test_rouge1_matches_metrics_module(tmp_path):
    run = _random_run(tmp_path / "run")
    # metrics.py skips failed dialogues and turns with an empty reply or reference.
    (run / "dialogues" / "9000.json").write_text(json.dumps({"id": 9000, "error": "boom"}))
    (run / "dialogues" / "9001.json").write_text(
        json.dumps(
            {
                "id": 9001,
                "dialogue": [
                    {"teacher_response": "", "ground_truth_teacher": "参考"},
                    {"teacher_response": "回答", "ground_truth_teacher": "回答参考"},
                ],
            }
        )
    )

    names, sums, counts = jb.load_rouge1(run)
    preds, refs = extract_predictions_and_references(run / "dialogues")

    assert "9000.json" not in names
    assert counts.sum() == len(preds)
    assert sums.sum() / counts.sum() == pytest.approx(compute_rouge(preds, refs)["rouge1"])


def test_rouge1_uses_words_for_english_and_characters_for_chinese():
    english, chinese = jb.turn_rouge1([("the cat sat", "the dog sat"), ("你好吗", "你好")])
    assert english == pytest.approx(200 / 3)  # 2 of 3 words on each side
    assert chinese == pytest.approx(80.0)  # precision 2/3, recall 1


def test_interval_contains_estimate_and_is_reproducible(tmp_path):
    metric = jb.load_judge(_random_run(tmp_path / "run"))

    first = jb.bootstrap_run(metric, n_boot=2000, seed=3)
    again = jb.bootstrap_run(metric, n_boot=2000, seed=3)
    other = jb.bootstrap_run(metric, n_boot=2000, seed=4)

    assert first == again
    assert first["lo"] < first["estimate"] < first["hi"]
    assert first["estimate"] == other["estimate"]
    assert (first["lo"], first["hi"]) != (other["lo"], other["hi"])
    assert first["n_dialogues"] == 40
    assert first["n_undefined"] == 0


def test_resampling_matches_an_explicit_loop(tmp_path):
    """The matrix form equals drawing dialogue indices and pooling their turns."""
    names, sums, counts = jb.load_judge(_random_run(tmp_path / "run", n=12))
    rng = np.random.default_rng(5)
    draws_n = jb.resample_counts(len(names), 50, rng)
    assert draws_n.sum(axis=1).tolist() == [12] * 50

    fast = (draws_n @ sums) / (draws_n @ counts)
    slow = []
    for row in draws_n:
        idx = np.repeat(np.arange(len(names)), row)
        slow.append(sums[idx].sum() / counts[idx].sum())
    assert fast == pytest.approx(slow)


def test_identical_runs_have_zero_difference(tmp_path):
    a = jb.load_judge(_random_run(tmp_path / "a"))
    b = jb.load_judge(_random_run(tmp_path / "b"))

    diff = jb.bootstrap_paired(a, b, n_boot=500)["difference"]
    assert diff["estimate"] == 0.0
    assert diff["lo"] == 0.0 and diff["hi"] == 0.0


def test_paired_interval_is_narrower_than_unpaired_for_a_constant_shift(tmp_path):
    scores = {f"{d:04d}.json": [float(3 + d % 6)] * 4 for d in range(30)}
    a = jb.load_judge(
        _write_run(tmp_path / "a", {k: [x + 1 for x in v] for k, v in scores.items()})
    )
    b = jb.load_judge(_write_run(tmp_path / "b", scores))

    paired = jb.bootstrap_paired(a, b, n_boot=1000)
    diff = paired["difference"]
    assert diff["estimate"] == pytest.approx(1.0)
    # Every resample moves both runs together, so the difference never varies.
    assert diff["lo"] == pytest.approx(1.0) and diff["hi"] == pytest.approx(1.0)
    assert jb.bootstrap_run(a, n_boot=1000)["se"] > 0.1


def test_paired_uses_only_shared_dialogues(tmp_path):
    a = jb.load_judge(
        _write_run(tmp_path / "a", {"0001.json": [9, 9], "0002.json": [5], "0003.json": [0] * 5})
    )
    b = jb.load_judge(
        _write_run(tmp_path / "b", {"0001.json": [7], "0002.json": [5, 3], "0009.json": [10] * 5})
    )

    paired = jb.bootstrap_paired(a, b, n_boot=300)
    assert paired["n_shared_dialogues"] == 2
    assert paired["n_only_a"] == 1 and paired["n_only_b"] == 1
    assert paired["n_turns_a"] == 3 and paired["n_turns_b"] == 3
    assert paired["mean_a"] == pytest.approx(23 / 3)
    assert paired["mean_b"] == pytest.approx(5.0)
    assert paired["difference"]["estimate"] == pytest.approx(23 / 3 - 5.0)


def test_paired_without_shared_dialogues_fails(tmp_path):
    a = jb.load_judge(_write_run(tmp_path / "a", {"0001.json": [9]}))
    b = jb.load_judge(_write_run(tmp_path / "b", {"0002.json": [9]}))
    with pytest.raises(ValueError, match="share no dialogue"):
        jb.bootstrap_paired(a, b, n_boot=10)


def test_missing_folder_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        jb.load_judge(tmp_path)
    with pytest.raises(FileNotFoundError):
        jb.load_rouge1(tmp_path)


def test_cli_writes_json(tmp_path, monkeypatch, capsys):
    a = _random_run(tmp_path / "a", shift=1.0)
    b = _random_run(tmp_path / "b")
    out = tmp_path / "out" / "judge.json"
    argv = ["judge_bootstrap.py", str(a), str(b), "--paired", "--rouge", "--n-boot", "300"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])

    jb.main()

    data = json.loads(out.read_text())
    assert [r["run"] for r in data["runs"]] == ["a", "b"]
    assert data["n_boot"] == 300
    assert data["paired"]["judge"]["difference"]["estimate"] > 0
    assert data["paired"]["rouge1"]["difference"]["estimate"] == pytest.approx(0.0)
    assert "Difference a minus b" in capsys.readouterr().out


def test_cli_rejects_paired_with_one_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["judge_bootstrap.py", str(tmp_path), "--paired"])
    with pytest.raises(SystemExit):
        jb.main()
