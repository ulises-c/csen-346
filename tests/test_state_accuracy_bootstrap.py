"""Tests for scripts/state_accuracy_bootstrap.py on small synthetic runs (no model calls)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.project.metrics import compute_state_accuracy

_spec = importlib.util.spec_from_file_location(
    "state_accuracy_bootstrap",
    Path(__file__).resolve().parents[1] / "scripts" / "state_accuracy_bootstrap.py",
)
assert _spec is not None and _spec.loader is not None
sab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sab)

GT = ["a1", "b4", "c16", "d33", "e34"]


def _write_run(root: Path, predictions: dict[str, list[str]], gt: list[str] | None = None) -> Path:
    """One dialogue per entry; each predicted state is paired with the GT state of its turn."""
    dialogues = root / "dialogues"
    dialogues.mkdir(parents=True)
    for name, preds in predictions.items():
        turns = [
            {"student": "s", "state": p, "teacher_response": "t", "ground_truth_state": g}
            for p, g in zip(preds, gt or GT)
        ]
        (dialogues / name).write_text(json.dumps({"id": int(name[:4]), "dialogue": turns}))
    return root


def _mixed_run(root: Path, n: int = 40, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    preds = {f"{d:04d}.json": [g if rng.random() < 0.6 else "c9" for g in GT] for d in range(n)}
    return _write_run(root, preds)


def test_point_estimates_match_metrics_module(tmp_path):
    run = _mixed_run(tmp_path / "run")
    # A failed dialogue and a turn without a ground-truth state are skipped by metrics.py.
    (run / "dialogues" / "9000.json").write_text(json.dumps({"id": 9000, "error": "boom"}))
    (run / "dialogues" / "9001.json").write_text(
        json.dumps(
            {
                "id": 9001,
                "dialogue": [
                    {"state": "a1", "ground_truth_state": "a1"},
                    {"state": "b4", "ground_truth_state": ""},
                ],
            }
        )
    )

    names, correct, total = sab.load_run(run)
    point = sab.accuracies(correct.sum(axis=0), total.sum(axis=0))
    expected = compute_state_accuracy(run / "dialogues")

    assert "9000.json" not in names
    assert total[:, 0].sum() == expected["total_turns"]
    assert point[0] == pytest.approx(expected["overall"])
    for i, stage in enumerate(sab.STAGES):
        assert point[i + 1] == pytest.approx(expected["per_stage"][stage])
    assert point[6] == pytest.approx(np.mean([expected["per_stage"][s] for s in sab.STAGES]))


def test_interval_contains_estimate_and_is_reproducible(tmp_path):
    _names, correct, total = sab.load_run(_mixed_run(tmp_path / "run"))

    first = sab.bootstrap_run(correct, total, n_boot=2000, seed=3)
    again = sab.bootstrap_run(correct, total, n_boot=2000, seed=3)

    assert first == again
    assert set(first) == set(sab.ROWS)
    for row in first.values():
        assert row["lo"] <= row["estimate"] <= row["hi"]
        assert row["n_undefined"] == 0
    assert first["overall"]["hi"] - first["overall"]["lo"] > 0


def test_resampling_is_by_dialogue_not_by_turn(tmp_path):
    # Every dialogue is all right or all wrong, so turns are perfectly correlated.
    # A turn-level bootstrap would see 200 independent turns; the dialogue-level
    # standard error must be that of 40 dialogues: sqrt(0.25 / 40) = 7.9 points.
    preds = {f"{d:04d}.json": ["c16" if d % 2 else "c9"] * 5 for d in range(40)}
    _names, correct, total = sab.load_run(_write_run(tmp_path / "run", preds, gt=["c16"] * 5))

    res = sab.bootstrap_run(correct, total, n_boot=5000, seed=0)

    assert res["overall"]["estimate"] == pytest.approx(50.0)
    assert res["overall"]["se"] == pytest.approx(7.9, abs=0.5)


def test_constant_stage_has_zero_width_interval(tmp_path):
    preds = {
        f"{d:04d}.json": ["a1", "b4" if d % 3 else "b5", "c16", "d33", "e34"] for d in range(30)
    }
    _names, correct, total = sab.load_run(_write_run(tmp_path / "run", preds))

    res = sab.bootstrap_run(correct, total, n_boot=500, seed=0)

    assert (res["a"]["estimate"], res["a"]["lo"], res["a"]["hi"]) == (100.0, 100.0, 100.0)
    assert res["b"]["lo"] < res["b"]["hi"]


def test_missing_stage_is_reported_as_undefined(tmp_path):
    # Only one dialogue of four reaches stage e, so many resamples have no stage-e turn.
    preds = {f"{d:04d}.json": GT[: 5 if d == 0 else 4] for d in range(4)}
    _names, correct, total = sab.load_run(_write_run(tmp_path / "run", preds))

    res = sab.bootstrap_run(correct, total, n_boot=400, seed=0)

    assert res["e"]["n_undefined"] > 0
    assert res["stage_balanced"]["n_undefined"] == res["e"]["n_undefined"]
    assert res["overall"]["n_undefined"] == 0


def test_paired_difference_uses_shared_dialogues_and_joint_resampling(tmp_path):
    rng = np.random.default_rng(5)
    base = {f"{d:04d}.json": [g if rng.random() < 0.5 else "c9" for g in GT] for d in range(60)}
    # B equals A except that B is always wrong on the stage-d turn; A is always right there.
    preds_a = {k: [*v[:3], "d33", v[4]] for k, v in base.items()}
    preds_b = {k: [*v[:3], "d30", v[4]] for k, v in base.items()}
    preds_a["7000.json"] = GT  # only in A
    preds_b["8000.json"] = GT  # only in B
    run_a = sab.load_run(_write_run(tmp_path / "a", preds_a))
    run_b = sab.load_run(_write_run(tmp_path / "b", preds_b))

    res = sab.bootstrap_paired(run_a, run_b, n_boot=1000, seed=0)
    diff = res["difference"]

    assert (res["n_shared_dialogues"], res["n_only_a"], res["n_only_b"]) == (60, 1, 1)
    # Each dialogue differs by exactly one of five turns, in every resample: no spread.
    for row in ("overall", "d", "stage_balanced"):
        assert diff[row]["lo"] == pytest.approx(diff[row]["estimate"])
        assert diff[row]["hi"] == pytest.approx(diff[row]["estimate"])
    assert diff["overall"]["estimate"] == pytest.approx(20.0)
    assert diff["d"]["estimate"] == pytest.approx(100.0)
    assert diff["stage_balanced"]["estimate"] == pytest.approx(20.0)
    assert diff["c"]["estimate"] == pytest.approx(0.0)


def test_paired_needs_shared_dialogues(tmp_path):
    run_a = sab.load_run(_write_run(tmp_path / "a", {"0001.json": GT}))
    run_b = sab.load_run(_write_run(tmp_path / "b", {"0002.json": GT}))

    with pytest.raises(ValueError, match="share no dialogue"):
        sab.bootstrap_paired(run_a, run_b, n_boot=10)


def test_cli_prints_table_and_writes_json(tmp_path, monkeypatch, capsys):
    run_a = _mixed_run(tmp_path / "run-a", seed=1)
    run_b = _mixed_run(tmp_path / "run-b", seed=2)
    out = tmp_path / "out" / "ci.json"
    argv = ["x", str(run_a), str(run_b), "--paired", "--n-boot", "200", "--seed", "1"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])

    sab.main()

    printed = capsys.readouterr().out
    saved = json.loads(out.read_text())
    assert "stage-balanced" in printed and "Difference run-a minus run-b" in printed
    assert [r["run"] for r in saved["runs"]] == ["run-a", "run-b"]
    assert saved["runs"][0]["n_dialogues"] == 40 and saved["runs"][0]["n_turns"] == 200
    assert saved["n_boot"] == 200 and saved["seed"] == 1
    assert set(saved["paired"]["difference"]) == set(sab.ROWS)


def test_empty_run_directory_is_an_error(tmp_path):
    (tmp_path / "dialogues").mkdir()
    with pytest.raises(FileNotFoundError):
        sab.load_run(tmp_path)


def _write_cut_run(root: Path, n_pairs: int = 30) -> Path:
    """Dialogues with 6 ground-truth turns: half stop after 4 turns, half run all 6.

    Every dialogue has 3 correct turns, so accuracy over all ground-truth turns
    is 3/6 in each of them.
    """
    dialogues = root / "dialogues"
    dialogues.mkdir(parents=True)
    short = [("a1", "a1"), ("b4", "b4"), ("c16", "c16"), ("e34", "d33")]
    full = [
        ("a1", "a1"),
        ("b4", "b4"),
        ("c16", "c16"),
        ("d30", "d33"),
        ("d30", "d33"),
        ("c9", "e34"),
    ]
    for d in range(2 * n_pairs):
        turns = [{"state": p, "ground_truth_state": g} for p, g in (short if d % 2 else full)]
        record = {"id": d, "num_turns_ground_truth": 6, "dialogue": turns}
        (dialogues / f"{d:04d}.json").write_text(json.dumps(record))
    return root


def test_ground_truth_turns_align_with_load_run(tmp_path):
    run = _write_cut_run(tmp_path / "run", n_pairs=2)
    (run / "dialogues" / "9000.json").write_text(
        json.dumps({"id": 9000, "num_turns_ground_truth": 9, "error": "boom"})
    )

    names, _correct, total = sab.load_run(run)
    gt = sab.load_ground_truth_turns(run)

    assert gt is not None and gt.tolist() == [6, 6, 6, 6]
    assert len(gt) == len(names) and total[:, 0].tolist() == [6, 4, 6, 4]


def test_ground_truth_turns_are_none_when_a_dialogue_lacks_the_count(tmp_path):
    run = _mixed_run(tmp_path / "run", n=4)

    assert sab.load_ground_truth_turns(run) is None


def test_all_turn_accuracy_counts_cut_off_turns_as_wrong(tmp_path):
    run = _write_cut_run(tmp_path / "run")
    _names, correct, total = sab.load_run(run)
    gt = sab.load_ground_truth_turns(run)

    rows = sab.bootstrap_early_stop(correct, total, gt, n_boot=2000, seed=0)

    assert list(rows) == sab.EARLY_STOP_ROWS
    assert rows["overall_all_turns"]["estimate"] == pytest.approx(100 * 180 / 360)
    assert rows["dialogues_cut_short"]["estimate"] == pytest.approx(50.0)
    assert rows["turns_not_scored"]["estimate"] == pytest.approx(100 * 60 / 360)
    # The scored-turn figure of the same run is higher: 180 of 300.
    assert sab.bootstrap_run(correct, total, 200)["overall"]["estimate"] == pytest.approx(60.0)
    # Every dialogue is 3 of 6, so resampling dialogues cannot move this figure.
    assert rows["overall_all_turns"]["hi"] == pytest.approx(rows["overall_all_turns"]["lo"])
    cut = rows["dialogues_cut_short"]
    assert cut["lo"] < cut["estimate"] < cut["hi"] and cut["hi"] - cut["lo"] > 10


def test_cli_reports_early_stop_rows_when_counts_exist(tmp_path, monkeypatch, capsys):
    run = _write_cut_run(tmp_path / "run")
    out = tmp_path / "ci.json"
    monkeypatch.setattr(sys, "argv", ["x", str(run), "--n-boot", "200", "--out", str(out)])

    sab.main()

    printed = capsys.readouterr().out
    saved = json.loads(out.read_text())["runs"][0]
    assert "all ground-truth turns" in printed and "dialogues cut short" in printed
    assert saved["n_turns_ground_truth"] == 360
    assert set(saved["early_stop"]) == set(sab.EARLY_STOP_ROWS)


def test_cli_says_so_when_ground_truth_counts_are_missing(tmp_path, monkeypatch, capsys):
    run = _mixed_run(tmp_path / "run")
    out = tmp_path / "ci.json"
    monkeypatch.setattr(sys, "argv", ["x", str(run), "--n-boot", "200", "--out", str(out)])

    sab.main()

    assert "num_turns_ground_truth" in capsys.readouterr().out
    assert "early_stop" not in json.loads(out.read_text())["runs"][0]
