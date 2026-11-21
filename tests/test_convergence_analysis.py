"""Tests for scripts/convergence_analysis.py on small synthetic runs (no model calls)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.project.metrics import compute_all_metrics

_spec = importlib.util.spec_from_file_location(
    "convergence_analysis",
    Path(__file__).resolve().parents[1] / "scripts" / "convergence_analysis.py",
)
assert _spec is not None and _spec.loader is not None
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

GT = ["a1", "b4", "c16", "d33", "e34"]
REPLIES = ["你觉得为什么会这样呢？", "我们先回顾一下这个概念。", "很好，请你总结一下。"]


def _write_run(root: Path, n: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    dialogues = root / "dialogues"
    dialogues.mkdir(parents=True)
    for d in range(n):
        turns = [
            {
                "student": "s",
                "state": g if rng.random() < 0.55 else "c9",
                "teacher_response": REPLIES[int(rng.integers(3))],
                "ground_truth_teacher": REPLIES[int(rng.integers(3))],
                "ground_truth_state": g,
            }
            for g in GT[: int(rng.integers(3, 6))]
        ]
        (dialogues / f"{d:04d}.json").write_text(json.dumps({"id": d, "dialogue": turns}))
    return root


def _binary_signals(n_dialogues: int = 200, turns: int = 4) -> dict:
    """Each dialogue is entirely right or entirely wrong; accuracy 50%."""
    right = (np.arange(n_dialogues) % 2).astype(float)
    return {"state_acc": (right * turns * 100.0, np.full(n_dialogues, float(turns)))}


def test_full_run_values_match_metrics_module(tmp_path):
    run = _write_run(tmp_path / "run")
    (run / "dialogues" / "9000.json").write_text(json.dumps({"id": 9000, "error": "boom"}))

    names, signals = ca.load_run(run)
    result = ca.bootstrap_run("r", signals, [10], b=50)
    expected = compute_all_metrics(run / "dialogues")

    assert len(names) == 60 and result["n_dialogues"] == 60
    assert round(result["truth"]["state_acc"], 2) == expected["state_accuracy"]["overall"]
    assert round(result["truth"]["rouge1"], 2) == expected["rouge1"]
    assert round(result["truth"]["rouge2"], 2) == expected["rouge2"]
    assert signals["state_acc"][1].sum() == expected["state_accuracy"]["total_turns"]
    assert signals["rouge1"][1].sum() == expected["n_turns"]


def test_state_only_skips_text_metrics(tmp_path):
    _names, signals = ca.load_run(_write_run(tmp_path / "run"), text_metrics=False)
    assert list(signals) == ["state_acc"]


def test_standard_error_follows_one_over_sqrt_n():
    result = ca.bootstrap_run("r", _binary_signals(), [50, 200, 800], b=8000, seed=1)

    # Dialogue-level accuracy is 0 or 100 with equal odds: SE = 50 / sqrt(n) points.
    for n in (50, 200, 800):
        s = result["per_n"][n]["state_acc"]
        assert s["se"] == pytest.approx(50.0 / np.sqrt(n), rel=0.05)
        assert s["mean"] == pytest.approx(50.0, abs=0.5)
        assert s["half_width_95"] == pytest.approx(1.96 * s["se"], rel=0.1)


def test_spread_does_not_vanish_at_full_size():
    # The flaw of the earlier version: sampling without replacement gave zero
    # spread once the subsample was the whole run.
    result = ca.bootstrap_run("r", _binary_signals(n_dialogues=100), [50], b=4000, seed=2)

    assert sorted(result["per_n"]) == [50, 100]  # the full size is added to the grid
    full = result["per_n"][100]["state_acc"]
    assert full["se"] == pytest.approx(5.0, rel=0.08)
    assert full["abs_dev_p95"] > 5.0
    assert full["p2_5"] < 50.0 < full["p97_5"]


def test_detectable_difference_rules():
    result = ca.bootstrap_run("r", _binary_signals(), [200], b=1000, seed=3)
    s = result["per_n"][200]["state_acc"]

    assert s["detectable_diff"]["vs_fixed"] == pytest.approx(2.8 * s["se"])
    assert s["detectable_diff"]["two_runs"] == pytest.approx(2.8 * np.sqrt(2) * s["se"])
    assert s["std"] == s["se"]


def test_same_seed_gives_same_numbers():
    signals = _binary_signals()
    assert ca.bootstrap_run("r", signals, [30], b=300, seed=7) == ca.bootstrap_run(
        "r", signals, [30], b=300, seed=7
    )


def test_rejects_non_positive_sample_size():
    with pytest.raises(ValueError, match="positive"):
        ca.bootstrap_run("r", _binary_signals(), [0], b=10)


def test_report_states_what_each_n_can_detect():
    result = ca.bootstrap_run("r", _binary_signals(), [100], b=500, seed=0)

    report = ca.format_report(result)

    assert "n=100:" in report and "n=200:" in report
    assert "rules of thumb" in report
    assert "with replacement" in report


def test_cli_writes_per_run_and_aggregate_files(tmp_path, monkeypatch, capsys):
    run = _write_run(tmp_path / "my-run", n=30)
    out_dir = tmp_path / "out"
    argv = ["x", str(run), "--n-grid", "10,20", "--n-boot", "200", "--seed", "5"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-dir", str(out_dir)])

    ca.main()

    per_run = json.loads((out_dir / "my-run" / "convergence.json").read_text())
    aggregate = json.loads((out_dir / "aggregate.json").read_text())
    assert sorted(per_run["per_n"]) == ["10", "20", "30"]
    assert set(per_run["truth"]) == {"state_acc", "rouge1", "rouge2", "bleu4_sent"}
    assert aggregate["B"] == 200 and aggregate["rng_seed"] == 5
    assert aggregate["runs"]["my-run"] == per_run
    # Keys read by scripts/convergence_summary.py are still present.
    assert "abs_dev_p95" in per_run["per_n"]["10"]["bleu4_sent"]
    assert "State accuracy, in plain words" in capsys.readouterr().out
