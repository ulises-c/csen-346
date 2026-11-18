"""Tests for scripts/judge_validation.py with a scripted judge (no model calls)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_scripts = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("judge_validation", _scripts / "judge_validation.py")
assert _spec is not None and _spec.loader is not None
jv = importlib.util.module_from_spec(_spec)
sys.modules["judge_validation"] = jv
_spec.loader.exec_module(jv)

STAGES = ["a1", "b4", "c16", "d30", "e34"]


def _write_run(tmp_path, n_dialogues=8):
    dialogues = tmp_path / "dialogues"
    dialogues.mkdir()
    for d in range(n_dialogues):
        turns = [
            {
                "student": f"student {d}-{i}",
                "state": state,
                "ground_truth_state": state,
                "teacher_response": f"system reply {d}-{i}?",
                "ground_truth_teacher": f"reference reply {d}-{i}?",
            }
            for i, state in enumerate(STAGES)
        ]
        (dialogues / f"{d:04d}.json").write_text(json.dumps({"id": d, "dialogue": turns}))
    return tmp_path


def _args(run, **overrides):
    base = {
        "results_dir": run,
        "backend": "api",
        "model": "fake-judge",
        "base_url": "http://x",
        "api_key": "k",
        "workers": 2,
        "per_stage": 4,
        "stages": "bcde",
        "seed": 1,
        "repeats": 2,
        "n_boot": 200,
        "out": None,
        "dry_run": False,
    }
    return SimpleNamespace(**{**base, **overrides})


def _rubric(total_by_reply):
    """A judge that scores by the text of the reply it is shown."""

    def score_turn(client, model, dialogue, turn_idx, backend="api"):
        reply = dialogue["dialogue"][turn_idx]["teacher_response"]
        total = total_by_reply(reply, dialogue, turn_idx)
        if total is None:
            return None
        s = {"socratic_validity": 0, "advancement": 0, "age_appropriateness": 0, "question_form": 0}
        left = total
        for axis in jv.AXES:
            s[axis] = min(jv.AXIS_MAX[axis], left)
            left -= s[axis]
        return {**s, "total": total}

    return score_turn


def test_items_cover_every_control_that_applies(tmp_path):
    run = _write_run(tmp_path)

    items = jv.build_items(run / "dialogues", per_stage=4, stages="bcde", seed=1)

    by_variant = {v: [i for i in items if i["variant"] == v] for v in jv.VARIANTS}
    assert len(by_variant["system"]) == 16
    assert len(by_variant["reference"]) == 16
    assert len(by_variant["repeat"]) == 16
    # The closing summary is only brought forward to stages b and c.
    assert {i["stage"] for i in by_variant["premature_closure"]} == {"b", "c"}
    for item in by_variant["swapped"]:
        own = f"system reply {int(item['file'][:4])}-"
        assert not item["reply"].startswith(own)
    assert all(i["reply"] == jv.STATEMENT_REPLY for i in by_variant["statement"])


def test_sample_is_the_same_for_every_judge(tmp_path):
    run = _write_run(tmp_path)

    first = jv.build_items(run / "dialogues", 4, "bcde", seed=1)
    second = jv.build_items(run / "dialogues", 4, "bcde", seed=1)
    other = jv.build_items(run / "dialogues", 4, "bcde", seed=2)

    assert jv.sample_fingerprint(first) == jv.sample_fingerprint(second)
    assert jv.sample_fingerprint(first) != jv.sample_fingerprint(other)


def test_history_is_untouched_and_only_the_target_reply_changes(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    seen = []

    def score_turn(client, model, dialogue, turn_idx, backend="api"):
        seen.append((dialogue, turn_idx))
        return {**dict.fromkeys(jv.AXES, 1), "total": 4}

    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", score_turn)
    original = json.loads((run / "dialogues" / "0000.json").read_text())
    item = {"file": "0000.json", "turn": 2, "variant": "statement", "reply": "X"}

    jv.score_item(None, "m", "api", original, item)

    scored, idx = seen[0]
    assert scored["dialogue"][idx]["teacher_response"] == "X"
    assert scored["dialogue"][1] == original["dialogue"][1]
    assert original["dialogue"][2]["teacher_response"] == "system reply 0-2?"


def test_a_discriminating_judge_detects_every_bad_control(tmp_path, monkeypatch, capsys):
    run = _write_run(tmp_path)

    def total(reply, dialogue, idx):
        own = f"system reply {dialogue['id']}-{idx}?"
        reference = f"reference reply {dialogue['id']}-{idx}?"
        return 9 if reply in (own, reference) else 3

    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", _rubric(total))

    assert jv.run(_args(run)) == 0

    out = json.loads((run / "judge_validation_fake-judge.json").read_text())
    summary = out["summary"]
    assert summary["n_bad_controls"] == len(jv.BAD_CONTROLS)
    assert summary["n_bad_controls_detected"] == len(jv.BAD_CONTROLS)
    assert summary["controls"]["swapped"]["mean_drop_total"] == pytest.approx(6.0)
    assert summary["controls"]["reference"]["mean_drop_total"] == pytest.approx(0.0)
    assert summary["repeatability"]["share_identical_total"] == 1.0
    assert out["meta"]["n_items_scored"] == out["meta"]["n_items"]
    assert "NOT DETECTED" not in capsys.readouterr().out


def test_a_lenient_judge_is_reported_as_detecting_nothing(tmp_path, monkeypatch, capsys):
    run = _write_run(tmp_path)
    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", _rubric(lambda *_: 9))

    assert jv.run(_args(run)) == 0

    summary = json.loads((run / "judge_validation_fake-judge.json").read_text())["summary"]
    assert summary["n_bad_controls_detected"] == 0
    assert summary["controls"]["statement"]["share_tied_with_system"] == 1.0
    assert "NOT DETECTED" in capsys.readouterr().out


def test_a_failing_judge_exits_non_zero(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", _rubric(lambda *_: None))

    assert jv.run(_args(run)) == jv.FAILURE_EXIT_CODE


def test_an_interrupted_pass_resumes_without_rescoring(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    calls = []

    def total(reply, dialogue, idx):
        calls.append(reply)
        return 8

    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", _rubric(total))
    jv.run(_args(run))
    first_pass = len(calls)

    jv.run(_args(run))

    assert first_pass > 0
    assert len(calls) == first_pass


def test_dry_run_calls_no_judge(tmp_path, monkeypatch, capsys):
    run = _write_run(tmp_path)
    monkeypatch.setattr(
        jv.llm_judge_eval, "score_turn", lambda *a, **k: pytest.fail("judge was called")
    )

    assert jv.run(_args(run, dry_run=True)) == 0
    assert "Judge calls:" in capsys.readouterr().out
    assert not list(run.glob("judge_validation_*.json"))


def test_agreement_between_two_judges(tmp_path, monkeypatch):
    run = _write_run(tmp_path)

    def strict(reply, dialogue, idx):
        return 4 + idx if reply.startswith("system") else 1

    def lenient(reply, dialogue, idx):
        return 6 + idx if reply.startswith("system") else 5

    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", _rubric(strict))
    jv.run(_args(run, model="strict"))
    monkeypatch.setattr(jv.llm_judge_eval, "score_turn", _rubric(lenient))
    jv.run(_args(run, model="lenient"))
    a = json.loads((run / "judge_validation_strict.json").read_text())
    b = json.loads((run / "judge_validation_lenient.json").read_text())

    result = jv.agreement(a, b)

    assert result["n_items"] == a["meta"]["n_items"]
    assert result["spearman_total_system_replies_only"] == pytest.approx(1.0)
    assert result["mean_abs_diff_total"] > 1

    b["meta"]["sample_fingerprint"] = "different"
    with pytest.raises(ValueError):
        jv.agreement(a, b)
