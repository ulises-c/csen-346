"""Unit tests for the LLM-judge rubric parser (scripts/llm_judge_eval.py).

Only _parse_rubric is exercised — it's the fragile seam that turns free-form
judge output into validated axis scores. The backends themselves (openai client /
`claude -p` subprocess) are integration-only.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "llm_judge_eval", Path(__file__).resolve().parents[1] / "scripts" / "llm_judge_eval.py"
)
assert _spec and _spec.loader
llm_judge_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_judge_eval)
_parse_rubric = llm_judge_eval._parse_rubric


def test_parse_rubric_clean_json():
    scores = _parse_rubric(
        '{"socratic_validity": 3, "advancement": 2, "age_appropriateness": 2, '
        '"question_form": 1, "comment": "ok"}'
    )
    assert scores is not None
    assert scores["total"] == 8
    assert scores["socratic_validity"] == 3


def test_parse_rubric_strips_markdown_fence():
    scores = _parse_rubric(
        '```json\n{"socratic_validity": 1, "advancement": 1, '
        '"age_appropriateness": 0, "question_form": 0}\n```'
    )
    assert scores is not None
    assert scores["total"] == 2


def test_parse_rubric_missing_axis_returns_none():
    assert _parse_rubric('{"socratic_validity": 3, "advancement": 2}') is None


def test_parse_rubric_non_json_returns_none():
    assert _parse_rubric("I cannot score this.") is None


def test_parse_rubric_coerces_stringified_ints():
    scores = _parse_rubric(
        '{"socratic_validity": "3", "advancement": "0", '
        '"age_appropriateness": "1", "question_form": "2"}'
    )
    assert scores is not None
    assert scores["total"] == 6


import json as _json


def _make_dialogues(tmp_path):
    ddir = tmp_path / "dialogues"
    ddir.mkdir()
    # 10 dialogues, each with one turn per stage a-e (state codes a1,b4,c16,d33,e34).
    states = ["a1", "b4", "c16", "d33", "e34"]
    for n in range(10):
        turns = [
            {"student": "s", "teacher_response": "t", "ground_truth_state": st} for st in states
        ]
        (ddir / f"{n:04d}.json").write_text(_json.dumps({"id": n, "dialogue": turns}))
    return ddir


def test_stratified_sample_counts_and_stage_filtering(tmp_path):
    ddir = _make_dialogues(tmp_path)
    sample = llm_judge_eval.build_stratified_sample(ddir, per_stage=3, stages="bcde", seed=42)
    counts = {}
    for _f, _i, s in sample:
        counts[s] = counts.get(s, 0) + 1
    assert counts == {"b": 3, "c": 3, "d": 3, "e": 3}  # 3 each, no 'a'
    assert len(sample) == 12


def test_stratified_sample_deterministic_under_seed(tmp_path):
    ddir = _make_dialogues(tmp_path)
    s1 = llm_judge_eval.build_stratified_sample(ddir, per_stage=3, stages="bcde", seed=42)
    s2 = llm_judge_eval.build_stratified_sample(ddir, per_stage=3, stages="bcde", seed=42)
    s3 = llm_judge_eval.build_stratified_sample(ddir, per_stage=3, stages="bcde", seed=7)
    assert s1 == s2
    assert s1 != s3


def test_stratified_sample_takes_all_when_pool_small(tmp_path):
    ddir = _make_dialogues(tmp_path)  # only 10 turns per stage available
    sample = llm_judge_eval.build_stratified_sample(ddir, per_stage=50, stages="c", seed=42)
    assert len(sample) == 10  # capped at pool size, no error


# ── completeness of a judge pass ──────────────────────────────────────────────


def _write_run(tmp_path, n_dialogues=3, n_turns=2):
    dialogues = tmp_path / "dialogues"
    dialogues.mkdir()
    for d in range(n_dialogues):
        turns = [
            {"teacher_response": "q?", "ground_truth_teacher": "q?", "ground_truth_state": "b4"}
            for _ in range(n_turns)
        ]
        (dialogues / f"{d:04d}.json").write_text(_json.dumps({"id": d, "dialogue": turns}))
    return tmp_path


def _fake_score(fail_dialogue_ids=()):
    def score_turn(client, model, dialogue, turn_idx, backend="api"):
        if dialogue["id"] in fail_dialogue_ids:
            return None
        return {
            "socratic_validity": 2,
            "advancement": 2,
            "age_appropriateness": 2,
            "question_form": 2,
            "total": 8,
        }

    return score_turn


def test_judge_summary_marks_complete_pass(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    monkeypatch.setattr(llm_judge_eval, "score_turn", _fake_score())

    summary = llm_judge_eval.judge_results_dir(run, None, "m", workers=2)

    assert summary["complete"] is True
    assert (summary["n_dialogues_judged"], summary["n_dialogues_total"]) == (3, 3)
    assert (summary["n_turns_judged"], summary["n_turns_total"]) == (6, 6)


def test_judge_summary_flags_partial_pass(tmp_path, monkeypatch, capsys):
    run = _write_run(tmp_path)
    monkeypatch.setattr(llm_judge_eval, "score_turn", _fake_score(fail_dialogue_ids={1}))

    summary = llm_judge_eval.judge_results_dir(run, None, "m", workers=2)

    assert summary["complete"] is False
    assert (summary["n_dialogues_judged"], summary["n_dialogues_total"]) == (2, 3)
    assert (summary["n_turns_judged"], summary["n_turns_total"]) == (4, 6)
    assert "incomplete judge pass" in capsys.readouterr().err
    on_disk = _json.loads((run / "judge_summary.json").read_text())
    assert on_disk["complete"] is False


def test_sequential_smoke_run_is_reported_incomplete(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    monkeypatch.setattr(llm_judge_eval, "score_turn", _fake_score())

    summary = llm_judge_eval.judge_results_dir(
        run, None, "m", workers=1, backend="claude-code", max_turns=3
    )

    assert summary["n_turns_judged"] == 3
    assert summary["n_turns_total"] == 6
    assert summary["complete"] is False


def test_judge_pass_resumes_from_saved_scores(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    monkeypatch.setattr(llm_judge_eval, "score_turn", _fake_score(fail_dialogue_ids={1}))
    first = llm_judge_eval.judge_results_dir(run, None, "m", workers=2)
    assert first["complete"] is False

    calls = []

    def counting_score(client, model, dialogue, turn_idx, backend="api"):
        calls.append(dialogue["id"])
        return _fake_score()(client, model, dialogue, turn_idx)

    monkeypatch.setattr(llm_judge_eval, "score_turn", counting_score)
    second = llm_judge_eval.judge_results_dir(run, None, "m", workers=2)

    assert second["complete"] is True
    assert set(calls) == {1}  # only the dialogue that was not fully scored is judged again
    assert (second["n_dialogues_judged"], second["n_turns_judged"]) == (3, 6)


def test_saved_scores_from_another_judge_are_not_reused(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    monkeypatch.setattr(llm_judge_eval, "score_turn", _fake_score())
    llm_judge_eval.judge_results_dir(run, None, "judge-a", workers=2)

    calls = []

    def counting_score(client, model, dialogue, turn_idx, backend="api"):
        calls.append(dialogue["id"])
        return _fake_score()(client, model, dialogue, turn_idx)

    monkeypatch.setattr(llm_judge_eval, "score_turn", counting_score)
    llm_judge_eval.judge_results_dir(run, None, "judge-b", workers=2)

    assert set(calls) == {0, 1, 2}
