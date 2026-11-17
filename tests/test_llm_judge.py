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
