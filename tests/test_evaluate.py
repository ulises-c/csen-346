import json

import pytest

pytest.importorskip("rouge_score")
pytest.importorskip("sacrebleu")

from src.project import evaluate


def test_evaluate_run_returns_empty_dict_when_dialogues_missing(tmp_path):
    result = evaluate.evaluate_run(tmp_path / "missing")

    assert result == {}


def test_evaluate_run_saves_metrics_summary(tmp_path, monkeypatch):
    results_dir = tmp_path / "baseline"
    dialogues_dir = results_dir / "dialogues"
    dialogues_dir.mkdir(parents=True)
    (dialogues_dir / "0001.json").write_text("{}", encoding="utf-8")
    metrics = {"rouge1": 10.0}

    monkeypatch.setattr(evaluate, "compute_all_metrics", lambda path: metrics)
    monkeypatch.setattr(evaluate, "format_metrics_table", lambda data: "formatted")

    result = evaluate.evaluate_run(results_dir)

    assert result == metrics
    assert json.loads((results_dir / "metrics_summary.json").read_text()) == metrics


def test_compare_runs_uses_existing_metrics_files_when_present(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    gemma = tmp_path / "gemma"
    baseline.mkdir()
    gemma.mkdir()
    (baseline / "metrics_summary.json").write_text(
        json.dumps({"rouge1": 1, "rouge2": 2, "rougeL": 3, "bleu4": 4, "state_accuracy": {"overall": 5}}),
        encoding="utf-8",
    )
    (gemma / "metrics_summary.json").write_text(
        json.dumps({"rouge1": 6, "rouge2": 7, "rougeL": 8, "bleu4": 9, "state_accuracy": {"overall": 10}}),
        encoding="utf-8",
    )

    calls = []
    monkeypatch.setattr(evaluate, "evaluate_run", lambda path: calls.append(path) or {})

    evaluate.compare_runs([baseline, gemma])

    assert calls == []
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["baseline"]["rouge1"] == 1
    assert comparison["gemma"]["bleu4"] == 9


def test_compare_runs_computes_missing_metrics_when_needed(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    gemma = tmp_path / "gemma"
    baseline.mkdir()
    gemma.mkdir()
    (baseline / "metrics_summary.json").write_text(
        json.dumps({"rouge1": 1, "rouge2": 2, "rougeL": 3, "bleu4": 4, "state_accuracy": {"overall": 5}}),
        encoding="utf-8",
    )

    def fake_evaluate_run(path):
        return {"rouge1": 6, "rouge2": 7, "rougeL": 8, "bleu4": 9, "state_accuracy": {"overall": 10}}

    monkeypatch.setattr(evaluate, "evaluate_run", fake_evaluate_run)

    evaluate.compare_runs([baseline, gemma])

    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["gemma"]["rouge2"] == 7
