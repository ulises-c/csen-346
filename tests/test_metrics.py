import json

import pytest

pytest.importorskip("rouge_score")
pytest.importorskip("sacrebleu")

from src.project.metrics import (
    compute_all_metrics,
    compute_bleu,
    compute_rouge,
    compute_state_accuracy,
    extract_predictions_and_references,
    format_metrics_table,
)


def write_dialogue(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_compute_rouge_identical_chinese_strings():
    scores = compute_rouge(["你好世界"], ["你好世界"])

    assert scores["rouge1"] == 100.0
    assert scores["rouge2"] == 100.0
    assert scores["rougeL"] == 100.0


def test_compute_bleu_identical_chinese_strings():
    score = compute_bleu(["今天天气很好"], ["今天天气很好"])

    assert score > 99.0


def test_extract_predictions_and_references_skips_error_files(tmp_path):
    dialogues_dir = tmp_path / "dialogues"
    dialogues_dir.mkdir()
    write_dialogue(
        dialogues_dir / "0001.json",
        {
            "dialogue": [
                {
                    "teacher_response": "提示一",
                    "ground_truth_teacher": "提示一",
                }
            ]
        },
    )
    write_dialogue(dialogues_dir / "0002.json", {"id": 2, "error": "boom"})

    predictions, references = extract_predictions_and_references(dialogues_dir)

    assert predictions == ["提示一"]
    assert references == ["提示一"]


def test_extract_predictions_and_references_skips_empty_turns(tmp_path):
    dialogues_dir = tmp_path / "dialogues"
    dialogues_dir.mkdir()
    write_dialogue(
        dialogues_dir / "0001.json",
        {
            "dialogue": [
                {"teacher_response": "", "ground_truth_teacher": "参考"},
                {"teacher_response": "生成", "ground_truth_teacher": ""},
                {"teacher_response": "有效生成", "ground_truth_teacher": "有效参考"},
            ]
        },
    )

    predictions, references = extract_predictions_and_references(dialogues_dir)

    assert predictions == ["有效生成"]
    assert references == ["有效参考"]


def test_compute_state_accuracy_overall_and_per_stage(tmp_path):
    dialogues_dir = tmp_path / "dialogues"
    dialogues_dir.mkdir()
    write_dialogue(
        dialogues_dir / "0001.json",
        {
            "dialogue": [
                {"state": "a1", "ground_truth_state": "a1"},
                {"state": "b1", "ground_truth_state": "b2"},
                {"state": "b3", "ground_truth_state": "b3"},
                {"state": "c2", "ground_truth_state": "c9"},
            ]
        },
    )

    metrics = compute_state_accuracy(dialogues_dir)

    assert metrics["overall"] == 50.0
    assert metrics["per_stage"]["a"] == 100.0
    assert metrics["per_stage"]["b"] == 50.0
    assert metrics["per_stage"]["c"] == 0.0
    assert metrics["total_turns"] == 4


def test_compute_all_metrics_returns_error_when_no_valid_dialogues(tmp_path):
    dialogues_dir = tmp_path / "dialogues"
    dialogues_dir.mkdir()
    write_dialogue(dialogues_dir / "0001.json", {"id": 1, "error": "failed"})

    metrics = compute_all_metrics(dialogues_dir)

    assert metrics == {"error": "No dialogue data found"}


def test_format_metrics_table_contains_expected_sections():
    table = format_metrics_table(
        {
            "n_turns": 10,
            "rouge1": 44.61,
            "rouge2": 26.04,
            "rougeL": 38.02,
            "bleu4": 19.60,
            "state_accuracy": {
                "overall": 25.94,
                "per_stage": {"a": 95.15, "b": 36.93},
            },
        }
    )

    assert "EVALUATION METRICS" in table
    assert "Total turns evaluated: 10" in table
    assert "ROUGE-1:  44.61" in table
    assert "BLEU-4:   19.6" in table
    assert "Stage a: 95.15%" in table
