import json
import sys
from types import SimpleNamespace

sys.modules.setdefault("openai", SimpleNamespace(Client=object))
from src.project import kele


def fake_config():
    return SimpleNamespace(
        teacher=SimpleNamespace(model_name="teacher-model", base_url="http://teacher"),
        consultant=SimpleNamespace(model_name="consultant-model", base_url="http://consultant"),
        max_teaching_rounds=8,
    )


def install_fake_metrics(monkeypatch, metrics):
    fake_module = SimpleNamespace(
        compute_all_metrics=lambda path: metrics,
        format_metrics_table=lambda data: "ok",
    )
    monkeypatch.setitem(sys.modules, "src.project.metrics", fake_module)


def test_run_batch_evaluation_creates_expected_output_files(tmp_path, monkeypatch):
    dataset = [
        {"id": 1, "question": "Q1", "answer": "A1", "dialogue": []},
        {"id": 2, "question": "Q2", "answer": "A2", "dialogue": []},
    ]
    metrics = {"rouge1": 1.0, "state_accuracy": {"overall": 50.0, "per_stage": {}, "total_turns": 2}}
    system = SimpleNamespace(teacher_model_name="teacher-model", consultant_model_name="consultant-model")

    monkeypatch.setattr(kele, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(kele, "create_system", lambda *args, **kwargs: system)
    monkeypatch.setattr(
        kele,
        "run_single_dialogue",
        lambda _system, item: {"id": item["id"], "dialogue": [{"state": "a1", "ground_truth_state": "a1"}]},
    )
    monkeypatch.setattr(kele, "load_config", lambda *args, **kwargs: fake_config())
    install_fake_metrics(monkeypatch, metrics)

    output_dir = tmp_path / "results"
    kele.run_batch_evaluation(output_dir)

    assert (output_dir / "dialogues" / "0001.json").exists()
    assert (output_dir / "dialogues" / "0002.json").exists()
    assert (output_dir / "progress.log").exists()
    assert json.loads((output_dir / "run_config.json").read_text())["completed"] == 2
    assert json.loads((output_dir / "metrics_summary.json").read_text()) == metrics


def test_run_batch_evaluation_skips_existing_dialogue_files(tmp_path, monkeypatch):
    dataset = [
        {"id": 1, "question": "Q1", "answer": "A1", "dialogue": []},
        {"id": 2, "question": "Q2", "answer": "A2", "dialogue": []},
    ]
    output_dir = tmp_path / "results"
    dialogues_dir = output_dir / "dialogues"
    dialogues_dir.mkdir(parents=True)
    existing = {"id": 1, "dialogue": [{"state": "a1"}]}
    (dialogues_dir / "0001.json").write_text(json.dumps(existing), encoding="utf-8")

    seen_ids = []
    system = SimpleNamespace(teacher_model_name="teacher-model", consultant_model_name="consultant-model")

    monkeypatch.setattr(kele, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(kele, "create_system", lambda *args, **kwargs: system)
    monkeypatch.setattr(
        kele,
        "run_single_dialogue",
        lambda _system, item: seen_ids.append(item["id"]) or {"id": item["id"], "dialogue": []},
    )
    monkeypatch.setattr(kele, "load_config", lambda *args, **kwargs: fake_config())
    install_fake_metrics(monkeypatch, {"rouge1": 0})

    kele.run_batch_evaluation(output_dir)

    assert seen_ids == [2]
    assert json.loads((dialogues_dir / "0001.json").read_text()) == existing


def test_run_batch_evaluation_writes_error_json_for_failed_dialogue(tmp_path, monkeypatch):
    dataset = [{"id": 1, "question": "Q1", "answer": "A1", "dialogue": []}]
    system = SimpleNamespace(teacher_model_name="teacher-model", consultant_model_name="consultant-model")

    monkeypatch.setattr(kele, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(kele, "create_system", lambda *args, **kwargs: system)
    monkeypatch.setattr(
        kele,
        "run_single_dialogue",
        lambda _system, item: (_ for _ in ()).throw(RuntimeError("dialogue failed")),
    )
    monkeypatch.setattr(kele, "load_config", lambda *args, **kwargs: fake_config())
    install_fake_metrics(monkeypatch, {"rouge1": 0})

    output_dir = tmp_path / "results"
    kele.run_batch_evaluation(output_dir)

    saved = json.loads((output_dir / "dialogues" / "0001.json").read_text())
    assert saved["error"] == "dialogue failed"
