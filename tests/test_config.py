from pathlib import Path

import pytest

from src.project import config


def clear_config_env(monkeypatch):
    keys = [
        "CONSULTANT_API_KEY",
        "CONSULTANT_BASE_URL",
        "CONSULTANT_MODEL_NAME",
        "TEACHER_API_KEY",
        "TEACHER_BASE_URL",
        "TEACHER_MODEL_NAME",
        "DEBUG_MODE",
        "MAX_TEACHING_ROUNDS",
        "TEACHER_LOCAL_PATH",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_load_env_file_ignores_comments_and_blank_lines(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# comment\nCONSULTANT_API_KEY=test-key\n\nTEACHER_BASE_URL=http://localhost\n",
        encoding="utf-8",
    )

    config.load_env_file(env_file)

    assert config.os.environ["CONSULTANT_API_KEY"] == "test-key"
    assert config.os.environ["TEACHER_BASE_URL"] == "http://localhost"


def test_load_env_file_strips_quotes(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CONSULTANT_MODEL_NAME=\"gpt-4o\"\nTEACHER_API_KEY='secret'\n",
        encoding="utf-8",
    )

    config.load_env_file(env_file)

    assert config.os.environ["CONSULTANT_MODEL_NAME"] == "gpt-4o"
    assert config.os.environ["TEACHER_API_KEY"] == "secret"


def test_load_config_reads_required_variables(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "CONSULTANT_API_KEY=consultant-key",
                "CONSULTANT_BASE_URL=https://consultant.example",
                "CONSULTANT_MODEL_NAME=gpt-4o",
                "TEACHER_BASE_URL=http://localhost:8001/v1",
                "TEACHER_MODEL_NAME=SocratTeachLLM",
            ]
        ),
        encoding="utf-8",
    )

    cfg = config.load_config(root_dir=tmp_path)

    assert cfg.consultant.api_key == "consultant-key"
    assert cfg.consultant.base_url == "https://consultant.example"
    assert cfg.teacher.base_url == "http://localhost:8001/v1"
    assert cfg.teacher.api_key == "not-needed"


def test_load_config_experiment_file_takes_precedence(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    (tmp_path / "configs").mkdir()
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "CONSULTANT_API_KEY=shared-secret",
                "CONSULTANT_BASE_URL=https://default.example",
                "CONSULTANT_MODEL_NAME=default-model",
                "TEACHER_BASE_URL=http://default-teacher",
                "TEACHER_MODEL_NAME=teacher-default",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "baseline.env").write_text(
        "\n".join(
            [
                "CONSULTANT_BASE_URL=https://experiment.example",
                "CONSULTANT_MODEL_NAME=experiment-model",
                "TEACHER_BASE_URL=http://experiment-teacher",
                "TEACHER_MODEL_NAME=teacher-experiment",
            ]
        ),
        encoding="utf-8",
    )

    cfg = config.load_config(experiment="baseline", root_dir=tmp_path)

    assert cfg.consultant.api_key == "shared-secret"
    assert cfg.consultant.base_url == "https://experiment.example"
    assert cfg.consultant.model_name == "experiment-model"
    assert cfg.teacher.base_url == "http://experiment-teacher"


def test_load_config_raises_when_required_variable_missing(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    (tmp_path / ".env").write_text("CONSULTANT_API_KEY=only-one\n", encoding="utf-8")

    with pytest.raises(EnvironmentError, match="CONSULTANT_BASE_URL"):
        config.load_config(root_dir=tmp_path)


def test_load_config_parses_debug_mode_and_max_rounds(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "CONSULTANT_API_KEY=consultant-key",
                "CONSULTANT_BASE_URL=https://consultant.example",
                "CONSULTANT_MODEL_NAME=gpt-4o",
                "TEACHER_BASE_URL=http://localhost:8001/v1",
                "TEACHER_MODEL_NAME=SocratTeachLLM",
                "DEBUG_MODE=false",
                "MAX_TEACHING_ROUNDS=11",
            ]
        ),
        encoding="utf-8",
    )

    cfg = config.load_config(root_dir=tmp_path)

    assert cfg.debug_mode is False
    assert cfg.max_teaching_rounds == 11


def test_load_config_uses_default_teacher_local_path(tmp_path, monkeypatch):
    clear_config_env(monkeypatch)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "CONSULTANT_API_KEY=consultant-key",
                "CONSULTANT_BASE_URL=https://consultant.example",
                "CONSULTANT_MODEL_NAME=gpt-4o",
                "TEACHER_BASE_URL=http://localhost:8001/v1",
                "TEACHER_MODEL_NAME=SocratTeachLLM",
            ]
        ),
        encoding="utf-8",
    )

    cfg = config.load_config(root_dir=tmp_path)

    assert cfg.teacher_local_path.endswith(str(Path("hf_models") / "SocratTeachLLM"))
