from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.project.hf_callback import HFCheckpointCallback


def _state(step: int, epoch: float = 0.3, loss: float = 0.65, acc: float = 0.80):
    log = {"loss": loss, "mean_token_accuracy": acc}
    return SimpleNamespace(global_step=step, epoch=epoch, log_history=[log])


def _args(tmp_path, step: int):
    (tmp_path / f"checkpoint-{step}").mkdir()
    return SimpleNamespace(output_dir=str(tmp_path))


_ctrl = SimpleNamespace()


def test_skips_non_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = _args(tmp_path, 30)
    cb.on_save(args, _state(30), _ctrl)
    assert cb._thread is None


def test_pushes_at_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = _args(tmp_path, 50)
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(args, _state(50), _ctrl)
        cb._thread.join()
        mock_push.assert_called_once()
        ckpt_dir, step, commit_msg, epoch, loss, acc = mock_push.call_args[0]
        assert step == 50
        assert "loss 0.6500" in commit_msg
        assert "acc 0.8000" in commit_msg


def test_skips_when_thread_alive(tmp_path, capsys):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    alive_thread = MagicMock()
    alive_thread.is_alive.return_value = True
    cb._thread = alive_thread
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(_args(tmp_path, 50), _state(50), _ctrl)
        mock_push.assert_not_called()
    assert "skipping" in capsys.readouterr().out


def test_skips_when_checkpoint_dir_missing(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = SimpleNamespace(output_dir=str(tmp_path))
    cb.on_save(args, _state(50), _ctrl)
    assert cb._thread is None


def test_commit_msg_without_log(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    empty_state = SimpleNamespace(global_step=100, epoch=0.5, log_history=[])
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(_args(tmp_path, 100), empty_state, _ctrl)
        cb._thread.join()
        _, step, commit_msg, *_ = mock_push.call_args[0]
        assert "step 100" in commit_msg
        assert "loss" not in commit_msg


def test_train_end_forces_push_at_non_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    with patch.object(cb, "_push") as mock_push:
        cb.on_train_end(_args(tmp_path, 37), _state(37), _ctrl)
        cb._thread.join()
        mock_push.assert_called_once()
        _, step, *_ = mock_push.call_args[0]
        assert step == 37


def test_train_end_joins_alive_thread(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    cb._thread = mock_thread
    # no checkpoint dir → _maybe_push returns before spawning a new thread
    # but on_train_end still joins the existing alive thread
    args = SimpleNamespace(output_dir=str(tmp_path))
    cb.on_train_end(args, _state(37), _ctrl)
    mock_thread.join.assert_called_once()


# ---------------------------------------------------------------------------
# Persistence — .hf_last_push file
# ---------------------------------------------------------------------------


def test_writes_and_reads_last_pushed(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    cb._write_last_pushed(100)
    assert cb._read_last_pushed() == 100
    cb._write_last_pushed(200)
    assert cb._read_last_pushed() == 200


def test_last_pushed_defaults_to_neg_one(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    assert cb._read_last_pushed() == -1


def test_writes_last_pushed_on_successful_push(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    with patch("huggingface_hub.HfApi.upload_folder"):
        cb._push(tmp_path, 50, "test")
    assert (tmp_path / ".hf_last_push").read_text() == "50"


# ---------------------------------------------------------------------------
# Step-skip guard — skip if already pushed
# ---------------------------------------------------------------------------


def test_skips_already_pushed_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    (tmp_path / ".hf_last_push").write_text("50")
    args = _args(tmp_path, 50)
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(args, _state(50), _ctrl)
        mock_push.assert_not_called()


def test_force_push_bypasses_last_pushed(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    (tmp_path / ".hf_last_push").write_text("50")
    with patch.object(cb, "_push") as mock_push:
        cb.on_train_end(_args(tmp_path, 50), _state(50), _ctrl)
        mock_push.assert_called_once()


# ---------------------------------------------------------------------------
# Launch check — on_init_end
# ---------------------------------------------------------------------------


def test_on_init_pushes_latest_checkpoint(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-1000").mkdir()
    (tmp_path / "checkpoint-1200").mkdir()
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init_end(args, SimpleNamespace(), _ctrl)
        mock_push.assert_called_once()
        _, step, commit_msg, *_ = mock_push.call_args[0]
        assert step == 1200
        assert "resume push" in commit_msg


def test_on_init_skips_if_already_pushed(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-1200").mkdir()
    (tmp_path / ".hf_last_push").write_text("1200")
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init_end(args, SimpleNamespace(), _ctrl)
        mock_push.assert_not_called()


def test_on_init_skips_non_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-1670").mkdir()  # 1670 % 50 != 0
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init_end(args, SimpleNamespace(), _ctrl)
        mock_push.assert_not_called()


def test_on_init_skips_if_no_checkpoints(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init_end(args, SimpleNamespace(), _ctrl)
        mock_push.assert_not_called()


def test_on_init_picks_highest_step_numerically(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-90").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init_end(args, SimpleNamespace(), _ctrl)
        mock_push.assert_called_once()
        _, step, *_ = mock_push.call_args[0]
        assert step == 100


# ---------------------------------------------------------------------------
# Checkpoint log — SFT_STAGE2B_CHECKPOINT_LOG.md
# ---------------------------------------------------------------------------


def test_build_log_row_with_metrics():
    row = HFCheckpointCallback._build_log_row(1230, 0.255, 0.6465, 0.798)
    assert row == "| checkpoint-1230/ | 1230 | 0.255 | 0.6465 | 0.7980 |"


def test_build_log_row_without_metrics():
    row = HFCheckpointCallback._build_log_row(1220, 0.253, None, None)
    assert row == "| checkpoint-1220/ | 1220 | 0.253 | — | — |"


def test_update_checkpoint_log_starts_fresh_when_no_existing(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    with patch("huggingface_hub.HfApi.hf_hub_download", side_effect=Exception("not found")):
        with patch("huggingface_hub.HfApi.upload_file") as mock_upload:
            cb._update_checkpoint_log(50, 0.3, 0.65, 0.80)
            mock_upload.assert_called_once()
            content = mock_upload.call_args[1]["path_or_fileobj"].decode()
            assert "checkpoint-50/" in content
            assert "0.6500" in content


def test_update_checkpoint_log_appends_new_row(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    existing_file = tmp_path / "existing_log.md"
    existing_file.write_text("| checkpoint-50/ | 50 | 0.300 | 0.6500 | 0.8000 |\n")
    with patch("huggingface_hub.HfApi.hf_hub_download", return_value=str(existing_file)):
        with patch("huggingface_hub.HfApi.upload_file") as mock_upload:
            cb._update_checkpoint_log(100, 0.5, 1.23, 0.90)
            mock_upload.assert_called_once()
            content = mock_upload.call_args[1]["path_or_fileobj"].decode()
            assert "checkpoint-50/" in content
            assert "checkpoint-100/" in content
            assert "1.2300" in content


def test_update_checkpoint_log_replaces_existing_row(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    existing_file = tmp_path / "existing_log.md"
    existing_file.write_text("| checkpoint-50/ | 50 | 0.300 | 0.6500 | 0.8000 |\n")
    with patch("huggingface_hub.HfApi.hf_hub_download", return_value=str(existing_file)):
        with patch("huggingface_hub.HfApi.upload_file") as mock_upload:
            cb._update_checkpoint_log(50, 0.4, 0.70, 0.82)
            mock_upload.assert_called_once()
            content = mock_upload.call_args[1]["path_or_fileobj"].decode()
            assert "checkpoint-50/" in content
            assert "0.400" in content
            assert content.count("checkpoint-50/") == 1
