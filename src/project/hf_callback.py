from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


class HFCheckpointCallback(TrainerCallback):
    """Push checkpoint-N to a HF model repo after every N-th save.

    Activated by TRAIN_HF_REPO=<repo_id>.  Cadence controlled by
    TRAIN_HF_PUSH_EVERY (default 50 steps).  Runs in a daemon thread so it
    never blocks the training loop; if the previous push is still in-flight
    when the next one fires, the new push is skipped and logged.  Also fires
    unconditionally at on_train_end to capture the final state.

    Persists the last pushed step in ``.hf_last_push`` inside the output dir
    so that resumed / crash-recovered runs do not re-push already-uploaded
    checkpoints.  On ``on_init`` the callback scans for the latest on-disk
    checkpoint and pushes it synchronously if it was never uploaded.
    """

    def __init__(self, repo_id: str, push_every: int) -> None:
        self._repo_id = repo_id
        self._push_every = push_every
        self._thread: threading.Thread | None = None
        self._output_dir: str | None = None

    # ------------------------------------------------------------------
    # Persistence — track last pushed step so we never re-push on resume
    # ------------------------------------------------------------------

    def _last_push_file(self) -> Path | None:
        if not self._output_dir:
            return None
        return Path(self._output_dir) / ".hf_last_push"

    def _read_last_pushed(self) -> int:
        p = self._last_push_file()
        if p and p.exists():
            try:
                return int(p.read_text().strip())
            except (ValueError, OSError):
                pass
        return -1

    def _write_last_pushed(self, step: int) -> None:
        p = self._last_push_file()
        if p:
            p.write_text(str(step))

    # ------------------------------------------------------------------
    # Checkpoint log — SFT_STAGE2B_CHECKPOINT_LOG.md on the HF repo
    # ------------------------------------------------------------------

    _LOG_HEADER: ClassVar[str] = """# Checkpoints in This Repo

| Checkpoint | Step | Epoch | Loss | Token Accuracy |
|---|---|---|---|---|
"""

    @staticmethod
    def _build_log_row(step: int, epoch: float, loss: float | None, acc: float | None) -> str:
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else "—"
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else "—"
        return f"| checkpoint-{step}/ | {step} | {epoch:.3f} | {loss_str} | {acc_str} |"

    def _update_checkpoint_log(
        self,
        step: int,
        epoch: float,
        loss: float | None,
        acc: float | None,
    ) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        rows: dict[int, str] = {}
        try:
            path = api.hf_hub_download(
                repo_id=self._repo_id,
                filename="SFT_STAGE2B_CHECKPOINT_LOG.md",
                repo_type="model",
            )
            with open(path) as f:
                for line in f:
                    if line.startswith("| checkpoint-"):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 3:
                            try:
                                # part[1] = "checkpoint-1230/"
                                ckpt_name = parts[1].rstrip("/")
                                s = int(ckpt_name.split("-")[-1])
                                rows[s] = line.rstrip()
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass

        new_row = self._build_log_row(step, epoch, loss, acc)
        rows[step] = new_row

        lines = [self._LOG_HEADER.rstrip("\n")]
        for s in sorted(rows):
            lines.append(rows[s])
        lines.append("")
        content = "\n".join(lines)

        try:
            api.upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo="SFT_STAGE2B_CHECKPOINT_LOG.md",
                repo_id=self._repo_id,
                repo_type="model",
                commit_message=f"checkpoint log: step {step}",
            )
        except Exception as exc:
            print(f"  [HF] checkpoint log update failed (step {step}): {exc}", flush=True)

    # ------------------------------------------------------------------
    # Push logic
    # ------------------------------------------------------------------

    def _push(
        self,
        ckpt_dir: Path,
        step: int,
        commit_msg: str,
        epoch: float = 0.0,
        loss: float | None = None,
        acc: float | None = None,
    ) -> None:
        from huggingface_hub import HfApi

        try:
            HfApi().upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoint-{step}",
                repo_id=self._repo_id,
                repo_type="model",
                commit_message=commit_msg,
            )
            print(f"  [HF] pushed checkpoint-{step} → {self._repo_id}", flush=True)
            self._write_last_pushed(step)
            self._update_checkpoint_log(step, epoch, loss, acc)
        except Exception as exc:
            print(f"  [HF] push failed (step {step}): {exc}", flush=True)

    def _maybe_push(
        self,
        args: TrainingArguments,
        state: TrainerState,
        force: bool = False,
    ) -> None:
        step = state.global_step
        if not force and step % self._push_every != 0:
            return
        if not force and step <= self._read_last_pushed():
            return
        output_dir = args.output_dir
        if not output_dir:
            return
        ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            return
        if self._thread and self._thread.is_alive():
            print(
                f"  [HF] previous push still running, skipping step {step}",
                flush=True,
            )
            return
        log = state.log_history[-1] if state.log_history else {}
        loss = log.get("loss", log.get("train_loss"))
        acc = log.get("mean_token_accuracy")
        epoch = state.epoch or 0.0
        if isinstance(loss, float) and isinstance(acc, float):
            commit_msg = (
                f"checkpoint-{step} (step {step}, epoch {epoch:.3f}, "
                f"loss {loss:.4f}, acc {acc:.4f})"
            )
        else:
            commit_msg = f"checkpoint-{step} (step {step}, epoch {epoch:.3f})"
        self._thread = threading.Thread(
            target=self._push, args=(ckpt_dir, step, commit_msg, epoch, loss, acc), daemon=True
        )
        self._thread.start()
        print(f"  [HF] pushing checkpoint-{step} in background...", flush=True)

    # ------------------------------------------------------------------
    # Launch push — push latest checkpoint on init if not already pushed
    # ------------------------------------------------------------------

    def _push_latest_on_launch(
        self,
        args: TrainingArguments,
    ) -> None:
        output_dir = Path(args.output_dir) if args.output_dir else None
        if not output_dir or not output_dir.exists():
            return
        ckpt_dirs = list(output_dir.glob("checkpoint-*"))
        if not ckpt_dirs:
            return
        latest = max(ckpt_dirs, key=lambda d: int(d.name.split("-")[-1]))
        latest_step = int(latest.name.split("-")[-1])
        if latest_step <= self._read_last_pushed():
            return
        if latest_step % self._push_every != 0:
            return
        commit_msg = f"checkpoint-{latest_step} (step {latest_step}, resume push)"
        print(f"  [HF] launch push: checkpoint-{latest_step}", flush=True)
        self._push(latest, latest_step, commit_msg, epoch=0.0, loss=None, acc=None)

    # ------------------------------------------------------------------
    # Trainer callback hooks
    # ------------------------------------------------------------------

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._output_dir = args.output_dir
        self._push_latest_on_launch(args)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._maybe_push(args, state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._maybe_push(args, state, force=True)
        if self._thread and self._thread.is_alive():
            print("  [HF] waiting for final push to complete...", flush=True)
            self._thread.join()
