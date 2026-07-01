"""W&B (wandb.ai) tracking, detached from the eval/train entrypoints.

Three small classes, split by side of the pipeline:

- ``WandbAuth``  — shared availability + auth check, reused by both trackers.
- ``SftTracker`` — SFT side (scripts/train_sft.py). HF's SFTTrainer owns
  ``wandb.init`` via ``report_to``; this only gates it and resolves
  project/run-name.
- ``EvalTracker`` — eval side (src/project/kele.py). Logs eval metrics as a
  W&B run so base↔SFT evals compare in one dashboard.

Both trackers degrade to no-ops when wandb is unauthenticated, so neither the
training loop nor the eval pipeline depends on W&B being configured.
"""

from __future__ import annotations

import os
import sys


class WandbAuth:
    """Shared wandb availability + authentication."""

    @staticmethod
    def authed() -> bool:
        """True if wandb is importable and has a usable API key.

        ``relogin=False`` avoids interactive prompts in nohup runs; the
        ``login`` callable check guards against the local ``wandb/``
        run-artifacts directory shadowing the installed package.
        """
        try:
            import wandb  # pyright: ignore[reportMissingImports]

            if not callable(getattr(wandb, "login", None)):
                raise ImportError("wandb package not importable (shadowed by local wandb/ dir?)")
            return bool(wandb.login(relogin=False))
        except Exception:
            return False


class SftTracker:
    """SFT-side tracking. The SFTTrainer performs ``wandb.init`` itself via
    ``report_to=["wandb"]``; this class only decides whether to enable it
    (``.enabled``) and sets the default project as a side effect."""

    DEFAULT_PROJECT = "csen346-sft"

    def __init__(self) -> None:
        self.enabled = WandbAuth.authed()
        if self.enabled:
            os.environ.setdefault("WANDB_PROJECT", self.DEFAULT_PROJECT)
        else:
            print(
                "WARNING: W&B not authenticated — tracking disabled. "
                "Run `wandb login` or set WANDB_API_KEY to enable.",
                file=sys.stderr,
            )


class EvalTracker:
    """Eval-side tracking. Logs rouge/bleu + state-accuracy as a W&B run when
    ``WANDB_EVAL`` is set and wandb is authed. Off by default so smoke/sanity
    runs don't spawn runs.

    The run stays open across the eval so metrics can be logged incrementally
    (``log_step`` per N completed dialogues, step = completed count). A
    crash-resumed eval starts a fresh W&B run whose steps continue from where
    the previous one stopped; same-named runs overlay into one curve in the UI.
    """

    DEFAULT_PROJECT = "csen346-eval"

    def __init__(self) -> None:
        self._requested = bool(os.getenv("WANDB_EVAL"))
        self.enabled = self._requested and WandbAuth.authed()
        self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    @staticmethod
    def _flatten(metrics: dict) -> dict:
        state_acc = metrics.get("state_accuracy", {})
        flat = {
            "eval/n_turns": metrics.get("n_turns"),
            "eval/rouge1": metrics.get("rouge1"),
            "eval/rouge2": metrics.get("rouge2"),
            "eval/rougeL": metrics.get("rougeL"),
            "eval/bleu4": metrics.get("bleu4"),
            "eval/state_accuracy_overall": state_acc.get("overall"),
        }
        for stage, acc in state_acc.get("per_stage", {}).items():
            flat[f"eval/state_acc/{stage}"] = acc
        return flat

    def start(self, run_name: str) -> None:
        if not self.enabled:
            if self._requested:
                print("WANDB_EVAL set but wandb is not authed — skipping. Run `wandb login`.")
            return
        import wandb  # pyright: ignore[reportMissingImports]

        self._run = wandb.init(
            project=os.getenv("WANDB_PROJECT", self.DEFAULT_PROJECT),
            name=run_name,
            job_type="eval",
            config={"experiment": run_name},
            reinit=True,
        )

    def log_step(self, metrics: dict, step: int) -> None:
        if self._run is not None:
            self._run.log(self._flatten(metrics), step=step)

    def finish(self, metrics: dict | None = None, step: int | None = None) -> None:
        if self._run is None:
            return
        if metrics is not None:
            self._run.log(self._flatten(metrics), step=step)
        self._run.finish()
        self._run = None
