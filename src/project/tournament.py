"""
Tournament system for comparing multiple LLMs on the KELE Socratic Teaching benchmark.

Sequential evaluation (one model per 32 GB VRAM slot at a time):
  round N: boot model → run n=50 eval → kill server → record state_accuracy
  eliminate: drop worst performer(s) until 3 finalists remain
  finalize:  run survivors to n=681

Crash-safe: state persists to results/tournament/state.json after each model.

CLI:
  uv run tournament run [--n N] [--unified] [--thinking-budget N]
  uv run tournament status
  uv run tournament eliminate [N]
  uv run tournament finalize [--unified]
  uv run tournament archive
  uv run tournament restore [<timestamp>]
  uv run tournament reset
  uv run tournament download
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = REPO_ROOT / "results" / "tournament" / "state.json"
PORT = int(os.environ.get("PORT", "8080"))
LLAMA_URL = f"http://localhost:{PORT}"


def _find_llama_server() -> Path:
    if "LLAMA_SERVER" in os.environ:
        return Path(os.environ["LLAMA_SERVER"])
    candidates = [
        Path.home() / "Documents" / "models" / "llama.cpp" / "build" / "bin" / "llama-server",
        Path.home() / "Github" / "llama.cpp" / "build" / "bin" / "llama-server",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


LLAMA_SERVER = _find_llama_server()
os.environ.setdefault("LLAMA_SERVER", str(LLAMA_SERVER))


@dataclass
class ModelSpec:
    id: str
    name: str
    alias: str
    weight_path: str  # ~ expanded at runtime
    serve_script: str  # relative to repo root
    config_name: str  # configs/<name>.env — used for no-think runs
    hf_repo: str
    hf_file: str  # primary file (first shard for split GGUFs); used for display and existence check
    hf_include: str | None = None  # glob for --include when files live in a subdir or are split
    on_disk: bool = True
    thinking_config_name: str | None = None  # configs/<name>.env — used when thinking_budget > 0


MODEL_REGISTRY: list[ModelSpec] = [
    # ── On disk ──────────────────────────────────────────────────────────────
    ModelSpec(
        id="qwen35-9b",
        name="Qwen3.5-9B Q4",
        alias="Qwen 9B Q4",
        weight_path="~/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        serve_script="scripts/serve_tournament_qwen35_9b.sh",
        config_name="tournament-qwen35-9b",
        thinking_config_name="tournament-qwen35-9b-think",
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_file="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwen27b",
        name="Qwen3.6-27B Q5",
        alias="Qwen 27B Q5",
        weight_path="~/Documents/models/weights/Qwen3.6-27B-UD-Q5_K_XL.gguf",
        serve_script="scripts/serve_qwen27b_q5.sh",
        config_name="qwen27b-local",
        thinking_config_name="tournament-qwen27b-think",
        hf_repo="unsloth/Qwen3.6-27B-GGUF",
        hf_file="Qwen3.6-27B-UD-Q5_K_XL.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwen27b-q4",
        name="Qwen3.6-27B Q4",
        alias="Qwen 27B Q4",
        weight_path="~/models/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf",
        serve_script="scripts/serve_qwen27b_q4_local.sh",
        config_name="tournament-qwen27b-q4",
        thinking_config_name="tournament-qwen27b-q4-think",
        hf_repo="unsloth/Qwen3.6-27B-GGUF",
        hf_file="Qwen3.6-27B-Q4_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwen35b-a3b",
        name="Qwen3.6-35B-A3B Q4 MoE",
        alias="Qwen 35B A3B",
        weight_path="~/Documents/models/weights/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        serve_script="scripts/serve_qwen35b_a3b.sh",
        config_name="qwen35b-a3b-local",
        thinking_config_name="tournament-qwen35b-a3b-think",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        hf_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="gemma4-31b",
        name="Gemma 4 31B Q5",
        alias="Gemma 4 31B",
        weight_path="~/Documents/models/weights/gemma-4-31B-it-UD-Q5_K_XL.gguf",
        serve_script="scripts/serve_gemma4_31b_q5.sh",
        config_name="gemma4-31b-local",
        hf_repo="unsloth/gemma-4-31b-it-GGUF",
        hf_file="gemma-4-31B-it-UD-Q5_K_XL.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="gemma4-26b-a4b",
        name="Gemma 4 26B-A4B Q4 MoE",
        alias="Gemma 4 26B A4B",
        weight_path="~/Documents/models/weights/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        serve_script="scripts/serve_gemma4_26b_a4b.sh",
        config_name="tournament-gemma4-26b-a4b",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        hf_file="gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        on_disk=True,
    ),
    # ── Downloadable (add more as weights arrive) ─────────────────────────────
    ModelSpec(
        id="glm47-23b",
        name="GLM-4.7-Flash REAP 23B-A3B Q4",
        alias="GLM 23B A3B",
        weight_path="~/models/GLM-4.7-Flash-REAP-23B-A3B/GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf",
        serve_script="scripts/serve_glm47_23b.sh",
        config_name="tournament-glm47-23b",
        thinking_config_name="tournament-glm47-23b-think",
        hf_repo="unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF",
        hf_file="GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwopus35b-a3b",
        name="Qwopus3.6-35B-A3B Q4",
        alias="Qwopus 35B A3B",
        weight_path="~/models/Qwopus3.6-35B-A3B-v1/Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf",
        serve_script="scripts/serve_qwopus35b_a3b.sh",
        config_name="tournament-qwopus35b-a3b",
        thinking_config_name="tournament-qwopus35b-a3b-think",
        hf_repo="Jackrong/Qwopus3.6-35B-A3B-v1-GGUF",
        hf_file="Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="deepseek-r1-14b",
        name="DeepSeek-R1-Distill-Qwen-14B Q5",
        alias="DeepSeek R1 14B",
        weight_path="~/Documents/models/weights/DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf",
        serve_script="scripts/serve_deepseek_r1_14b.sh",
        config_name="tournament-deepseek-r1-14b",
        thinking_config_name="tournament-deepseek-r1-14b-think",
        hf_repo="unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        hf_file="DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="phi4-14b",
        name="Phi-4 14B Q5",
        alias="Phi 4 14B",
        weight_path="~/Documents/models/weights/phi-4-Q5_K_M.gguf",
        serve_script="scripts/serve_phi4_14b.sh",
        config_name="tournament-phi4-14b",
        hf_repo="unsloth/phi-4-GGUF",
        hf_file="phi-4-Q5_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="gemma3-27b",
        name="Gemma 3 27B Q4",
        alias="Gemma 3 27B",
        weight_path="~/Documents/models/weights/gemma-3-27b-it-Q4_K_M.gguf",
        serve_script="scripts/serve_gemma3_27b.sh",
        config_name="tournament-gemma3-27b",
        hf_repo="unsloth/gemma-3-27b-it-GGUF",
        hf_file="gemma-3-27b-it-Q4_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="mistral-24b",
        name="Mistral-Small-3.2-24B Q4",
        alias="Mistral Small 24B",
        weight_path="~/Documents/models/weights/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        serve_script="scripts/serve_mistral_24b.sh",
        config_name="tournament-mistral-24b",
        hf_repo="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        hf_file="Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwen3-14b",
        name="Qwen3-14B Q4",
        alias="Qwen 14B Q4",
        weight_path="~/Documents/models/weights/Qwen3-14B-UD-Q4_K_XL.gguf",
        serve_script="scripts/serve_qwen35_14b.sh",
        config_name="tournament-qwen35-14b",
        thinking_config_name="tournament-qwen35-14b-think",
        hf_repo="unsloth/Qwen3-14B-GGUF",
        hf_file="Qwen3-14B-UD-Q4_K_XL.gguf",
        on_disk=True,
    ),
]

MODEL_BY_ID: dict[str, ModelSpec] = {m.id: m for m in MODEL_REGISTRY}


# ── Tournament state ──────────────────────────────────────────────────────────


def _new_run_id() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


@dataclass
class TournamentState:
    run_id: str = field(default_factory=_new_run_id)
    round: int = 0
    n_per_round: int = 50
    thinking_budget: int = 0  # 0 = disabled; >0 = token budget passed to chat_template_kwargs
    models_active: list[str] = field(default_factory=list)
    models_eliminated: list[str] = field(default_factory=list)
    # {model_id: {round_N: overall_score}}
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    # models that completed the current round (crash recovery)
    round_complete: list[str] = field(default_factory=list)
    final_scores: dict[str, float] = field(default_factory=dict)
    # {model_id: {round_N: full_metrics_dict}}
    metrics: dict[str, dict[str, dict]] = field(default_factory=dict)


def load_state() -> TournamentState:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            data = json.load(f)
        # Back-compat: states written before run_id was added get one assigned now
        if not data.get("run_id"):
            data["run_id"] = _new_run_id()
        return TournamentState(**data)
    return TournamentState(
        round=0,
        n_per_round=50,
        models_active=[m.id for m in MODEL_REGISTRY if m.on_disk],
        models_eliminated=[],
        scores={m.id: {} for m in MODEL_REGISTRY if m.on_disk},
    )


def save_state(state: TournamentState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2)


# ── Server lifecycle ──────────────────────────────────────────────────────────


def _probe_server() -> str | None:  # pragma: no cover
    try:
        import urllib.request

        with urllib.request.urlopen(f"{LLAMA_URL}/v1/models", timeout=3) as r:
            return r.read().decode()
    except Exception:
        return None


def _server_has_alias(response: str, alias: str) -> bool:
    return f'"{alias}"' in response


def _server_ready(alias: str) -> bool:  # pragma: no cover
    resp = _probe_server()
    if not resp:
        return False
    if '"Loading model"' in resp:
        return False
    return _server_has_alias(resp, alias)


def _boot_server(spec: ModelSpec) -> subprocess.Popen[bytes]:  # pragma: no cover
    script = REPO_ROOT / spec.serve_script
    if not script.exists():
        print(f"  ERROR: serve script not found: {script}", file=sys.stderr)
        sys.exit(1)
    weight = Path(spec.weight_path).expanduser()
    if not weight.exists():
        print(f"  ERROR: weight file not found: {weight}", file=sys.stderr)
        sys.exit(1)
    proc = subprocess.Popen(
        ["bash", str(script)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _kill_server(proc: subprocess.Popen[bytes] | None) -> None:  # pragma: no cover
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        for _ in range(15):
            if proc.poll() is not None:
                return
            time.sleep(1)
        proc.kill()
    except OSError:
        pass


def ensure_server(
    spec: ModelSpec,
) -> tuple[subprocess.Popen[bytes] | None, bool]:  # pragma: no cover
    """Return (proc, we_booted). Boots the server if not already running."""
    resp = _probe_server()
    if resp is not None:
        if _server_has_alias(resp, spec.alias):
            if '"Loading model"' not in resp:
                print(f"  Reusing existing server (alias '{spec.alias}' ready).")
                return None, False
            # Already loading our model — just wait
            print(f"  Existing server loading '{spec.alias}' — waiting...")
            proc = None
        else:
            print(
                f"  ERROR: server is up but serves a different model (want '{spec.alias}').\n"
                f"  Stop the existing server first.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"  Booting server for {spec.name} (may take 30-120 s)...")
        proc = _boot_server(spec)

    print("  Waiting for server", end="", flush=True)
    for i in range(180):
        if _server_ready(spec.alias):
            print(f" ready (~{(i + 1) * 2}s)")
            return proc, True
        if proc is not None and proc.poll() is not None:
            print(" FAILED — server exited early", file=sys.stderr)
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(2)
    print(" TIMEOUT after 360 s", file=sys.stderr)
    _kill_server(proc)
    sys.exit(1)


# ── Eval runner ───────────────────────────────────────────────────────────────


def run_kele(
    spec: ModelSpec, out_dir: Path, n: int, unified: bool, thinking_budget: int = 0
) -> dict | None:  # pragma: no cover
    """Run kele test --n N and return the full metrics dict, or None on failure."""
    if thinking_budget > 0 and spec.thinking_config_name:
        config = spec.thinking_config_name
    else:
        config = spec.config_name
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.project.kele",
        "--experiment",
        config,
        "test",
        "--n",
        str(n),
        "--output",
        str(out_dir),
    ]
    if unified:
        cmd.append("--unified")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  kele exited with code {result.returncode}", file=sys.stderr)
        return None

    metrics_file = out_dir / "metrics_summary.json"
    if not metrics_file.exists():
        print(f"  metrics_summary.json not found at {metrics_file}", file=sys.stderr)
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)
    if metrics.get("state_accuracy", {}).get("overall") is None:
        print("  state_accuracy.overall missing from metrics", file=sys.stderr)
    return metrics


# ── Leaderboard ───────────────────────────────────────────────────────────────


def print_leaderboard(state: TournamentState) -> None:
    round_key = f"round{state.round}"
    prev_key = f"round{state.round - 1}" if state.round > 1 else None

    # Build rows: (name, latest, best, delta, status, hist_str)
    active_rows = []
    for mid in state.models_active:
        spec = MODEL_BY_ID.get(mid)
        name = spec.name if spec else mid
        scores = state.scores.get(mid, {})
        latest = scores.get(round_key)
        best = max(scores.values()) if scores else None
        prev = scores.get(prev_key) if prev_key else None
        delta = (latest - prev) if (latest is not None and prev is not None) else None
        hist = "  ".join(f"r{k[5:]}: {v:.3f}" for k, v in sorted(scores.items()))
        active_rows.append((name, latest, best, delta, "active", hist))

    elim_rows = []
    for mid in state.models_eliminated:
        spec = MODEL_BY_ID.get(mid)
        name = spec.name if spec else mid
        scores = state.scores.get(mid, {})
        best = max(scores.values()) if scores else None
        hist = "  ".join(f"r{k[5:]}: {v:.3f}" for k, v in sorted(scores.items()))
        elim_rows.append((name, None, best, None, "ELIMINATED", hist))

    active_rows.sort(key=lambda r: -(r[1] or -99))
    rows = active_rows + elim_rows

    def _score(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "    —"

    def _delta(d: float | None) -> str:
        if d is None:
            return "    —"
        return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

    print()
    print(
        f"  {'#':>2}  {'Model':<34}  {'Latest':>7}  {'Best':>7}  {'Δ':>7}  {'Status':<11}  History"
    )
    print("─" * 105)
    for i, (name, latest, best, delta, status, hist) in enumerate(rows):
        rank = f"{i + 1:>2}" if status == "active" else "  "
        marker = "✓" if status == "active" else "✗"
        print(
            f"  {rank}  {name:<34}  {_score(latest):>7}  {_score(best):>7}  {_delta(delta):>7}"
            f"  {status:<11}  {hist}  {marker}"
        )
    print()
    print(
        f"Round: {state.round}  |  n/round: {state.n_per_round}"
        f"  |  Active: {len(state.models_active)}  |  Eliminated: {len(state.models_eliminated)}"
    )
    print()

    # ── Per-round metrics detail table ────────────────────────────────────────
    all_round_keys = sorted(
        {rk for scores in state.scores.values() for rk in scores},
        key=lambda k: int(k[5:]),
    )
    if state.metrics and all_round_keys:
        print(
            f"  {'Model':<34}  {'Round':<7}  {'Turns':>5}  {'R1':>6}  {'R2':>6}"
            f"  {'RL':>6}  {'BLEU':>5}  {'stg-a':>5}  {'stg-b':>5}  {'stg-c':>5}"
            f"  {'stg-d':>5}  {'stg-e':>5}  {'Score':>6}  {'Time':>6}"
        )
        print("─" * 145)
        all_mids = list(state.models_active) + list(state.models_eliminated)
        for mid in all_mids:
            model_metrics = state.metrics.get(mid, {})
            if not model_metrics:
                continue
            spec = MODEL_BY_ID.get(mid)
            name = spec.name if spec else mid
            for rk in all_round_keys:
                m = model_metrics.get(rk)
                if m is None:
                    continue
                sa = m.get("state_accuracy", {})
                ps = sa.get("per_stage", {})

                def _pct(v: float | None) -> str:
                    return f"{v:5.1f}" if v is not None else "    —"

                elapsed_s = m.get("elapsed_secs")
                time_str = (
                    f"{elapsed_s // 60}m{elapsed_s % 60:02d}s"
                    if elapsed_s is not None
                    else "     —"
                )
                print(
                    f"  {name:<34}  {rk:<7}  {m.get('n_turns', '?'):>5}"
                    f"  {_pct(m.get('rouge1')):>6}  {_pct(m.get('rouge2')):>6}"
                    f"  {_pct(m.get('rougeL')):>6}  {_pct(m.get('bleu4')):>5}"
                    f"  {_pct(ps.get('a')):>5}  {_pct(ps.get('b')):>5}"
                    f"  {_pct(ps.get('c')):>5}  {_pct(ps.get('d')):>5}"
                    f"  {_pct(ps.get('e')):>5}  {_pct(sa.get('overall')):>6}"
                    f"  {time_str:>6}"
                )
        print()


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_status(_args: argparse.Namespace) -> None:
    state = load_state()
    if state.round == 0 and not state.scores:
        print("No tournament data yet. Run:  uv run tournament run")
        return
    print_leaderboard(state)


def _round_dialogues_incomplete(state: TournamentState) -> bool:
    """True if the current round exists but any active model has fewer than n_per_round dialogues.

    Catches the warmup-then-continue case: warmup runs n=5 with n_per_round=50,
    clears round_complete, so the next `run` call would otherwise bump to the next
    round. Checking actual dialogue counts keeps it on the current round instead.
    """
    if state.round == 0:
        return False
    tournament_dir = REPO_ROOT / "results" / "tournament"
    for mid in state.models_active:
        dlg_dir = tournament_dir / f"round{state.round}" / mid / "dialogues"
        if dlg_dir.exists():
            n_existing = len(list(dlg_dir.glob("*.json")))
            if 0 < n_existing < state.n_per_round:
                return True
    return False


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover
    state = load_state()
    # Start a new round only when not resuming an interrupted/incomplete one.
    # round_complete handles mid-run crashes; _round_dialogues_incomplete handles
    # warmup → continue (e.g. warmup n=5 then full run targeting n_per_round=50).
    if not state.round_complete and not _round_dialogues_incomplete(state):
        state.round += 1
    round_key = f"round{state.round}"
    n = args.n if args.n is not None else state.n_per_round
    # Don't persist an explicit --n override as the new default (e.g. warmup n=5)
    if args.n is None:
        state.n_per_round = n
    unified = args.unified
    if args.thinking_budget is not None:
        state.thinking_budget = args.thinking_budget

    print(f"=== Tournament Round {state.round} ===")
    print(f"n={n}  unified={unified}  active={len(state.models_active)} models")
    print()

    # Warn about time cost
    secs_per_dialogue = 15  # conservative estimate for 32 GB GPU
    est_minutes = (n * secs_per_dialogue * len(state.models_active)) // 60
    print(f"Estimated time: ~{est_minutes} min ({est_minutes // 60}h {est_minutes % 60}m)")
    print()

    for mid in list(state.models_active):
        if mid in state.round_complete:
            score = state.scores.get(mid, {}).get(round_key, "?")
            print(f"[{mid}] already done this round (score={score}) — skipping.")
            continue

        spec = MODEL_BY_ID.get(mid)
        if spec is None:
            print(f"[{mid}] unknown model ID — skipping.", file=sys.stderr)
            continue

        weight = Path(spec.weight_path).expanduser()
        if not weight.exists():
            print(f"[{mid}] weight file not found: {weight} — skipping.")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Model: {spec.name}")
        print(f"  Alias: {spec.alias}")

        proc, _we_booted = ensure_server(spec)

        out_dir = REPO_ROOT / "results" / "tournament" / f"round{state.round}" / mid
        print(f"  Output: results/tournament/round{state.round}/{mid}")

        t0 = time.time()
        metrics = run_kele(spec, out_dir, n, unified, thinking_budget=state.thinking_budget)
        elapsed = int(time.time() - t0)

        _kill_server(proc)

        if metrics is not None:
            score = metrics.get("state_accuracy", {}).get("overall")
            if score is not None:
                state.scores.setdefault(mid, {})[round_key] = score
                print(f"  Score: {score:.3f}")
            metrics["elapsed_secs"] = elapsed
            state.metrics.setdefault(mid, {})[round_key] = metrics
        else:
            print("  Score: ERROR (see output above)")

        if mid not in state.round_complete:
            state.round_complete.append(mid)
        save_state(state)

    state.round_complete = []
    save_state(state)

    print(f"\n{'═' * 60}")
    print_leaderboard(state)


def cmd_eliminate(args: argparse.Namespace) -> None:
    state = load_state()
    n = args.n
    round_key = f"round{state.round}"

    if len(state.models_active) <= 3:
        print(f"Only {len(state.models_active)} model(s) remain — none eliminated.")
        print("Run:  uv run tournament finalize")
        return

    scored = [(mid, state.scores.get(mid, {}).get(round_key)) for mid in state.models_active]
    unscored = [mid for mid, s in scored if s is None]
    if unscored:
        print(f"WARNING: these models have no score for {round_key}: {unscored}")
        print("Run the round first:  uv run tournament run")
        return

    ranked = sorted(scored, key=lambda t: t[1] or 0.0)  # ascending (worst first)
    to_drop = min(n, len(ranked) - 3)  # keep at least 3
    if to_drop <= 0:
        print("No models to eliminate (floor: 3 finalists).")
        return

    for mid, score in ranked[:to_drop]:
        print(f"  Eliminated: {MODEL_BY_ID[mid].name} (score={score:.3f})")
        state.models_active.remove(mid)
        state.models_eliminated.append(mid)

    save_state(state)
    print(f"\n{len(state.models_active)} model(s) remain.")
    print_leaderboard(state)


def cmd_finalize(args: argparse.Namespace) -> None:  # pragma: no cover
    state = load_state()
    if len(state.models_active) > 3:
        print(
            f"WARNING: {len(state.models_active)} models still active. Eliminate down to 3 first.\n"
            f"  uv run tournament eliminate {len(state.models_active) - 3}"
        )

    unified = args.unified
    print(f"=== Tournament Finalize ===  (n=681, unified={unified})")
    print(f"Finalists: {state.models_active}")
    print()

    for mid in state.models_active:
        spec = MODEL_BY_ID.get(mid)
        if spec is None:
            continue
        weight = Path(spec.weight_path).expanduser()
        if not weight.exists():
            print(f"[{mid}] weight file not found: {weight} — skipping.")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Model: {spec.name}")

        proc, _ = ensure_server(spec)
        out_dir = REPO_ROOT / "results" / "tournament" / "final" / mid

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.project.kele",
            "--experiment",
            spec.config_name,
            "evaluate",
            "--output",
            str(out_dir),
        ]
        if unified:
            cmd.append("--unified")

        subprocess.run(cmd, cwd=str(REPO_ROOT))
        _kill_server(proc)

        metrics_file = out_dir / "metrics_summary.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            score = metrics.get("state_accuracy", {}).get("overall")
            if score is not None:
                state.final_scores[mid] = score
                print(f"  Final score: {score:.3f}")
        save_state(state)

    print(f"\n{'═' * 60}")
    print("Final results:")
    for mid, score in sorted(state.final_scores.items(), key=lambda t: -t[1]):
        spec = MODEL_BY_ID.get(mid)
        name = spec.name if spec else mid
        print(f"  {name:<36}  {score:.3f}")


def cmd_add(args: argparse.Namespace) -> None:
    state = load_state()

    if getattr(args, "all", False):
        candidates = [
            m
            for m in MODEL_REGISTRY
            if m.on_disk and m.id not in state.models_active and m.id not in state.models_eliminated
        ]
    elif args.model_id:
        spec = MODEL_BY_ID.get(args.model_id)
        if spec is None:
            print(f"Unknown model ID: {args.model_id}")
            print(f"Known IDs: {', '.join(MODEL_BY_ID)}")
            return
        candidates = [spec]
    else:
        print("Specify a model ID or use --all.")
        return

    added = []
    for spec in candidates:
        if spec.id in state.models_active:
            print(f"[{spec.id}] already active — skipping")
            continue
        if spec.id in state.models_eliminated:
            print(f"[{spec.id}] already eliminated — cannot re-add")
            continue
        if not spec.on_disk:
            print(f"[{spec.id}] not on disk — download first:  uv run tournament download")
            continue
        state.models_active.append(spec.id)
        state.scores.setdefault(spec.id, {})
        added.append(spec.name)
        print(f"  Added: {spec.name}")

    if not added:
        print("Nothing added.")
        return

    save_state(state)
    print(f"\n{len(added)} model(s) added. Run a warmup to test them:")
    print("  uv run tournament run --n 5 --unified")
    print_leaderboard(state)


def cmd_archive(_args: argparse.Namespace) -> None:
    """Move current round dirs + state into a timestamped archive subfolder, then reset."""
    tournament_dir = STATE_FILE.parent
    if not tournament_dir.exists() or not STATE_FILE.exists():
        print("No tournament state to archive.")
        return

    state = load_state()
    ts = _archive_current(tournament_dir)

    # Reset state for the next run (preserve active models and n_per_round)
    new_state = TournamentState(
        round=0,
        n_per_round=state.n_per_round,
        models_active=list(state.models_active),
        models_eliminated=[],
        scores={mid: {} for mid in state.models_active},
    )
    save_state(new_state)

    run_id = ts  # ts is the run_id returned by _archive_current
    print(f"\nArchived  id={run_id}  thinking_budget={state.thinking_budget}")
    print("Fresh state created. To resume a previous run:")
    print("  uv run tournament restore")
    print("To start a new thinking run:")
    print("  uv run tournament run --thinking-budget 8192 --unified")


def _list_archives(tournament_dir: Path) -> None:
    archive_base = tournament_dir / "archive"
    if not archive_base.exists():
        print("No archives found.")
        return
    archives = sorted(archive_base.iterdir())
    if not archives:
        print("No archives found.")
        return
    print(f"  {'ID':<10}  {'Archived at':<20}  {'Round':>5}  {'TB':>6}  {'Active':>6}  Dirs")
    print("  " + "─" * 74)
    for a in archives:
        state_file = a / "state.json"
        if not state_file.exists():
            continue
        with open(state_file) as f:
            s = json.load(f)
        run_id = s.get("run_id", a.name)
        tb = s.get("thinking_budget", 0)
        rnd = s.get("round", "?")
        active = len(s.get("models_active", []))
        archived_at = s.get("archived_at", "—")[:19]
        dirs = [d.name for d in sorted(a.iterdir()) if d.is_dir()]
        print(
            f"  {run_id:<10}  {archived_at:<20}  {rnd:>5}  {tb:>6}  {active:>6}"
            f"  {', '.join(dirs) or '—'}"
        )


def _archive_current(tournament_dir: Path) -> str | None:
    """Archive current state into archive/<run_id>/. Returns the run_id, or None if no state."""
    from datetime import datetime

    if not STATE_FILE.exists():
        return None

    state = load_state()
    run_id = state.run_id
    archive_dir = tournament_dir / "archive" / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    state_dict = asdict(state)
    state_dict["archived_at"] = datetime.now().isoformat()
    with open(archive_dir / "state.json", "w") as f:
        json.dump(state_dict, f, indent=2)

    for item in sorted(tournament_dir.iterdir()):
        if item.is_dir() and item.name not in ("archive",):
            dest = archive_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(item), str(dest))
            print(f"  {item.name}/ → archive/{run_id}/{item.name}/")

    STATE_FILE.unlink(missing_ok=True)
    return run_id


def cmd_restore(args: argparse.Namespace) -> None:
    tournament_dir = STATE_FILE.parent

    if not args.timestamp:
        _list_archives(tournament_dir)
        return

    run_id = args.timestamp
    archive_dir = tournament_dir / "archive" / run_id
    if not archive_dir.exists():
        print(f"Archive not found: archive/{run_id}")
        print("Run without an ID to list available archives.")
        return

    # Archive current state first (non-destructive)
    if STATE_FILE.exists():
        saved_id = _archive_current(tournament_dir)
        print(f"Current state saved to archive/{saved_id}/")
    print()

    # Restore round dirs from archive back into tournament_dir
    restored = []
    for item in sorted(archive_dir.iterdir()):
        if item.is_dir():
            dest = tournament_dir / item.name
            shutil.move(str(item), str(dest))
            restored.append(item.name)

    # Restore state.json (copy — archive entry keeps its snapshot for future re-archives)
    shutil.copy(str(archive_dir / "state.json"), str(STATE_FILE))

    print(f"Restored from archive/{run_id}/")
    if restored:
        print(f"  Round dirs: {', '.join(restored)}")
    state = load_state()
    print(f"  id={state.run_id}  Round: {state.round}  |  thinking_budget: {state.thinking_budget}")
    print()
    print_leaderboard(state)


def cmd_reset(args: argparse.Namespace) -> None:
    tournament_dir = STATE_FILE.parent
    if not args.confirm:
        print(f"This will wipe {tournament_dir} (state + all round output files).")
        print("Re-run with --confirm to proceed.")
        return
    if tournament_dir.exists():
        shutil.rmtree(tournament_dir)
        print(f"Removed {tournament_dir}")
    print("State reset. Run:  uv run tournament run")


def cmd_download(args: argparse.Namespace) -> None:
    dry_run: bool = getattr(args, "dry_run", False)
    missing = [m for m in MODEL_REGISTRY if not Path(m.weight_path).expanduser().exists()]
    if not missing:
        print("All model weights are present locally.")
        return

    label = "Commands (dry-run)" if dry_run else f"Downloading {len(missing)} missing model weights"
    print(f"{label}\n")
    for m in MODEL_REGISTRY:
        weight_path = Path(m.weight_path).expanduser()
        local_dir = weight_path.parent
        if weight_path.exists():
            print(f"[{m.id}] already present — skipping")
            print()
            continue
        print(f"[{m.id}] {m.name}")
        if m.hf_include:
            # Split/sharded or subdirectory GGUF — local_dir is one level above the subdir
            local_dir = weight_path.parent.parent
            cmd = [
                "hf",
                "download",
                m.hf_repo,
                "--include",
                m.hf_include,
                "--local-dir",
                str(local_dir),
            ]
        else:
            cmd = ["hf", "download", m.hf_repo, m.hf_file, "--local-dir", str(local_dir)]
        print(f"  {' '.join(cmd)}")
        if dry_run:
            print()
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if result.returncode != 0:
                print(f"  ERROR: exit {result.returncode}", file=sys.stderr)
            else:
                print(f"  OK — saved to {local_dir}")
        except FileNotFoundError:
            print(
                "  ERROR: 'hf' not found — install with: pip install 'huggingface-hub[cli]'",
                file=sys.stderr,
            )
        print()
    if not dry_run:
        print("Done. Run the tournament with:  uv run tournament run --unified")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KELE tournament: compare LLMs with elimination rounds."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p = sub.add_parser("run", help="Run one elimination round for all active models")
    p.add_argument("--n", type=int, default=None, help="Dialogues per model (default: 50)")
    p.add_argument("--unified", action="store_true", help="Use fusion architecture")
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        dest="thinking_budget",
        metavar="N",
        help="Token budget for model thinking (0 = disabled). Persisted in state.",
    )

    # status
    sub.add_parser("status", help="Print current leaderboard")

    # eliminate
    p = sub.add_parser("eliminate", help="Drop the N worst-scoring models")
    p.add_argument("n", type=int, nargs="?", default=1, help="How many to eliminate (default: 1)")

    # finalize
    p = sub.add_parser("finalize", help="Run survivors to n=681 (full evaluation)")
    p.add_argument("--unified", action="store_true")

    # add
    p = sub.add_parser("add", help="Add a model to the active pool mid-tournament")
    p.add_argument("model_id", nargs="?", default=None, help="Model ID to add")
    p.add_argument("--all", action="store_true", help="Add all on-disk models not yet active")

    # archive
    sub.add_parser(
        "archive",
        help="Move current round dirs + state into a timestamped archive and reset for a new run",
    )

    # restore
    p = sub.add_parser(
        "restore",
        help="Restore a previous archived run (no args = list available archives)",
    )
    p.add_argument(
        "timestamp",
        nargs="?",
        default=None,
        metavar="ID",
        help="Run ID to restore (e.g. a3f2b891). Omit to list available archives.",
    )

    # reset
    p = sub.add_parser("reset", help="Wipe tournament state")
    p.add_argument("--confirm", action="store_true")

    # download
    p = sub.add_parser("download", help="Download pending model weights via hf CLI")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "status": cmd_status,
        "add": cmd_add,
        "eliminate": cmd_eliminate,
        "finalize": cmd_finalize,
        "archive": cmd_archive,
        "restore": cmd_restore,
        "reset": cmd_reset,
        "download": cmd_download,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
