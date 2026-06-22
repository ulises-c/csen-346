# Handoff — Stage 2 Gemma 4 31B QLoRA crawl

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up after session 10.

---

## TL;DR

Training is **RUNNING** — monitor auto-started at session begin, recovered from one crash,
and is at step **1305/4826 ≈ 27.0%** as of 2026-06-06 19:35.

Working tree is **clean**. All session 9–10 changes committed.

Progress banked: **checkpoint-1290 / 4826 steps ≈ 26.7% complete.**

Session 9 focus: attempted standalone reproduction of `MT64x64x64_ISA1201_DTVB1` tile.
Nine probe variants all dispatch `MT128x128x32` instead. The leading hypothesis is a
**system-vs-torch-bundled rocBLAS library difference** (see Open items). The dense-VA
reproducer script is now committed. Upstream #7992 update is due.

---

## Immediate actions

### 1. Check training status (likely already running)

```bash
pgrep -f monitor_stage2 && echo 'RUNNING' || echo 'NOT RUNNING'
tail -20 outputs/monitor_stage2.log
# If not running:
make gpu-preflight   # must PASS
nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &
```

### 2. File the AMD upstream issue (SYNC log parsed — ready to file)

All evidence is collected. File at **https://github.com/ROCm/rocm-libraries/issues**,
component: **rocBLAS / Tensile**.

Include:
- `scripts/repro_gfx1201_dense.py` (improved dense-VA standalone reproducer)
- Full ShaderName from run #9 / diagnostic log
- Operand descriptors: A=(608,21504) row-major bf16, B=(21504,5376) col-major NF4→bf16
- Fault addresses (all 2 MB-aligned — confirmed in SYNC log: fault `0x7efb33000000`)
- Key finding: `bias=True` required (BH_Bias_UserArgs epilogue selects this tile); col-major B
  is intrinsic to PyTorch + bitsandbytes 4bit matmul — no userspace workaround possible
- Env snapshot: `docs/diagnostics/gfx1201-report-env-20260605-101605.txt`
- SYNC log confirms same `MT64x64x64_ISA1201_DTVB1_VWA2_VWB1` ShaderName at fault point
- Tile selection divergence note: standalone probe dispatches `MT128x128x32`, not `MT64x64x64`.
  **System-rocBLAS hypothesis DISPROVED** (session 10): both system (`/opt/rocm`) and
  torch-bundled libraries dispatch identically — MT128x128x32 for all probe variants tested.
  Tile divergence between standalone and production remains unexplained; AMD can reproduce
  via rocprof on the full model training loop.

---

## Current state

| Item | Value |
|---|---|
| Training script | `scripts/train_sft.py` via `make train-gemma4-31b-stage2-unsloth` |
| Monitor | **RUNNING** — active as of 2026-06-06 19:35, step 1305/4826 |
| Monitor log | `outputs/monitor_stage2.log` |
| Train log | `outputs/sft-stage2-gemma4-31b/train.log` |
| Latest checkpoint | `checkpoint-1290` (safe); actively training past it |
| Total steps | 4,826 (1 epoch, 77k records, batch 1×16) |
| Per-step time | ~71 s/it (async); slower under SYNC |
| GPU state | **Clean** — GPU preflight PASSED (post-reboot, session 10) |
| HF backup repo | `ulises-c/SocratesLM-31B-stage2b-QLoRA` (auto-push every 50 steps) |
| W&B project | `csen346-sft` at `uchavarria-santa-clara-university` |

---

## Recurring venv issue — check this first every session

**`torch+rocm7.2` gets silently overwritten by `torch+cu130` (CUDA build)** after `uv sync`
or any pip/uv install outside the Makefile. This has happened in sessions 6 and 7.

Always verify at the start of any session before touching the GPU:

```bash
uv run --no-sync python -c "import torch; print(torch.version.hip)"
# Must print e.g. "7.2.26015". If it prints "None" → ROCm torch is missing.
make install-rocm   # fix: reinstalls torch==2.11.0+rocm7.2
```

---

## How the crawl works

`monitor_stage2.sh` runs a loop: launch training → poll every 5 min → on crash: archive
log + dmesg, clean GPU (waits up to 180s for KFD to drain), quarantine any partial
checkpoint, relaunch with a **fresh `TRAIN_DATA_SEED=$(date +%s)`**.

The rotating seed (commit `587b60e`) reshuffles the post-resume sample sequence each cycle,
converting the sticky-deterministic gfx1201 page fault back to probabilistic. At a fixed seed
the fault occurs deterministically within steps 22–24 from checkpoint-20; from checkpoint-1230
the window is 0–100 steps (wider, still seed-dependent).

**Forward-progress guard:** `MAX_RETRIES=8` consecutive no-progress retries → monitor posts
`STALLED` to the Training Log issue (#120) and exits. A new checkpoint resets the counter.

**Where posts go:** all events — progress (every `PROGRESS_EVERY=50` steps, with
time/loss/grad_norm/lr), crash, STALLED, COMPLETE — append a row to a **single pinned
comment** (the "Live training log" table) on **issue #120**, not PR #101. The comment id is
pinned in `monitor_stage2.sh` as `LOG_COMMENT_ID`; `log_row` fetches that comment, appends a row,
and PATCHes it back, so restarts and `outputs/` wipes never lose history or mint a second comment.
**Starting a brand-new crawl:** create a fresh placeholder comment on #120 and paste its numeric id
into `LOG_COMMENT_ID` (same "bump for a new run" convention as `WANDB_RUN_ID`).

**HF auto-backup:** `HFCheckpointCallback` (via `TRAIN_HF_REPO` env var set in Makefile)
pushes each saved checkpoint to `ulises-c/SocratesLM-31B-stage2b-QLoRA` in a daemon thread,
at a cadence of `TRAIN_HF_PUSH_EVERY` (default 50) step boundaries.

- **Persistence:** After each successful push, the step is recorded in `{output_dir}/.hf_last_push`.
  On resume, the callback reads this file so already-uploaded checkpoints are skipped.
- **Crash recovery:** On `on_init` (trainer initialisation), the callback scans
  `output_dir/checkpoint-*`, picks the highest step, and pushes it **synchronously**
  if `.hf_last_push` reports a lower step. This catches the scenario where a daemon
  thread was killed mid-push by a crash — the checkpoint gets uploaded before training
  resumes, so it is never lost.
- **Thread safety:** The existing `self._thread.is_alive()` guard prevents concurrent
  pushes. Since the launch push is synchronous, there is no race with training-time pushes.
- **On-disk limit:** `save_total_limit=5` keeps only the last 5 local checkpoints;
  HF stores every checkpoint pushed.
- **Logging verbosity:** `TRANSFORMERS_VERBOSITY=error` (level 1) is now set in
  `train_sft.py` to suppress noisy INFO (level 3) output during long SFT runs.

---

## If training is running

Leave it alone. Watch issue #120 (Training Log) for crash/progress posts. The training is autonomous — no
action needed unless:
- The monitor posts `STALLED` → see below
- A crash post shows something unexpected (OOM, not a page fault)
- Loss diverges (grad_norm consistently > 50, loss climbing after step 100)

---

## If training crashed and the monitor is handling it

Normal — monitor will clean, reseed, and relaunch. You'll see an issue #120 comment like
`**Stage 2 CRASHED** (no-progress retry N/8, latest ckpt step X)`. No action needed
unless it STALLs.

---

## If the monitor STALLED

Monitor posted: `## Stage 2 Training: STALLED (no progress in 8 retries)`

**Most likely cause:** wedged GPU KFD context. Fix sequence:

1. Verify GPU dirty: `rocm-smi --showmeminfo vram` → expect ~24 GB despite no processes
2. `lsof /dev/kfd /dev/dri/renderD128` → `pkill nvtop` if listed; recheck after 5s
3. If still dirty: `sudo modprobe -r amdgpu && sudo modprobe amdgpu`
4. `make gpu-preflight` → must PASS
5. `nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &`
6. Watch for tokenization deadlock on first restart: if `train.log` stalls at
   `Tokenizing train dataset (num_proc=12): XX%` for >60s without advancing, the
   multiprocessing pool deadlocked. `pkill -9 -f 'train_sft\.py'` — monitor will retry.
   If repeated: patch `train_sft.py` to use `dataset_num_proc=1`.
7. If STALLs again after clean GPU → note in issue #120, escalate to cloud GPU.

---

## If training completed

Monitor posted: `## Stage 2 Training: COMPLETE ✓`

Adapter at `outputs/sft-stage2-gemma4-31b/final/`. Next steps:

1. **Verify adapter loads:**
   ```bash
   uv run --no-sync python -c "
   from peft import PeftModel
   m = PeftModel.from_pretrained('outputs/sft-stage2-gemma4-31b/final',
       'unsloth/gemma-4-31B-it-unsloth-bnb-4bit')
   print('OK')
   "
   ```
2. Merge and export: `uv run --no-sync python scripts/merge_lora_gemma4_sft.py`
3. Run downstream eval (BERT consultant routing + LLM judge) — post results to issue #120
4. Update `docs/GFX1201_FAULT_ABLATION_LOG.md` with final step count + fault events

---

## Key invariants — do not violate

- **Always `uv run --no-sync`** — bare `uv run` reinstalls CUDA torch over ROCm
- **Always verify `torch.version.hip` is not None** before any GPU work (regression recurs)
- **`make gpu-preflight` before any manual launch** — dirty KFD cascades into early faults
- **Do not add `AMD_SERIALIZE_KERNEL` or `HIP_LAUNCH_BLOCKING` to the monitor** — SYNC mode
  makes the fault deterministic; all SYNC runs are diagnostic only
- **Do not restart the monitor without checking if it's already running** (`pgrep -f monitor_stage2`)

---

## What is closed — do not re-investigate

The gfx1201 ISA1201 Tensile GEMM bug is **fully root-caused and documented**. All
ablation arms are exhausted. See `docs/GFX1201_FAULT_ABLATION_LOG.md`.

The `ROCBLAS_LAYER=2` bench logging path is a dead end through PyTorch: PyTorch routes
through `libhipblas.so → librocblas.so` internal dispatch, bypassing the C API logging
hooks. Both B-matrix encodings for `rocblas-bench gemm_ex` dispatch `ISA000` (MLIR
generic), never `ISA1201`. Confirmed in session 6.

The standalone reproducer (`scripts/repro_gfx1201_rocblas.py`) dispatches the faulting
kernel but does NOT crash in a small process — the wild address lands on mapped VA because
the process VA footprint is too small. This is expected and documented in the script. Do not
re-investigate or try to make it crash in isolation.

Do not:
- Re-run probe scripts (probes 1–3 done; `capture-rocblas-bench.sh` / `replay-rocblas-bench.sh` confirmed dead end)
- Try `TORCH_USE_HIPBLASLT=1` (run #11 — no kernel for this shape, Tensile fallback)
- Try `BNB_FORCE_B_CONTIGUOUS=1` (run #12 — Python `.contiguous()` can't reach BLAS descriptor)
- Try `expandable_segments:True` (run #10 — silently ignored on gfx1201)
- Try `ignore_data_skip=True` (rejected — silent garbage adapter, no stall alarm)

---

## Open items

- **File AMD upstream issue** — all evidence collected (see Immediate actions §2 above).
  URL: https://github.com/ROCm/rocm-libraries/issues · component: rocBLAS / Tensile.
  SYNC log parsed: fault `0x7efb33000000` (2MB-aligned), `MT64x64x64_ISA1201_DTVB1_VWA2_VWB1`
  kernel confirmed at crash point, last B-operand probe shape=(21504,5376) col-major.
  System-rocBLAS hypothesis CLOSED in session 10 (both libraries dispatch identically).

- **Monitor running** — no action needed. Check `tail -20 outputs/monitor_stage2.log` for status.

---

## Session history

**Session 4 (2026-06-02):** Crawl launched from checkpoint-90. Forward progress to
checkpoint-1230 overnight (~50 crashes, all recovered). Tokenization deadlock after
step-1230 crash wedged the GPU; monitor STALLed 8/8 at 17:19 on 2026-06-03.

**Session 5 (2026-06-04):** Monitor and training dead, GPU dirty (24 GB wedged KFD).
No code changes. Commits: env snapshot script, rocBLAS probe scripts.

**Session 6 (2026-06-05 AM):**
- Ran `capture-rocblas-bench.sh` + `replay-rocblas-bench.sh` → confirmed `ROCBLAS_LAYER`
  dead through PyTorch's hipBLAS path; `rocblas-bench gemm_ex` only dispatches `ISA000`
- Fixed venv torch build regression (first occurrence): `torch+cu130` → `torch+rocm7.2`
- Env snapshot: `docs/diagnostics/gfx1201-report-env-20260605-101605.txt`
- Created HF model repo `ulises-c/SocratesLM-31B-stage2b-QLoRA` with README + ATTRIBUTION
- `scripts/train_sft.py`: `save_total_limit` 2→5; `HFCheckpointCallback` added
- Extracted `HFCheckpointCallback` → `src/project/hf_callback.py` + 7 unit tests
- Commits: `71d6111` (refactor HFCheckpointCallback), `e6ed01a` (HF auto-push + limit)

**Session 7 (2026-06-05 PM):**
- GPU was clean at start — no driver reset needed (24 GB wedge from session 4 was gone)
- ROCm torch regression recurred: `torch+cu130` again; fixed with `make install-rocm`
- **Wild address analysis:** all 12 production crash addresses are exactly 2 MB-aligned;
  fault consistently lands ~1.6 GB above B.end regardless of ASLR
- **Standalone reproducer written:** `scripts/repro_gfx1201_rocblas.py` — `bnb.nn.Linear4bit`
  at exact fault shape (in=21504, out=5376, bias=True, nf4, bf16); does not crash in small
  process (sparse VA); confirmed behaviour expected and documented
- **Diagnostic SYNC run launched** from checkpoint-1230: AMD_SERIALIZE_KERNEL=3,
  AMD_LOG_LEVEL=3, BNB_DEQUANT_PROBE=1, TRAIN_SAVE_STEPS=99999. Log at
  `docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log`
- Monitor NOT restarted — diagnostic run occupied GPU at end of session
- Commit: `109eb72` (standalone reproducer + codespell *.log skip)

**Session 9 (2026-06-06):**
- Machine rebooted to clear 31 GB wedged KFD context (stuck from probe run ending session)
- **Dense-VA reproducer run** (`scripts/repro_gfx1201_dense.py`): 20 GB filler allocated,
  200 iters forward+backward — no crash. `MT128x128x32_ISA1201_DTVB1` dispatched 401 times.
  `MT64x64x64` NOT dispatched (only appears in script's own print statement).
- **Tile-selection probe** (`scripts/probe_gfx1201_tile.py`): 9 variants tested — all
  dispatch `MT128x128x32`. Variants: 2D/3D shape, with/without `torch.utils.checkpoint`
  (reentrant=False and True), `torch.autocast`, skip-warmup (cold Tensile cache),
  non-contiguous A (stride=[43008,1]).
- **Bitsandbytes forward traced**: `dequantize_4bit` returns `(21504,5376)` col-major
  (stride=(1,21504)). After `.t()` = `(5376,21504)` row-major. `F.linear` calls
  `addmm(bias, A, W.t())` where `W.t()` = `(21504,5376)` col-major → this IS the DTVB1
  path. Both standalone and production make structurally identical BLAS calls.
- **Leading hypothesis**: torch-bundled rocBLAS library (`torch/lib/rocblas/library/`)
  does not include `MT64x64x64` kernels for gfx1201; system ROCm 7.2 library
  (`/opt/rocm/lib/rocblas/library/`) does. `ROCBLAS_TENSILE_LIBPATH` override blocked by
  Railguard path fence — needs user approval for next session.
- New probe script committed: `scripts/probe_gfx1201_tile.py`
- Comment posted to issue #113 with full findings
- Commits: `aac092f` (dense-VA reproducer + handoff update)

**Session 10 (2026-06-06 evening):**
- GPU preflight PASSED (clean post-reboot)
- Monitor was already running (PID 17208) — auto-started by the session's environment
- Training recovered from one crash and progressed: step 1305/4826 at 19:35 PST
- **System-rocBLAS hypothesis DISPROVED**: compared `probe_3d_plain.log` vs
  `probe_system_rocblas.log` — both dispatch `MT128x128x32` identically. The two apparent
  `MT64x64x64` hits were from the probe script's own `print()` statements, not actual
  kernel dispatches. System library and torch-bundled library are equivalent for gfx1201.
  Tile selection divergence (standalone→MT128x128x32, production→MT64x64x64) remains
  unexplained at the binary level; AMD must reproduce via rocprof on the full model.
- Commits: this session (handoff + probe script formatting)

**Session 8 (2026-06-05 evening):**
- **SYNC diagnostic log parsed** (26 GB, 230M lines): Fault confirmed at `0x7efb33000000`
  (2MB-aligned). `MT64x64x64_ISA1201_DTVB1_VWA2_VWB1` ShaderName confirmed at crash point
  — same kernel as all prior crashes (runs #5, #7, #9). Last B-operand probe: shape=(21504,5376)
  col-major. SYNC fault occurred within ~1–2 steps of resuming checkpoint-1230 (SYNC makes
  it deterministic). All evidence ready for AMD upstream report.
- **Regular crawl** progressed from checkpoint-1230 → checkpoint-1290 (60 more steps at ~71 s/it)
- Training crashed again (~step 1290+), process was spinning at 141% CPU
- All training/wandb processes killed (PIDs 372896, 372901, 373486, 373502)
- Monitor NOT restarted — GPU dirty after crash

```
109eb72  feat(diag): standalone gfx1201 ISA1201 Tensile GEMM reproducer
71d6111  refactor(train): extract HFCheckpointCallback to module level + tests
e6ed01a  feat(train): HF auto-push callback + raise save_total_limit to 5
587b60e  feat(monitor): rotate TRAIN_DATA_SEED per resume to break sticky fault
```

---

## Key files

| File | Purpose |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Canonical run log — append after every run |
| `scripts/monitor_stage2.sh` | Crawl harness — rotating seed, KFD cleanup, issue #120 progress/crash posts |
| `scripts/train_sft.py` | Training script; HF auto-push + `TRAIN_DATA_SEED` + `BNB_DEQUANT_PROBE` wired |
| `scripts/repro_gfx1201_rocblas.py` | Standalone AMD upstream reproducer (dispatches ISA1201 kernel) |
| `src/project/hf_callback.py` | `HFCheckpointCallback` — async HF push with `.hf_last_push` persistence, `on_init` crash recovery, skip-if-in-flight |
| `tests/test_hf_callback.py` | 14 unit tests for `HFCheckpointCallback` |
| `outputs/sft-stage2-gemma4-31b/crashlogs/` | Per-crash full log + dmesg archive |
| `docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log` | Session 7 diagnostic — parse first |
| `docs/diagnostics/gfx1201-report-env-20260605-101605.txt` | Env snapshot for AMD upstream report |
| PR #101 | Full diagnostic thread + all session findings |
