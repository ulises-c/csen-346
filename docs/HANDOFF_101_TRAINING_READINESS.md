# Handoff — Get Stage-2b SFT 100% training-ready (closes #94, finishes #101)

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training`
**Goal:** get a Gemma 4 31B QLoRA Stage-2b run that will **not** repeat the #94 collapse, as the final step of PR #101.

### Machine split (important)
- **You are on the NVIDIA dev box** (no gfx1201, can't load the 31B / can't train here).
- **Do here:** Phase 0 code change (§3), tests (CPU, fully mocked), `--dry-run` plumbing check, then **commit + push** — so the R9700 can pull.
- **Runs on the AMD R9700** (separate box, 32 GB, ROCm 7.2, gfx1201): the short-checkpoint **EOS gate** (§4) and the full QLoRA run (§5). You cannot execute those from here; leave them as the runbook for the R9700 session / human.

> Read top-to-bottom once, then execute §3 (Phase 0) → §3c (tests) → commit+push. §4–§5 are the R9700 runbook. Verify line numbers against the live files before editing — they drift.

---

## 1. What #94 was (the failure you're preventing)

The first Stage-2b SFT model was **generatively broken**: on real eval prompts it never emitted EOS and degenerated into a repetition loop (`这样可以帮助他建立更强的数学基础` ×70+), running to the context limit (~45 min/call). Root cause = **train/serve schema mismatch**: the model was trained on a format that didn't match what `SocraticTeachingSystem.socrates_teacher()` sends at inference. Decisive A/B (same model/GGUF/server): training-shape prompt → 20 tokens + clean EOS; inference-shape prompt → 2048-cap repetition.

The fix is to make the SFT training data render **identically** to the inference prompt. PR #101 fixed the *structural* part. Two residuals remain (Phase 0 below) — finishing them is what makes you "100% ready."

---

## 2. Current state (already done — do NOT redo)

Commits on the branch:
- `d04bc83` — structural schema-drift fix (`socrat-zh-sft` / `socrat-en-sft`: one `(system,user,assistant)` triple per dialogue turn in the inference shape; dialogue-level split) + ROCm/`device_map`/`quantization_config` fixes + W&B.
- `e5a8a7d` — `tests/test_sft_inference_format.py` (11 tests): render-diff gate that drives the **real** `socrates_teacher()` path and asserts the SFT record renders identically under the chat template's per-message `| trim`.
- `492b2a0` — pre-commit hook fix (use `python -m` so it works under `--no-sync`).

Verified facts (don't re-investigate):
- The Gemma 4 training chat template applies `{{ content | trim }}` to every message (`scripts/train_sft.py:121,123`); the serving template trims too. So leading/trailing whitespace differences between train and serve are **immaterial** — render-equivalence reduces to `content.strip()` equality per message.
- The user message already matches inference byte-for-byte **except** the two residual lines below.
- Both inference paths are **Chinese-only** (`socrates_teacher` and the single-call `socratic_teaching_unified.py`); the consultant is Chinese-only.

**Deferred — DO NOT touch for this task:**
- **#108** (bilingual EN/ZH train+serve) — large follow-up; explicitly out of scope. The `socrat-en-sft` loader stays Chinese-scaffolded for now.
- Do **not** modify `socratic_teaching_unified.py` or the `zh` inference rendering in any way that changes rendered tokens — it risks re-baselining the locked headline (unified 72.24) and Tables 6/14.
- **#30** in `ulises-c/Computer-Setup` (global hook fix) — informational only.

---

## 3. Remaining work — Phase 0 (this is the rest of the #94 fix)

The committed loader still drifts from inference on **two lines** of the user message. Both verified against real `ulises-c/SocratDataset` / `SocratDataset-EN` records.

| Line | Loader emits today | Inference actually sends | Fix |
|---|---|---|---|
| `苏格拉底教学顾问建议的操作:` | dataset `turn["action"]` | `get_action_for_state(state)` (canonical Chinese map; `socratic_teaching_system.py:494`) | use `get_action_for_state(state)` — **byte-exact, deterministic** |
| `苏格拉底教学顾问评估结果:` | templated `学生处于 {state} 状态` | live free-form consultant prose | use the dataset's existing free-form `evaluation` field (closer shape) |

Action drift is **systematic**: `a1` (every dialogue's first teaching turn) never matches (`生成一个问题` vs `生成一个与解题相关的子问题`), plus some c-states. The dataset **already carries** a per-turn `evaluation` field (both ZH and EN) — no re-annotation needed.

### 3a. `src/project/socratic_teaching_system.py` — extract a single source of truth

Move the `state_to_action` dict out of `__init__` to module level so the loader can import it (behavior-preserving):

```python
# after `import openai`
_DEFAULT_ACTION = "继续提问"

STATE_TO_ACTION = {
    "a0": "引导学生提出问题",
    "a1": "生成一个与解题相关的子问题",
    # ... copy the FULL existing map verbatim (a0..e34) ...
    "e34": "对题目进行总结",
}

def get_action_for_state(state: str) -> str:
    return STATE_TO_ACTION.get(state, _DEFAULT_ACTION)
```

Then in `__init__` replace the inline dict with:
```python
        self.state_to_action = STATE_TO_ACTION
```
Leave the existing `get_action_for_state(self, state)` method as-is (still delegates via `self.state_to_action`). No inference behavior changes.

### 3b. `src/project/dataset.py` — use the canonical action + free-form eval

Change `_build_inference_user_message` to take `evaluation` instead of `state`:
```python
def _build_inference_user_message(history_turns, student_input, evaluation, action):
    ...
    return (
        f"\n历史对话记录:\n{formatted_history}\n\n"
        f"当前学生输入: {student_input}\n\n"
        f"苏格拉底教学顾问评估结果: {evaluation}\n"
        f"苏格拉底教学顾问建议的操作: {action}\n"
    )
```

In `_socrat_zh_to_sft_records` (and the EN twin):
```python
    from src.project.socratic_teaching_system import get_action_for_state
    ...
    state = _strip_quotes(turn.get("state", ""))                 # EN: turn.get("state", "")
    if not (state and _strip_quotes(turn.get("action", ""))):    # keep target-turn gate on dataset annotation
        history.append((turn["student"], turn["teacher"]))
        continue
    action = get_action_for_state(state)                         # match inference exactly
    evaluation = _strip_quotes(turn.get("evaluation", "")) or f"学生处于 {state} 状态"   # EN: drop _strip_quotes
    user_msg = _build_inference_user_message(history, turn["student"], evaluation, action)
```
Keep gating target-turn selection on dataset `state`+`action` presence so the set of training records is unchanged.

### 3c. Update `tests/test_sft_inference_format.py` (un-rig the action dimension)

The current render-diff gate **passes the dataset action straight into inference**, so it doesn't actually test the action line. After the fix:
1. Add an `evaluation` field to the `_dialogue_record` fixture turns.
2. In the parity test, build the captured inference prompt with `action = get_action_for_state(state)` and `evaluation = <fixture eval>` — assert system **and** user render-equal (now including the action line → zero drift).
3. The freeform-eval residual test: pass a *different* free-form eval → assert the **only** drifting line is `苏格拉底教学顾问评估结果:`.
4. Update unit-test assertions that hardcode `学生处于 a1 状态` / `action_0` to the new outputs (eval = fixture text; action = `get_action_for_state("a1")` = `生成一个与解题相关的子问题`).

Run: `uv run --no-sync python -m pytest tests/test_sft_inference_format.py -q` (must be green).

> Note: this exact Phase-0 change was prototyped and reverted earlier in the session; it is well-understood and low-risk. The `zh` inference path stays byte-rendered-identical (trim).

---

## 4. HARD GATE before the full run — faithful EOS check  *(runs on the R9700, not the dev box)*

After Phase 0, train a short checkpoint, then prove the model terminates on a **real inference-shaped prompt** before trusting the multi-hour run.

**Critical:** the gate prompt must use a **live free-form consultant evaluation**, NOT the templated `学生处于 {state} 状态`. A gate built on the template is blind to the eval-line residual and will pass while shipping a broken model.

Two ways, gold first:
1. **Gold:** serve the trained adapter (merge → GGUF → llama.cpp, same path as eval) and run ONE real turn end-to-end through `SocraticTeachingSystem.process_student_input` (consultant produces the real eval → teacher sees the true inference prompt). Confirm the teacher response emits EOS, is short (≈1 question, <~150 tokens), and does not loop.
2. **Lighter:** capture the inference prompt the way `tests/test_sft_inference_format.py::_capture_inference_prompt` does, but with a realistic multi-sentence Chinese `evaluation`, run HF `model.generate` on the adapter, and confirm clean EOS / no repetition.

Only after the gate passes → commit to the full run.

---

## 5. Pre-launch + launch + monitor  *(runs on the R9700, not the dev box)*

```bash
# 1. GPU must be clean — a prior rc=134 fault leaves the KFD dirty and cascades into random faults
rocm-smi --showpids | grep python && echo "DIRTY — kill -9 and recheck" || echo "clean"

# 2. Dry-run plumbing (optional sanity)
make train-gemma4-31b-dry-run

# 3. Launch (prequant unsloth bnb-4bit; skips the ~62 GB BF16 CPU staging)
make train-gemma4-31b-stage2-unsloth
#   target already sets TORCH_USE_HIPBLASLT=0 and
#   PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,expandable_segments:True

# 4. Monitor
tail -f outputs/sft-stage2-gemma4-31b/train.log
```

**Recompute the run size before launching** — the config comment ("~2298 steps / 23 checkpoints / ~11 GB") is **stale** (per-dialogue era). The per-turn format is ~77k train records → at `bs1 × ga16 × 3 epochs` expect **~14k steps** (multi-day; matches the "~55h" figure in the PR, not the ~12h the stale comment implies). With `TRAIN_SAVE_STEPS=100` that's **~140+ adapter checkpoints (~70 GB)** — raise `TRAIN_SAVE_STEPS` and/or set a `save_total_limit` to avoid filling disk.

Crash recovery: rc=134 → `kill -9 <pid>`, confirm `rocm-smi --showpids` empty before retry; consider `setsid` so the whole tree is killable as one group.

Optional perf: targets default `TORCH_USE_HIPBLASLT=0` (safe). #99 found Gemma 4 may tolerate `=1` (~4× matmul, no linear-attention crash path) — test 20+ steps on a clean GPU before trusting it.

---

## 6. Decisions for the human (surface; don't assume)

1. **Apply Phase 0?** Recommended/required for "100% ready" — removes both residuals. Alternative: train as-is and rely solely on the §4 EOS gate (defensible only if the gate uses a free-form eval; the action line still drifts on every opening turn).
2. **`TRAIN_SAVE_STEPS` / disk** — raise it given ~14k steps (see §5).
3. **hipBLASLt on/off** (perf vs. safety).
4. **2b-only vs 2a→2b** — config defaults to 2b (structural); escalate only if 2b underperforms (see config header / `TRAINING_PLAN.md §4`).

---

## 7. Acceptance checklist ("100% ready")

**On this dev box (you):**
- [ ] Phase 0 applied (3a, 3b); `socrat-*-sft` action line = `get_action_for_state(state)`, eval line = dataset `evaluation`.
- [ ] `uv run --no-sync python -m pytest tests/test_sft_inference_format.py -q` green, with the action dimension un-rigged and eval-line isolated as the sole residual.
- [ ] Full suite green: `uv run --no-sync python -m pytest -q` (expect 154 passed / 2 skipped + your updates).
- [ ] `zh` inference rendering unchanged (no re-baseline) — captured `socrates_teacher` prompt unchanged for a known turn.
- [ ] `make train-gemma4-31b-dry-run` passes (plumbing).
- [ ] **Commit + push** Phase 0 to `feat/gfx1201-rdna4-qlora-fla-training` so the R9700 can pull. → then it's 100% ready to train.

**On the R9700 (separate session/human):**
- [ ] GPU clean (`rocm-smi --showpids`).
- [ ] Run size + `SAVE_STEPS`/disk recomputed (§5).
- [ ] Short checkpoint trained → **EOS gate passed with a free-form eval** (§4).
- [ ] Launch the full run.

## 8. File map
- Inference teacher prompt: `src/project/socratic_teaching_system.py` (`socrates_teacher` ~400-477; system ~404-415; user ~445-453; action `get_action_for_state` / `state_to_action` ~52-88, ~396-398; `get_formatted_history` ~120-134).
- SFT loaders: `src/project/dataset.py` (`socrat-zh-sft`/`socrat-en-sft`, `_build_inference_user_message`, `_socrat_*_to_sft_records`).
- Chat template (trim proof): `scripts/train_sft.py:116-129`.
- Test gate: `tests/test_sft_inference_format.py`.
- Config: `configs/train-sft-stage2-gemma4-31b.env`; launch targets: `Makefile` (`train-gemma4-31b-stage2-unsloth`).
- Deferred design: #108 (bilingual). PR: #101. Original failure: #94.
