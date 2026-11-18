# Handoff — Gemma 4 12B SFT-uplift PoC (+ MTP speed test) on NVIDIA RTX 4000 Ada

**For:** the next Claude Code session, on the NVIDIA RTX 4000 Ada box.
**Branch to start from:** `feat/gemma4-12b-sft-poc-nvidia` (pushed to both
`ulises-c/csen-346` and `SCU-CSEN346/KELE`). Forked off
`feat/gfx1201-rdna4-qlora-fla-training`.
**Authoritative plan:** the scaffolding was built from a gated plan; the gate
ordering (G0–G9) is reproduced in §3 below.

---

## TL;DR — what this PoC does

Answer one question on the new, smaller **Gemma 4 12B**: **does 1-epoch Socratic
QLoRA SFT give measurable eval uplift?** Establish a baseline first (Qwen3.5-LoRA
state classifier as consultant + **base** 12B as teacher), then SFT, then re-eval
the same way and compare. Separately, A/B **llama.cpp MTP** (multi-token
prediction, PR #23398) on/off on the base teacher to measure inference speedup and
confirm quality is unaffected (it is lossless speculative decoding, so the quality
check should show ≈0 delta — the payoff is tokens/sec).

This is the NVIDIA sibling of the in-flight 31B AMD Stage-2 work. The 31B's
gfx1201/ROCm plumbing (`TORCH_USE_HIPBLASLT`, `PYTORCH_HIP_ALLOC_CONF`,
`gpu-preflight`, `patch-fla`) is intentionally **dropped** here; attention stays
`sdpa`. The host is **known-unstable** (see memory: training-host-hardware-fault),
so the run leans on frequent checkpoints + `train_sft.py` auto-resume.

---

## Session update — 2026-06-07 (on-box progress; reconciles the scaffold)

Picked up on the RTX 4000 Ada box. Findings the scaffold didn't anticipate, plus
corrections — read this before §3:

- **DONE:** G1 preflight ✅. G2 assets ✅ (base `…UD-Q8_K_XL.gguf` 13.6 GB + MTP drafter
  in `~/Documents/models/weights/`; GGUF repo paths confirmed — but the
  `unsloth/gemma-4-12b-it-bnb-4bit` repo the `TRAIN_PREQ=true` path implies **404s**, so
  the 12B pre-quant base is unverified; G5 may need the BF16 live-quant path instead).
- **Loadability blocker (fixed):** the 12B checkpoints are `model_type=gemma4_unified`,
  which **transformers 5.9.0 cannot load**. Bumped the lock to **5.10.2**
  (`uv lock --upgrade-package transformers`; trl→1.5.1, peft 0.19.1, bnb 0.49.2 — clean).
  The scaffold "dry-run validated" only parsed dataset+LoRA, never instantiated weights.
- **torch bumped 2.11→2.12.0+cu130** (user call; installed from the cu130 index, *not*
  in uv.lock). triton→3.7.0. **bnb 0.49.2 QLoRA smoke (Linear4bit fwd/bwd + PagedAdamW8bit)
  passes on 2.12** — QLoRA path validated.
- **Attention correction:** Gemma 4 12B text decoder is **pure softmax** (48 layers =
  40 `sliding_attention` + 8 `full_attention`; verified firsthand). **FLA is irrelevant**
  to Gemma 4 (the `cuda` extra is Qwen3.5-GDN kernels; classifier runs on CPU). The only
  lever is FA2-vs-SDPA: **stay on `sdpa`** — no prebuilt flash-attn wheel for torch2.12+cu130,
  and PyTorch `sdpa` on Ada is already FlashAttention-2-backed. `train_sft.py:227` already
  pins sdpa (that pin is the gfx1201 FA2-OOM artifact, harmless on NVIDIA). The preflight
  "FLA NOT active" WARN is genuinely ignorable here.
- **llama.cpp BUILT** (the scaffold assumed it present; it wasn't — hard prereq for ALL
  eval gates). Built from today's `main` (HEAD `9e3b928`; gemma4 + gemma4-assistant +
  `draft-mtp`/`spec-type` all confirmed in source) → `~/Documents/models/llama.cpp/build/bin/llama-server`
  (CUDA, `-DCMAKE_CUDA_ARCHITECTURES=89`; binary runs). The serve scripts auto-find this path.
  Toolchain: nvcc 13.2 (`/opt/cuda`, host gcc 15.2.1), needed only `cmake`+`ccache` (pacman).
  **ptxas SIGSEGV at `-j 24` with 90 GB RAM free → flaky silicon; built at `-j 6`.** The full
  gemma4_unified GGUF load-verify is still pending — it's the first action of G3.
- **W&B set up for BOTH sft and eval** (auth already valid: entity
  `uchavarria-santa-clara-university`). All wandb logic now lives in
  **`src/project/wandb_tracking.py`** — three classes: `WandbAuth` (shared auth),
  `SftTracker` (`.enabled`; used by `train_sft.py`, SFTTrainer does the `wandb.init`)
  and `EvalTracker` (`.log()`; used by `kele.py`). SFT (G5) auto-logs. Eval logs metrics
  as a run when **`WANDB_EVAL=1`** (already set on the `eval-…-full` make targets; smoke
  stays off) → base vs SFT compare in one dashboard. Plus **#111 Weave** call-tracing:
  env-gated `weave.init()` in `kele.py` (auto-patches the openai clients; off unless
  `WEAVE_PROJECT` set; `weave` extra already in uv.lock). To use Weave: export
  `WEAVE_PROJECT` for one base + one SFT eval (cloud-egress caveat).
- **TODO in `kele.py` (top):** rename `kele.py`→`MELE.py` + update imports across the
  board — a major architectural change, deferred. Touch only when explicitly doing it.
- **GPU is faulty under load** (power surge). Power-cap **`sudo nvidia-smi -pl 85`** before
  the GPU-heavy phases (serve/eval, train); lean on `save_steps=50` auto-resume.

New gates inserted into §3: **G1.5** (transformers→5.10.2 + torch→2.12, verify load/bnb)
and **G2.5** (build llama.cpp) — both before G3.

### → START POINTER moved: see "Session update — 2026-06-07 (cont.)" below.
G1–G2.5 done; FA2/SDPA decided (sdpa); W&B wired for sft+eval. PR **#129** (draft) tracks
this branch (eval crash tooling landed in `f638af2`). The "START AT G3 / `sudo nvidia-smi
-pl 85`" first-action that used to live here is **superseded** — the eval monitor now owns
serving, the power search, and the crash-crawl. Read the (cont.) section for the current
START pointer. Still: confirm the GGUF load-verify (G2.5) by serving once, use
**`uv run --no-sync`** for python/pytest (bare pytest isn't on PATH; torch is pinned outside
uv.lock at 2.12.0+cu130), note the unsloth 12B bnb-4bit 404 → G5 `TRAIN_PREQ` base unverified,
and pre-existing pyright warnings in `scripts/train_sft.py` are outside `include=["src"]`
and don't gate commits.

---

## Session update — 2026-06-07 (cont.) — eval tooling + refined run plan

User refined the plan and added eval‑side crash tooling (this is the NVIDIA sibling
of the 31B's training monitor). Read this before §3 — it overrides parts of it:

- **Eval is now THREE phases, not "1 full + short MTP benchmark":**
  1. **Base eval = TWO full n=681 runs — MTP OFF first, then MTP ON.** `results/
     gemma4-12b-base` and `results/gemma4-12b-base-mtp`. MTP is lossless spec‑decode,
     so accuracy *should* match; the runs confirm that and pick the **winner on
     accuracy & speed**. (Was a short tok/s benchmark — now a full A/B.)
  2. **Stage‑2b SFT** (`make train-gemma4-12b`).
  3. **SFT eval served with the MTP winner** from phase 1.
- **Eval flight recorder = issue #130** (sibling of #120). Live‑log comment id
  `4644703104` is wired into the monitor.
- **NEW `scripts/monitor_eval_gemma4_12b.sh`** (+ `make monitor-eval-gemma4-12b-{base,sft}`,
  `MTP=1` toggles the drafter & a `-mtp` output suffix). It OWNS serve+eval and
  crawls across crashes. Key difference from the training monitor, and the whole
  reason it exists: on this box a GPU fault kills the **llama.cpp server**, but the
  eval **client is CPU+HTTP so it survives** and error‑stamps the rest of the
  dataset, exiting 0 with a truncated `metrics_summary.json`. So the monitor (a)
  uses **server `/v1/models` health** as the crash signal, not process death; (b)
  **repairs** (deletes error/truncated dialogue JSONs) before each relaunch, since
  `kele.py:448` counts any non‑zero file as done; (c) defines **COMPLETE = valid
  (non‑error) dialogue count == dataset size**, not "metrics file exists".
- **Power is now an ADAPTIVE SEARCH, not a fixed `-pl 85`.** The monitor starts at
  the card's max limit and steps **−10 W per crash** (floor ~70 W) until a crawl
  completes — highest stable power wins. Safe because per‑dialogue checkpointing
  makes a fault cost only the in‑flight dialogue + a server cold‑reload. Override
  with `POWER_START_W` / `POWER_STEP_W` / `POWER_FLOOR_W`. Pre‑authorize `sudo` (it
  uses `sudo -n nvidia-smi -pl`; warns and continues if it can't).
- **SFT now auto‑pushes to HF every 50 steps** like the 31B: `train-gemma4-12b` sets
  `TRAIN_HF_REPO=ulises-c/SocratesLM-12B-QLoRA TRAIN_HF_PUSH_EVERY=50` (rename the
  repo if you want a different name). Guards the adapter against a host crash.
- **`serve_gemma4_12b_sft.sh` now honors `MTP=1`** (base‑derived drafter; lower
  acceptance on the SFT distribution but still lossless) so phase 3 can serve the winner.

### → Next instance: START AT phase 1a (base, MTP off).
Pre-flight: (1) pre-authorize `sudo` so the monitor's `sudo -n nvidia-smi -pl` power
search works (else it warns + runs at the current limit); (2) confirm the GGUF
load-verify (G2.5) by serving once; (3) run a **base smoke** (`make
eval-gemma4-12b-base-smoke`) and confirm **0 error-stamped dialogues** — that's what
makes the monitor's "any error == GPU fault" assumption sound.
Then: `make monitor-eval-gemma4-12b-base`, then `MTP=1 make monitor-eval-gemma4-12b-base`.
The monitor owns serve+eval, the power search, the crash-crawl (server-health signal,
debounced; repairs error/truncated dialogues; only steps power on a real server fault),
and issue-#130 logging. MTP off/on log as distinct W&B runs via `WANDB_EVAL_RUN_NAME`.
Still use `uv run --no-sync`. Tooling committed in `f638af2`.

---

## Project context — 5-bullet refresher

- CSEN-346 NLP project reproducing/extending **KELE** (multi-agent Socratic
  teaching; Peng et al. EMNLP 2025 Findings). A small state classifier routes
  pedagogical state to an LLM teacher.
- Eval hits a **served** OpenAI-compatible endpoint (`TEACHER_BASE_URL`), not
  in-process weights — so each model must be served (llama.cpp) before its eval.
- Metrics (`src/project/metrics.py`): ROUGE-1/2/L, BLEU-4, state-classification
  accuracy (overall + per-stage), written to `metrics_summary.json`. ROUGE/BLEU
  is diagnostic; the benchmark rewards memorisation, so weight the state-accuracy
  delta. Compare runs with `python -m src.project.evaluate --compare A B`.
- The consultant is the published **Qwen3.5-LoRA** classifier
  `ulises-c/socrates-state-classifier-qwen3.5-lora`, passed via
  `--bert-consultant <dir>` (accepts any SeqClassification checkpoint).
- Single 20 GB card: **cannot serve two models at once**, and training and serving
  compete for VRAM — do them sequentially.

---

## Decisions baked into the scaffolding

- **Eval scale:** smoke-gate → full (n=5 sanity, then full n=681 on base and SFT).
- **MTP:** base only (the drafter is base-derived → highest acceptance; the uplift
  comparison itself stays MTP-off so it isn't confounded).
- **Quant:** base served at user-chosen `gemma-4-12b-it-UD-Q8_K_XL.gguf` (13.6 GB);
  SFT served at `Q8_0` (UD is not stock-llama.cpp-producible). At Q8 the
  quant-scheme delta is ~noise, so base↔SFT differ effectively only by the adapter.

---

## What this branch already landed (don't redo)

Commits `fb82188` (scaffold) + `81686c2` (convert wrapper):

- `configs/train-sft-gemma4-12b-qlora.env` — QLoRA, 1 epoch (~4826 steps @
  eff-batch 16 over 77202 records), anchored LoRA regex (ports unchanged to the
  12B's 48 decoder layers), `socrat-zh-sft,socrat-en-sft` (inference-matching
  schema), `save_steps=50`, in-training eval off.
- `configs/gemma4-12b-local.env` / `configs/gemma4-12b-sft-local.env` — dual-role
  local-serve eval configs; distinct teacher aliases (`Gemma 4 12B` vs
  `Gemma 4 12B SFT`) so the eval fails fast against the wrong loaded weights.
- `scripts/serve_gemma4_12b.sh` — serves UD-Q8_K_XL via the generic engine
  `serve_gemma4_31b.sh` (CUDA auto-detect). `MTP=1` attaches the drafter
  (`MTP/gemma-4-12B-it-MTP-Q8_0.gguf`) with `--spec-type draft-mtp
  --spec-draft-model … --spec-draft-n-max 4` and forces f16 KV.
- `scripts/serve_gemma4_12b_sft.sh` — serves the merged SFT Q8_0 GGUF.
- `scripts/convert_gemma4_12b_sft_to_gguf.sh` — merge→GGUF with the **12B**
  NAME_TAG and auto-stage into the weights dir (no manual rename; default Q8_0).
- `Makefile` — `nvidia-preflight`, `train-gemma4-12b{,-dry-run}`,
  `serve-gemma4-12b{,-mtp,-sft}`, `eval-gemma4-12b-{base,sft}-{smoke,full}`,
  `_classifier-ckpt` (downloads the Qwen3.5 classifier on first eval).

Validated off-GPU: dry-run loads 77202/8578 records + accepts the LoRA config;
shellcheck clean; `make -n` parses all targets; venv torch is `2.11.0+cu130`
(`cuda_avail=True`).

---

## §3 — Gated run order (do these on the box)

```
G1  make nvidia-preflight                 # HARD GATE: CUDA fwd/bwd must pass
G1.5 uv lock --upgrade-package transformers   # → 5.10.2; uv sync. gemma4_unified needs ≥5.10
     uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.12.0  # optional bump
     # verify: AutoConfig.from_pretrained("unsloth/gemma-4-12b-it") loads + bnb 4-bit smoke
G2  Assets:
      hf download unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-UD-Q8_K_XL.gguf --local-dir ~/Documents/models/weights
      hf download unsloth/gemma-4-12b-it-GGUF MTP/gemma-4-12B-it-MTP-Q8_0.gguf --local-dir ~/Documents/models/weights
      # classifier auto-downloads on first eval (target: BERT_CKPT)
G2.5 Build llama.cpp (NOT installed by default — hard prereq for all eval gates):
      git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/Documents/models/llama.cpp
      cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
      cmake --build build --target llama-server -j 6    # -j 6 not 24: ptxas SIGSEGVs on flaky silicon
      # needs main ≥ PR #23398 for gemma4-assistant/MTP (today's main has it)
G3  make serve-gemma4-12b   (bg) ; poll: curl localhost:8080/v1/models  (alias "Gemma 4 12B")
    make eval-gemma4-12b-base-smoke         # SANITY GATE: non-degenerate state_accuracy; then stop server
                                            # (power-cap / crash-crawl is now the MONITOR's job, see below)
G4  make monitor-eval-gemma4-12b-base       # phase 1a: base MTP OFF → results/gemma4-12b-base   (CHECKPOINT)
    MTP=1 make monitor-eval-gemma4-12b-base # phase 1b: base MTP ON  → results/gemma4-12b-base-mtp (CHECKPOINT)
    # monitor owns serve+eval, power search (start max, -10W/crash), repair-resume, issue #130 logging.
    # pick the winner on accuracy (should be ~equal) & tok/s; that MTP setting carries to phase 3.
G5  make train-gemma4-12b                   # ~4826 steps; auto-resumes (save_steps=50) + HF push every 50
                                            # (→ ulises-c/SocratesLM-12B-QLoRA). GATE: outputs/sft-gemma4-12b-qlora/final/adapter_model.safetensors
G6  scripts/merge_lora_gemma4_sft.py --base google/gemma-4-12b-it \
      --adapter outputs/sft-gemma4-12b-qlora/final --out outputs/sft-gemma4-12b-qlora/merged
    bash scripts/convert_gemma4_12b_sft_to_gguf.sh   # → Q8_0 GGUF, auto-staged to weights dir
                                            # needs ~24 GB system RAM (BF16 staging); GPU idle
G7  make serve-gemma4-12b-sft ; make eval-gemma4-12b-sft-smoke    # SANITY GATE; then stop server
G8  [MTP=1] make monitor-eval-gemma4-12b-sft  # phase 3: SFT served with the phase-1 winner → results/gemma4-12b-sft(-mtp) (CHECKPOINT)
G9  python -m src.project.evaluate --compare results/gemma4-12b-base results/gemma4-12b-sft
    # uplift = SFT − base on state_accuracy (overall + per-stage), rouge*, bleu4
```

---

## Caveats / things to verify on the box (not code bugs)

1. **MTP needs a llama.cpp build ≥ PR #23398** (merged 2026-06-07; arch
   `gemma4-assistant`) — stock builds can't load the drafter. Also: the spec-flag
   names (`--spec-type draft-mtp / --spec-draft-model / --spec-draft-n-max`) are
   PR-sourced; verify against `llama-server --help` on the actual build — older
   llama.cpp speculative convention was `-md / --model-draft / --draft-max`.
2. **VRAM at Q8 is untested.** 13.6 GB weights + the engine's inherited `-b 2048`
   compute buffer + KV (+ draft model under MTP) on 20 GB. If serve OOMs, lower
   `-c` / `-b` / `-np` (all overridable; defaults `-c 32768`).
3. **System RAM ≥ ~32 GB** for the G6 BF16 merge.
4. **MTP-on-SFT** (if you extend beyond base): the base-derived drafter sees the
   SFT'd distribution → lower acceptance / smaller speedup, still lossless.

---

## Next-next (out of scope here)

If 1 epoch underfits, bump `TRAIN_EPOCHS` (clear `outputs/sft-gemma4-12b-qlora/
checkpoint-*` first). If uplift is real, the same pattern scales to a fuller
NVIDIA training run or a DPO stage. The 31B AMD track is independent and unchanged.
