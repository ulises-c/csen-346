# Weights & Biases export: `csen346-eval`

`csen346-eval.json` is an export of every real run in the W&B project
`uchavarria-santa-clara-university/csen346-eval`, taken on 2026-11-17. Three
debug runs (`step-debug`, `step-debug2`, `single-row-probe`) were left out.

## Why this is here

The machine that held `results/gemma4-12b-*/` has been wiped, and those result
directories were never committed. This export is the only surviving record of
the Gemma 4 12B base-versus-fine-tune evaluations besides the numbers quoted in
`docs/SFT_RESULTS_REPORT.md`.

## What it contains

For each run: id, name, URL, state, creation time, config, final summary, and
the full logged history.

The evaluation logged cumulative metrics every 10 completed dialogues
(`WANDB_EVAL_LOG_EVERY`, see `_make_incremental_metrics_logger` in
`src/project/kele.py`). Each history row therefore holds the metrics over all
dialogues finished so far:

- `eval/state_accuracy_overall` and `eval/state_acc/{a..e}`
- `eval/rouge1`, `eval/rouge2`, `eval/rougeL`, `eval/bleu4`
- `eval/n_turns` (not logged by the earliest run, `gemma4-12b-base-mtp`)

## What can and cannot be recovered

- **Per-block state accuracy can be recovered.** Differencing consecutive rows
  (`accuracy * n_turns`) gives the number of correct turns in each block of 10
  dialogues, to within about one turn, because the logged values are rounded to
  two decimals. That is enough for a block bootstrap on state accuracy.
- **Per-block ROUGE is recoverable only roughly.** The same rounding gives a
  noise of a few tenths of a point per block late in a run.
- **Blocks are in completion order, not dataset order.** Runs used parallel
  workers, so block k of one run does not hold the same dialogues as block k of
  another. Comparisons between runs from this file are unpaired.
- **No dialogue text and no per-turn data.** Nothing here can be rescored with a
  changed metric or re-judged. The ROUGE values for the English runs
  (`*-en`, `*-synth-en`) were computed character by character and are not
  meaningful; see the ROUGE fix in `src/project/metrics.py`.
- **Two runs are split across a crash and a resume**: `gemma4-12b-sft-en` and
  `gemma4-12b-sft-noconsult` each appear twice, and the resumed run continues
  the cumulative series of the crashed one. `gemma4-12b-base-synth-zh` also
  appears twice; its first attempt crashed before logging anything.
- **The paper's own runs are not here.** The Gemma 4 31B, Claude and
  SocratTeachLLM evaluations predate this logging.
