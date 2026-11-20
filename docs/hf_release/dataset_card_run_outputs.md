---
# TODO(owner): add a `license:` field before publishing. It is left out on purpose;
# see the Licence section below. Both source datasets are cc-by-4.0 on the Hub.
language:
- zh
- en
pretty_name: KELE reproduction, per-turn run outputs (November 2026 reruns)
task_categories:
- text-generation
tags:
- education
- socratic-teaching
- dialogue
- evaluation
- llm-judge
- kele
size_categories:
- 10K<n<100K
configs:
- config_name: turns
  default: true
  data_files:
  - split: train
    path: data/*/turns.jsonl
- config_name: dialogues
  data_files:
  - split: train
    path: data/*/dialogues.jsonl
dataset_info:
- config_name: turns
  features:
  - name: run
    dtype: string
  - name: dialogue_id
    dtype: int64
  - name: dialogue_file
    dtype: string
  - name: turn_idx
    dtype: int64
  - name: question
    dtype: string
  - name: student
    dtype: string
  - name: predicted_state
    dtype: string
  - name: ground_truth_state
    dtype: string
  - name: teacher_response
    dtype: string
  - name: ground_truth_teacher
    dtype: string
  - name: judge_qwen36_total
    dtype: int64
  - name: judge_qwen36_socratic_validity
    dtype: int64
  - name: judge_qwen36_advancement
    dtype: int64
  - name: judge_qwen36_age_appropriateness
    dtype: int64
  - name: judge_qwen36_question_form
    dtype: int64
  - name: judge_qwen36_comment
    dtype: string
- config_name: dialogues
  features:
  - name: run
    dtype: string
  - name: id
    dtype: int64
  - name: dialogue_file
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: num_turns_ground_truth
    dtype: int64
  - name: num_turns_generated
    dtype: int64
  - name: elapsed_seconds
    dtype: float64
---

# KELE reproduction: per-turn outputs of the November 2026 reruns

This dataset holds the outputs of 15 evaluation runs of a Socratic-teaching
pipeline, from the project "Reproducing and Extending KELE" ([code](https://github.com/ulises-c/csen-346)). In the
pipeline a *consultant* picks one of 34 teaching states for each student turn and a *teacher*
language model writes the reply to an elementary-school student. Each run pairs one consultant
with one teacher and replays the student turns of a set of reference dialogues.

It contains 43,238 generated turns from 7,185 dialogue replays: the
student turn, the state the consultant predicted, the reference state, the teacher's reply,
the reference reply, and the scores an LLM judge gave the reply. The run summaries and the
run settings are included. The dialogue text is Chinese. The judge comments are English;
some quote the Chinese text.

It is published so that the numbers in the paper can be recomputed. It is not a training set,
and several runs are included only to document a problem. Read the Caveats section before
using any number.

## Loading

```python
from datasets import load_dataset

turns = load_dataset("REPO_ID", "turns", split="train")          # one row per turn
dialogues = load_dataset("REPO_ID", "dialogues", split="train")  # one row per dialogue
one_run = turns.filter(lambda row: row["run"] == "r1b-qwen35clfzh-gemma31b-fewshot10-n681-seed42")
```

Replace `REPO_ID` with this repository's id. Every run has the same columns. `runs.csv` is
a plain CSV file and is not a config: download it with
`huggingface_hub.hf_hub_download("REPO_ID", "runs.csv", repo_type="dataset")` and read it
with any CSV reader.

The column types are declared in this card. If you read the JSONL files without the card
(for example `load_dataset("json", data_files=...)`), pass the types yourself: the first
run in file order has no per-turn judge scores, so its judge columns are all null and the
types cannot be inferred from it.

## Runs

The same table, with more columns, is in `runs.csv`.

| Run | Status | Consultant | Teacher | Set | Dialogues | Turns | State acc. (scored turns) | ROUGE-1 | BLEU-4 | Judge `qwen36` mean (complete) |
|---|---|---|---|---|---|---|---|---|---|---|
| `r1-qwen35clf-gemma31b-fewshot10-n681-seed42` | leak-documentation-only | Qwen3.5-0.8B LoRA 34-state classifier (earlier checkpoint, trained on test dialogues) | Gemma 4 31B | test-split-681 | 681 | 4134 | 54.50 | 36.98 | 9.06 | other judge (Qwen3.8-27B-Q5_K_M) |
| `r1b-qwen35clfzh-gemma31b-fewshot10-n681-seed42` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 31B | test-split-681 | 681 | 4120 | 52.14 | 36.86 | 9.01 | 8.63 (yes) |
| `r2-bgeclf-gemma31b-fewshot10-n681-seed42` | reportable | bge-small-zh 34-state classifier (clean retrain) | Gemma 4 31B | test-split-681 | 681 | 3805 | 48.36 | 35.92 | 8.34 | 8.26 (yes) |
| `r3-gemma31b-selfconsult-fewshot10-n681-seed42` | reportable | Gemma 4 31B (LLM consultant, same model as the teacher) | Gemma 4 31B | test-split-681 | 681 | 4270 | 35.34 | 35.43 | 8.15 | 8.38 (yes) |
| `r4-qwen35clfzh-socratteachllm-fewshot10-n681-seed42` | reportable-teacher-training-overlap-unknown | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | SocratTeachLLM | test-split-681 | 681 | 4056 | 55.92 | 46.54 | 20.18 | 7.20 (yes) |
| `r5-qwen35clfzh-glm49b-fewshot10-n681-seed42` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | GLM-4-9B-Chat (base) | test-split-681 | 681 | 4245 | 51.85 | 34.44 | 8.31 | 5.52 (yes) |
| `r6-qwen35clfzh-gemma12b-fewshot10-n681-seed42` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 12B | test-split-681 | 681 | 4174 | 49.52 | 35.53 | 8.31 | 8.23 (yes) |
| `r6-qwen35clfzh-gemma12bsft-fewshot10-n681-seed42` | contaminated-test-split | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 12B fine-tune (SocratesLM-12B) | test-split-681 | 681 | 4018 | 54.93 | 45.93 | 18.14 | 7.90 (yes) |
| `r6-synth75-qwen35clfzh-gemma12b-fewshot10` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 12B | synthetic-75 | 75 | 435 | not reported | 29.04 | 4.83 | 7.80 (yes) |
| `r6-synth75-qwen35clfzh-gemma12bsft-fewshot10` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 12B fine-tune (SocratesLM-12B) | synthetic-75 | 75 | 434 | not reported | 33.30 | 8.31 | 7.33 (yes) |
| `r7-synth75-qwen35clfzh-gemma31b-fewshot10` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 31B | synthetic-75 | 75 | 431 | not reported | 30.35 | 5.33 | 8.32 (yes) |
| `r7-synth75-qwen35clfzh-glm49b-fewshot10` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | GLM-4-9B-Chat (base) | synthetic-75 | 75 | 436 | not reported | 28.10 | 4.69 | 5.46 (yes) |
| `r7-synth75-qwen35clfzh-socratteachllm-fewshot10` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | SocratTeachLLM | synthetic-75 | 75 | 432 | not reported | 32.49 | 8.00 | 6.57 (yes) |
| `r8-qwen35clfzh-gemma31b-fewshot10-n681-rep2` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 31B | test-split-681 | 681 | 4124 | 52.01 | 36.79 | 8.92 | 8.63 (yes) |
| `r8-qwen35clfzh-gemma31b-fewshot10-n681-rep3` | reportable | Qwen3.5-0.8B LoRA 34-state classifier (clean retrain) | Gemma 4 31B | test-split-681 | 681 | 4124 | 51.92 | 36.72 | 8.94 | 8.67 (yes) |

Status values:

- `reportable`: a result, subject to the caveats below.
- `reportable-teacher-training-overlap-unknown`: a result, but the teacher is SocratTeachLLM on the test split,
  and its training data is not known to exclude that split.
- `contaminated-test-split`: the teacher was trained on the dialogues it is evaluated on. Not a
  clean measurement.
- `leak-documentation-only`: the consultant was trained on most of the evaluation dialogues. Kept to
  document the leak, not as a result.

Sets: `test-split-681` is the 681-dialogue test part of a 90/10 split by dialogue (seed 42) of
[ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset);
`synthetic-75` is all 75 dialogues of
[ulises-c/SocratDataset-SYNTHETIC](https://huggingface.co/datasets/ulises-c/SocratDataset-SYNTHETIC).
All runs use a 10-shot teacher prompt.

## Files

| Path | Content |
|---|---|
| `data/<run>/turns.jsonl` | One row per generated turn. |
| `data/<run>/dialogues.jsonl` | One row per dialogue. |
| `summaries/<run>/metrics_summary.json` | Turn count, ROUGE, BLEU-4 and state accuracy of the run. |
| `summaries/<run>/judge_summary*.json` | One file per judge: means, per-stage means, per-dialogue means, and whether the pass is complete. |
| `summaries/<run>/run_config.json` | Models, settings, git commit, start and end time. |
| `runs.csv` | One row per run. |

### Columns of `turns`

| Column | Meaning |
|---|---|
| `run` | Run name. |
| `dialogue_id` | Id of the dialogue in the source dataset. |
| `dialogue_file` | File name of the dialogue in the run's output folder. |
| `turn_idx` | Position of the turn in the dialogue, from 0. |
| `question` | The science question the dialogue is about. |
| `student` | The student's turn, taken from the reference dialogue. |
| `predicted_state` | The teaching state the consultant chose (`a1`, `b2` to `b7`, `c8` to `c29`, `d30` to `d33`, `e34`). |
| `ground_truth_state` | The state recorded in the source dataset for this turn. |
| `teacher_response` | The reply the teacher model wrote. |
| `ground_truth_teacher` | The reference reply in the source dataset. |
| `judge_qwen36_total` | Sum of the four axes for judge `qwen36`, 0 to 10. |
| `judge_qwen36_socratic_validity`, `judge_qwen36_advancement` | Axis scores, 0 to 3. |
| `judge_qwen36_age_appropriateness`, `judge_qwen36_question_form` | Axis scores, 0 to 2. |
| `judge_qwen36_comment` | The judge's short reason, in English. |

Judge columns are null where that judge did not score the turn. Judges present:

- `qwen36`: summary file `judge_summary.json`, per-turn scores from `judge_scores/judge_summary/`, judge model `Qwen3.6-27B-Q5_K_XL`.

### Columns of `dialogues`

| Column | Meaning |
|---|---|
| `run`, `id`, `dialogue_file` | Run name, dialogue id in the source dataset, and file name. |
| `question`, `answer` | The question and its answer as stored in the source dataset. `answer` is written as text here because the two source datasets store it in different types. |
| `num_turns_ground_truth` | Turns in the reference dialogue. |
| `num_turns_generated` | Turns the run produced. A dialogue stops when the consultant predicts the closing state `e34`, so this can be smaller. |
| `elapsed_seconds` | Wall-clock time of the dialogue. |

### Columns of `runs.csv`

| Column | Meaning |
|---|---|
| `run`, `status`, `consultant`, `teacher`, `teacher_weights_file`, `dataset`, `source_dataset` | What the run is. `consultant` and `teacher` come from `run_config.json`. |
| `n_dialogues`, `n_turns`, `n_ground_truth_turns` | Dialogues, generated turns, and turns in the reference dialogues. |
| `state_accuracy` | Percent of generated turns whose predicted state equals the reference state (from `metrics_summary.json`). |
| `state_accuracy_all_ground_truth_turns` | The same count of correct turns divided by all reference turns, so turns that were never generated count as wrong. Computed by the build script from the turn rows. |
| `state_accuracy_reportable` | `false` where the caveats below say the figure must not be reported. |
| `pct_dialogues_cut_short` | Percent of dialogues with fewer generated turns than reference turns. |
| `rouge1`, `rouge2`, `rougeL`, `bleu4` | Text overlap between the teacher's replies and the reference replies, over generated turns. |
| `judge_<name>_model`, `_mean`, `_complete`, `_turns_judged`, `_per_turn_scores` | The judge model, its mean total score, whether it scored every turn of the run, how many turns it scored, and how many per-turn scores are in `turns`. |
| `git_commit`, `git_dirty` | The code version recorded by the run. |
| `run_notes` | Serving settings recorded by the run, with host names replaced. |
| `notes` | The reasons for the status and other warnings. |

## How the numbers were produced

The code is at https://github.com/ulises-c/csen-346; `git_commit` in `runs.csv` gives the commit of each run.

- The evaluation pipeline writes one JSON file per dialogue and `metrics_summary.json`.
  ROUGE and BLEU-4 are computed on generated turns only.
- `scripts/llm_judge_eval.py` scores every teacher reply on four axes (Socratic validity 0 to
  3, advancement 0 to 3, age appropriateness 0 to 2, question form 0 to 2) and writes the
  per-turn scores and `judge_summary*.json`.
- `scripts/state_accuracy_bootstrap.py` gives confidence intervals for state accuracy, and
  `scripts/judge_bootstrap.py` gives them for the judge score and ROUGE-1. Both resample
  dialogues and both can compare two runs paired by dialogue. Intervals are not stored here.
- This folder is built from the run archives by `scripts/build_hf_run_release.py`, which
  checks that the row counts match each run's `metrics_summary.json`.

## Caveats

1. **Run `r1-…` used a classifier that had been trained on 607 of the 681 test dialogues.**
   It is included only to document that leak. Its state accuracy is not a result. Its judge
   summary was also written by a different judge model from the other runs, and its per-turn
   judge scores were not kept, so its judge columns are null. The run named `r1b-…` repeats
   it with a classifier retrained on the training part only.
2. **The Gemma 4 12B fine-tune (runs named `gemma12bsft`) was trained on turns from all 681
   test dialogues.** Its test-split run is contaminated. The 75-dialogue synthetic set is its
   only clean comparison with the base model.
3. **SocratTeachLLM's training data is not known to exclude the test split.** Treat its
   test-split scores, text-overlap scores above all, with that in mind.
4. **State accuracy on the synthetic set should not be reported.** The synthetic set's
   sub-state labels follow a different convention from SocratDataset, so the figure measures
   the labelling convention more than the consultant. The value stays in the files for
   completeness and is marked `state_accuracy_reportable = false`. The analysis is in
   `docs/SYNTHETIC_STATE_ACCURACY_NOTE.md` in the code repository.
5. **The judge is a language model.** The judge of record is Qwen3.6-27B (Q5_K_XL
   quantisation, served locally with llama.cpp); where present, Claude Sonnet 4.6 through
   Amazon Bedrock is a second judge. The judge has been checked only against deliberately
   broken control replies and against a second judge. A check against human scores is in
   progress and is not part of this release. Judge scores from different judge models are
   not comparable with each other.
6. **The runs were made on shared machines.** `run_config.json` records the serving settings
   in `env.KELE_RUN_NOTES`. Host names in that field were replaced, by exact string
   replacement and nothing else, with a description of the hardware: `rtx-a6000-node`, `rtx-4000-ada-node`, `rx-9700-node`. Some
   runs were made in one-hour job slices and resumed from disk, so
   `total_elapsed_seconds`, `started_at` and `finished_at` in their `run_config.json` cover
   the last slice only. The machines ran different llama.cpp builds, recorded in the notes.

Further limits:

- Each run is scored on the turns it generated, and a dialogue stops when the consultant
  predicts `e34`. Runs therefore cover different turns. Compare runs by dialogue, and give
  the turn count beside any turn-level figure.
- The teacher samples without a fixed temperature, and the server's sampler seed was random
  unless the notes say otherwise. The two `r8-…` runs repeat the `r1b-…` configuration to
  show run-to-run variation; they are repeats, not reproducible seeded runs.
- The reference replies and reference states come from the source datasets. According to
  the source dataset cards, both sets of dialogues were generated with language models, so
  the references are not human gold labels.

## Source data

- [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset): Chinese
  elementary-school science dialogues from the KELE paper. Licence on its Hub card:
  `cc-by-4.0`.
- [ulises-c/SocratDataset-SYNTHETIC](https://huggingface.co/datasets/ulises-c/SocratDataset-SYNTHETIC):
  75 synthetic dialogues generated with Claude Sonnet 4.6 for this project. Licence on its
  Hub card: `cc-by-4.0`.

The questions, student turns, reference replies and reference states in this dataset are
copied from those two datasets.

## Licence

**TODO (owner): choose the licence of this dataset before publishing.** It has not been
decided. This dataset copies text from the two source datasets above, both listed as
`cc-by-4.0` on the Hub, so their attribution terms apply to that text. SocratDataset was
first distributed by the KELE authors in their GitHub repository
(https://github.com/yuanpan1020/KELE), which had no licence file when checked on 2026-11-20;
the `cc-by-4.0` label was set on the Hub copy. Cite the KELE paper (Peng et al., Findings of
EMNLP 2025, https://aclanthology.org/2025.findings-emnlp.888/) when using the reference
dialogues.

## Contact

Questions and corrections: https://github.com/ulises-c/csen-346/issues
