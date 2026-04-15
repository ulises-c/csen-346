# Test Generation Plan

This document outlines a practical testing strategy for the KELE reproduction repo. The goal is to improve confidence in the evaluation pipeline, config loading, and local serving code without making the test suite dependent on paid APIs or GPU-heavy model inference.

## Goals

- Add fast, deterministic tests for core project logic.
- Minimize dependence on external APIs, local model weights, and GPU availability.
- Separate true unit tests from smoke/integration tests.
- Protect the most important research workflow: dataset loading -> dialogue replay -> result saving -> metric computation.

## Current State

- Existing tests:
  - `tests/test_consultant.py`
- Current coverage style:
  - Live API smoke tests for consultant connectivity and JSON output.
- Main untested modules:
  - `src/project/config.py`
  - `src/project/metrics.py`
  - `src/project/kele.py`
  - `src/project/evaluate.py`
  - `src/project/serve_teacher.py`

## Test Strategy

We should organize tests into three layers:

### 1. Unit Tests

Fast tests with no network and no model loading. These should run on every local change and in CI.

### 2. Integration Tests

Small offline workflow tests that exercise multiple modules together using mocks or synthetic data.

### 3. Smoke Tests

Optional tests that hit real APIs or local model servers. These should be opt-in and skipped by default when credentials or infrastructure are unavailable.

## Priority Order

1. `metrics.py`
2. `config.py`
3. `kele.py`
4. `evaluate.py`
5. `serve_teacher.py`
6. Existing live consultant smoke tests cleanup

## Proposed Test Files

### `tests/test_metrics.py`

Purpose: verify scoring logic and edge-case handling.

Tests to add:

- `test_compute_rouge_identical_chinese_strings`
  - Identical Chinese prediction/reference pairs should score near 100.
- `test_compute_rouge_handles_multiple_examples`
  - Aggregation across multiple examples returns stable averages.
- `test_compute_bleu_identical_chinese_strings`
  - Identical Chinese prediction/reference pairs should yield a very high BLEU score.
- `test_extract_predictions_and_references_skips_error_files`
  - Dialogue files with `"error"` should be ignored.
- `test_extract_predictions_and_references_skips_empty_turns`
  - Empty teacher/reference fields should not be included.
- `test_compute_state_accuracy_overall_and_per_stage`
  - Mixed correct/incorrect states should produce expected overall and per-stage values.
- `test_compute_all_metrics_returns_error_when_no_valid_dialogues`
  - Empty or invalid dialogue directories should return an error payload.
- `test_format_metrics_table_contains_expected_sections`
  - Output string should include total turns, overlap metrics, and stage accuracy.

Implementation notes:

- Use `tmp_path` to create synthetic dialogue JSON files.
- Avoid using the real dataset in unit tests.

### `tests/test_config.py`

Purpose: verify environment loading and config precedence.

Tests to add:

- `test_load_env_file_ignores_comments_and_blank_lines`
- `test_load_env_file_strips_quotes`
- `test_load_config_reads_required_variables`
- `test_load_config_experiment_file_takes_precedence`
- `test_load_config_uses_env_file_for_missing_shared_secrets`
- `test_load_config_raises_when_required_variable_missing`
- `test_load_config_parses_debug_mode_and_max_rounds`
- `test_load_config_uses_default_teacher_local_path`

Implementation notes:

- Use `monkeypatch` to control `os.environ`.
- Use temporary `.env` files instead of the real repo `.env`.
- Keep tests isolated so they do not depend on the machine running them.

### `tests/test_kele.py`

Purpose: verify dataset splitting and single-dialogue replay behavior.

Tests to add:

- `test_load_dataset_train_test_split_is_deterministic`
- `test_load_dataset_train_and_test_do_not_overlap`
- `test_load_dataset_all_returns_full_dataset`
- `test_run_single_dialogue_records_generated_and_ground_truth_fields`
- `test_run_single_dialogue_stops_when_e34_reached`
- `test_run_single_dialogue_resets_system_before_replay`

Implementation notes:

- Mock `SocraticTeachingSystem` with a lightweight fake object.
- Use a tiny synthetic dataset item rather than the full SocratDataset.

### `tests/test_kele_batch_eval.py`

Purpose: verify batch evaluation behavior and crash-safe output writing.

Tests to add:

- `test_run_batch_evaluation_creates_expected_output_files`
- `test_run_batch_evaluation_skips_existing_dialogue_files`
- `test_run_batch_evaluation_writes_error_json_for_failed_dialogue`
- `test_run_batch_evaluation_writes_run_config`
- `test_run_batch_evaluation_computes_metrics_after_completion`

Implementation notes:

- Mock `create_system`, `load_dataset`, and `compute_all_metrics`.
- Use `tmp_path` for output directories.
- Assert on file contents, not just file existence.

### `tests/test_evaluate.py`

Purpose: verify evaluation entry points for saved run directories.

Tests to add:

- `test_evaluate_run_returns_empty_dict_when_dialogues_missing`
- `test_evaluate_run_saves_metrics_summary`
- `test_compare_runs_uses_existing_metrics_files_when_present`
- `test_compare_runs_computes_missing_metrics_when_needed`
- `test_compare_runs_writes_comparison_json`

Implementation notes:

- Build minimal fake results directories under `tmp_path`.
- Monkeypatch metric functions where appropriate.

### `tests/test_serve_teacher.py`

Purpose: verify the OpenAI-compatible HTTP surface without loading the real model.

Tests to add:

- `test_list_models_returns_expected_shape`
- `test_chat_completions_rejects_streaming`
- `test_chat_completions_returns_openai_style_response`
- `test_chat_completions_uses_manual_prompt_fallback_when_template_fails`

Implementation notes:

- Patch `tokenizer` and `model` globals with fakes.
- Use FastAPI `TestClient`.
- Do not import the real module before mocks are in place if import-time model loading is too heavy.

Note:

`serve_teacher.py` currently loads the model at import time. That makes it harder to test. A small refactor would help:

- Move model/tokenizer initialization into a function such as `load_model()`.
- Keep app creation separate from heavy initialization.
- This would make both testing and startup behavior cleaner.

## Smoke Test Cleanup

### `tests/test_consultant.py`

This file is still useful, but it should be treated as an opt-in smoke test rather than a normal unit test.

Recommended improvements:

- Mark the file or tests with `@pytest.mark.smoke` or `@pytest.mark.external`.
- Skip by default unless the user explicitly enables smoke tests.
- Rename tests or file to make the external dependency obvious.
- Keep retry logic, but tighten failure messages so API-related failures are clearly separated from logic failures.

## Suggested Pytest Markers

Add markers such as:

- `unit`
- `integration`
- `smoke`
- `external`
- `gpu`

Example usage:

- Default local run: only unit and lightweight integration tests
- Explicit smoke run: consultant/API tests
- Explicit GPU run: local teacher server tests against real weights if ever needed

## Proposed Rollout

### Phase 1

Add the highest-value, lowest-friction tests:

- `tests/test_metrics.py`
- `tests/test_config.py`
- `tests/test_kele_batch_eval.py`

Expected outcome:

- Good coverage for the evaluation pipeline without touching live APIs.

### Phase 2

Add workflow and CLI-adjacent tests:

- `tests/test_kele.py`
- `tests/test_evaluate.py`

Expected outcome:

- Safer refactors to dataset handling and run-directory processing.

### Phase 3

Improve testability of the server module, then test it:

- Refactor `serve_teacher.py` to avoid model loading at import time.
- Add `tests/test_serve_teacher.py`.

Expected outcome:

- Stable coverage for the local OpenAI-compatible teacher server.

### Phase 4

Polish smoke-test ergonomics:

- Mark live consultant tests as opt-in.
- Document how to run smoke tests locally.

Expected outcome:

- Cleaner developer experience and fewer surprising test failures.

## Nice-to-Have Additions

- Add a tiny synthetic fixture dataset under `tests/fixtures/`.
- Add snapshot-like expected JSON outputs for one mini evaluation run.
- Add coverage reporting to track progress.
- Add a CI job that runs only unit/integration tests by default.
- Add a separate CI/manual workflow for smoke tests if credentials are available.

## Definition of Done

This plan is complete when:

- Core project logic is covered by offline tests.
- The default test suite runs without API keys or GPU access.
- External smoke tests are clearly separated and documented.
- The evaluation workflow has regression protection around config, metrics, and output writing.

## Recommended First Deliverable

If we want the fastest win, start with:

1. `tests/test_metrics.py`
2. `tests/test_config.py`
3. `tests/test_kele_batch_eval.py`

That gives the best coverage-per-hour and protects the repo's most important research pipeline.
