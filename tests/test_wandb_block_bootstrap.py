"""Tests for scripts/wandb_block_bootstrap.py (pure functions only)."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "wandb_block_bootstrap",
    Path(__file__).resolve().parents[1] / "scripts" / "wandb_block_bootstrap.py",
)
assert _spec is not None and _spec.loader is not None
wbb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wbb)


def _cumulative_rows(block_correct, block_turns, start_step=10):
    rows, correct, turns = [], 0, 0
    for i, (c, t) in enumerate(zip(block_correct, block_turns)):
        correct += c
        turns += t
        rows.append(
            {
                "_step": start_step + 10 * i,
                wbb.ACC: round(correct / turns * 100, 2),
                wbb.TURNS: turns,
            }
        )
    return rows


def test_blocks_are_recovered_from_cumulative_rows():
    block_correct = [30, 41, 25, 50, 38]
    block_turns = [60, 63, 58, 66, 61]

    correct, turns = wbb.blocks_from_history(_cumulative_rows(block_correct, block_turns))

    assert turns.tolist() == block_turns
    # Two-decimal rounding of the logged accuracy can move a block by one turn.
    assert np.abs(correct - np.array(block_correct)).max() <= 1
    assert abs(correct.sum() - sum(block_correct)) <= 1


def test_merged_history_joins_a_crash_and_its_resume():
    rows = _cumulative_rows([30, 41, 25, 50], [60, 63, 58, 66])
    runs = [
        {"name": "run", "created_at": "2026-06-25T01:00:00", "history": rows[:2]},
        {"name": "run", "created_at": "2026-06-25T09:00:00", "history": rows[2:]},
        {"name": "other", "created_at": "2026-06-25T01:00:00", "history": rows},
        {"name": "run", "created_at": "2026-06-24T01:00:00", "history": []},
    ]

    merged = wbb.merged_history(runs, "run")

    assert [r["_step"] for r in merged] == [10, 20, 30, 40]
    _correct, turns = wbb.blocks_from_history(merged)
    assert turns.sum() == 60 + 63 + 58 + 66


def test_rows_without_turn_counts_are_ignored():
    runs = [{"name": "run", "created_at": "x", "history": [{"_step": 0, wbb.ACC: 50.0}]}]

    assert wbb.merged_history(runs, "run") == []
    assert wbb.summarize("run", runs, 100, np.random.default_rng(0)) is None


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    turns = np.full(68, 63.0)
    correct = rng.binomial(63, 0.55, size=68).astype(float)

    dist = wbb.bootstrap_accuracy(correct, turns, 2000, np.random.default_rng(1))
    lo, hi = np.percentile(dist, [2.5, 97.5])
    point = correct.sum() / turns.sum() * 100

    assert lo < point < hi
    assert 1.0 < hi - lo < 6.0
