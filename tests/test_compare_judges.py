"""compare_judges: dialogue-level agreement between two judges."""

import json

from scripts import compare_judges


def _run(tmp_path, name, record_totals, other_avgs):
    """record_totals: {file: [per-turn totals]}; other_avgs: {file: (n_turns_judged, avg)}."""
    run = tmp_path / name
    cache = run / "judge_scores" / "judge_summary"
    cache.mkdir(parents=True)
    for fname, totals in record_totals.items():
        scores = [{"turn_idx": i, "total": t} for i, t in enumerate(totals)]
        (cache / fname).write_text(json.dumps({"judge_model": "m", "id": 1, "scores": scores}))
    per_dialogue = [
        {"file": f, "n_turns_judged": n, "avg_total": avg} for f, (n, avg) in other_avgs.items()
    ]
    (run / "other.json").write_text(json.dumps({"per_dialogue": per_dialogue}))
    return run


def test_record_average_covers_only_the_turns_the_other_judge_scored(tmp_path):
    run = _run(tmp_path, "a", {"1.json": [10, 8, 2, 2]}, {"1.json": (2, 7.0)})

    rows = compare_judges.paired_dialogue_means(run, "other.json", "judge_summary")

    assert rows == [{"file": "1.json", "n_turns": 2, "other": 7.0, "record": 9.0}]


def test_dialogue_missing_a_record_score_is_left_out(tmp_path):
    run = _run(tmp_path, "a", {"1.json": [10]}, {"1.json": (3, 7.0), "2.json": (1, 5.0)})

    assert compare_judges.paired_dialogue_means(run, "other.json", "judge_summary") == []


def test_summary_reports_run_order_under_each_judge(tmp_path):
    good = _run(
        tmp_path,
        "good",
        {f"{i}.json": [9, 9] for i in range(4)},
        {f"{i}.json": (2, 8.0 + i / 10) for i in range(4)},
    )
    bad = _run(
        tmp_path,
        "bad",
        {f"{i}.json": [5, 5] for i in range(4)},
        {f"{i}.json": (2, 9.0) for i in range(4)},
    )
    runs = {
        r.name: compare_judges.paired_dialogue_means(r, "other.json", "judge_summary")
        for r in (good, bad)
    }

    result = compare_judges.summarize(runs)

    assert result["order_record"] == ["good", "bad"]
    assert result["order_other"] == ["bad", "good"]
    assert result["per_run"]["good"]["n_turns"] == 8
    assert result["pooled_n_dialogues"] == 8
