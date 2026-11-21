"""Tests for scripts/human_scoring_page.py on a small made-up worksheet (no model calls)."""

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "human_scoring_page", ROOT / "scripts" / "human_scoring_page.py"
)
assert _spec is not None and _spec.loader is not None
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)

FINGERPRINT = "3f9a1c0de4b7a2586c1d9e0f7b3a4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d"
RUNS = {
    "gemma31b": "bert-consultant-fewshot10-gemma-n50",
    "socratteachllm": "bert-socratteachllm-fewshot10-n50",
    "glm49b": "bert-glm49b-chat-fewshot10-n50",
}
JUDGES = ("judge-alpha", "judge-beta")
AXES = ("socratic_validity", "advancement", "age_appropriateness", "question_form")

# item_id, system, dialogue id, turn, stage, judge-alpha axes, judge-beta axes
_ROWS = [
    ("h001", "gemma31b", 4, 1, "b", (3, 3, 2, 1), (1, 1, 1, 1)),
    ("h002", "socratteachllm", 4, 1, "b", (2, 2, 1, 1), (3, 3, 2, 2)),
    ("h003", "glm49b", 4, 1, "b", (3, 2, 2, 1), (2, 2, 1, 1)),
    ("h004", "gemma31b", 9, 0, "a", (3, 3, 2, 1), (2, 1, 1, 1)),
    ("h005", "socratteachllm", 9, 0, "a", (1, 1, 1, 1), (3, 3, 2, 1)),
    ("h006", "glm49b", 9, 0, "a", (2, 2, 2, 1), (2, 2, 1, 1)),
    ("h007", "gemma31b", 12, 3, "d", (2, 2, 2, 2), (2, 2, 2, 2)),
]
# Human axes for h001..h006. Totals: 10, 4, 6, 8, 6, 7.
HUMAN = {
    "h001": (3, 3, 2, 2),
    "h002": (1, 1, 1, 1),
    "h003": (2, 2, 1, 1),
    "h004": (2, 2, 2, 2),
    "h005": (2, 2, 1, 1),
    "h006": (3, 2, 1, 1),
}


def make_worksheet() -> dict:
    items = []
    for item_id, system, dlg, turn, stage, ja, jb in _ROWS:
        history = []
        if turn:
            history = [
                {
                    "role": "student",
                    "en": "Why does ice float on water?",
                    "zh": "为什么冰会浮在水面上？",
                },
                {
                    "role": "teacher",
                    "en": "What do you think happens to water when it freezes?",
                    "zh": "你觉得水结冰的时候会发生什么变化呢？",
                },
            ]
        items.append(
            {
                "item_id": item_id,
                "stage": stage,
                "question_en": f"Ice floats on water because it is less dense than water. ({item_id})",
                "question_zh": "冰浮在水面上，是因为冰的密度比水小。",
                "history": history,
                "student_en": "I think ice is lighter because it is cold.",
                "student_zh": "我觉得冰比较轻，因为它很冷。",
                "reference_en": "Does a cold stone float on water too?",
                "reference_zh": "一块很冷的石头也会浮在水面上吗？",
                "reply_en": f"If we cool a spoon, will it float? What does that tell us? ({item_id})",
                "reply_zh": "如果把勺子变冷，它会浮起来吗？这说明了什么？",
                "_system": system,
                "_run": RUNS[system],
                "_dialogue_file": f"{dlg:04d}.json",
                "_dialogue_id": dlg,
                "_turn_idx": turn,
                "_judge": {
                    name: {**dict(zip(AXES, axes, strict=True)), "total": sum(axes)}
                    for name, axes in zip(JUDGES, (ja, jb), strict=True)
                },
            }
        )
    return {
        "seed": 42,
        "translator_model": "some-translator",
        "fingerprint": FINGERPRINT,
        "items": items,
    }


def make_export(scores: dict | None = None, **overrides) -> dict:
    if scores is None:
        scores = {
            item_id: {
                **dict(zip(AXES, axes, strict=True)),
                "note": "",
                "translation_unclear": False,
            }
            for item_id, axes in HUMAN.items()
        }
    return {
        "kind": "kele_human_scoring",
        "fingerprint": FINGERPRINT,
        "annotator": "tester",
        "exported_utc": "2026-11-20T00:00:00.000Z",
        "scores": scores,
        **overrides,
    }


# --------------------------------------------------------------------------- build


def _embedded(html: str, element_id: str):
    m = re.search(rf'<script id="{element_id}" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    return json.loads(m.group(1))


def test_page_has_no_private_key_or_value():
    ws = make_worksheet()
    html = hs.render(ws)
    for key in ("_system", "_run", "_dialogue_file", "_dialogue_id", "_turn_idx", "_judge"):
        assert key not in html
    for value in (*RUNS, *RUNS.values(), *JUDGES, "0004.json", "0009.json", "0012.json"):
        assert value not in html
    assert "some-translator" not in html
    # Every embedded item carries exactly the public fields.
    public = {
        "item_id",
        "stage",
        "question_en",
        "question_zh",
        "history",
        "student_en",
        "student_zh",
        "reference_en",
        "reference_zh",
        "reply_en",
        "reply_zh",
    }
    items = _embedded(html, "items")
    assert [it["item_id"] for it in items] == [it["item_id"] for it in ws["items"]]
    assert all(set(it) == public for it in items)
    assert all(set(t) == {"role", "en", "zh"} for it in items for t in it["history"])
    meta = _embedded(html, "meta")
    assert meta["fingerprint"] == FINGERPRINT
    assert meta["store_key"].endswith(FINGERPRINT)


def test_build_refuses_when_private_value_is_in_visible_text():
    ws = make_worksheet()
    ws["items"][0]["reply_en"] = "This reply was written by gemma31b."
    with pytest.raises(ValueError, match="gemma31b"):
        hs.render(ws)


def test_unknown_underscore_keys_are_also_private():
    ws = make_worksheet()
    ws["items"][0]["_sampler_note"] = "picked-from-stratum-seven"
    html = hs.render(ws)
    assert "_sampler_note" not in html and "picked-from-stratum-seven" not in html


def test_item_text_cannot_close_the_script_block():
    ws = make_worksheet()
    ws["items"][0]["reply_en"] = "</script><script>alert(1)</script><!-- x"
    html = hs.render(ws)
    assert "alert(1)</script>" not in html
    assert _embedded(html, "items")[0]["reply_en"] == ws["items"][0]["reply_en"]


def test_page_is_self_contained_and_shows_the_rubric():
    html = hs.render(make_worksheet())
    for needle in (
        "http://",
        "https://",
        "fetch(",
        "XMLHttpRequest",
        "sendBeacon",
        "<link",
        "src=",
    ):
        assert needle not in html
    assert "localStorage" in html and "human_scores." in html
    meta = _embedded(html, "meta")
    assert [a["key"] for a in meta["axes"]] == list(AXES)
    assert [a["max"] for a in meta["axes"]] == [3, 3, 2, 2]
    assert set(meta["stages"]) == set("abcde")
    assert "one valid Socratic move, but not the only valid one" in meta["reference_label"]


def test_rubric_matches_the_judge_prompt_verbatim():
    judge_source = (ROOT / "scripts" / "llm_judge_eval.py").read_text(encoding="utf-8")
    for axis in hs.AXES:
        assert f'"{axis["key"]}": 0-{axis["max"]},' in judge_source
        assert hs.rubric_line(axis) in judge_source
    for letter, stage in hs.STAGES.items():
        assert f"- {letter} ({stage['name']}): {stage['text']}" in judge_source


def test_build_command_writes_the_page(tmp_path, capsys):
    ws_path = tmp_path / "worksheet.json"
    ws_path.write_text(json.dumps(make_worksheet(), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "page" / "ui.html"
    hs.main(["build", "--worksheet", str(ws_path), "--out", str(out)])
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "7 items" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_script_parses_with_node(tmp_path):
    html = hs.render(make_worksheet())
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    path = tmp_path / "page.js"
    path.write_text(scripts[0], encoding="utf-8")
    done = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------- score


def test_fingerprint_mismatch_is_rejected(tmp_path):
    ws = make_worksheet()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        hs.score(ws, make_export(fingerprint="0" * 64))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        hs.score(ws, {k: v for k, v in make_export().items() if k != "fingerprint"})
    # The command exits with an error instead of printing numbers.
    ws_path = tmp_path / "worksheet.json"
    ws_path.write_text(json.dumps(ws), encoding="utf-8")
    bad = tmp_path / "human_scores.x.json"
    bad.write_text(json.dumps(make_export(fingerprint="abc")), encoding="utf-8")
    with pytest.raises(SystemExit, match="fingerprint mismatch"):
        hs.main(["score", "--worksheet", str(ws_path), str(bad)])


def test_invalid_exports_are_rejected():
    ws = make_worksheet()
    with pytest.raises(ValueError, match="not a kele_human_scoring export"):
        hs.score(ws, make_export(kind="something_else"))
    scores = make_export()["scores"]
    with pytest.raises(ValueError, match="not in the worksheet"):
        hs.score(ws, make_export({**scores, "h999": scores["h001"]}))
    with pytest.raises(ValueError, match="outside 0-2"):
        hs.score(ws, make_export({**scores, "h001": {**scores["h001"], "question_form": 3}}))
    with pytest.raises(ValueError, match="outside 0-3"):
        hs.score(ws, make_export({**scores, "h001": {**scores["h001"], "advancement": True}}))


def test_per_system_means_and_order():
    res = hs.score(make_worksheet(), make_export(), n_boot=200)
    assert res["n_complete"] == 6 and res["n_items_in_worksheet"] == 7
    ps = res["per_system"]
    # Human totals: gemma31b 10, 8; socratteachllm 4, 6; glm49b 6, 7.
    assert ps["gemma31b"]["n"] == 2 and ps["gemma31b"]["n_in_worksheet"] == 3
    assert ps["gemma31b"]["human_mean_total"] == pytest.approx(9.0)
    assert ps["socratteachllm"]["human_mean_total"] == pytest.approx(5.0)
    assert ps["glm49b"]["human_mean_total"] == pytest.approx(6.5)
    # judge-alpha totals: gemma31b 9, 9; socratteachllm 6, 4; glm49b 8, 7.
    alpha = {s: ps[s]["judges"]["judge-alpha"] for s in ps}
    assert alpha["gemma31b"]["judge_mean_total"] == pytest.approx(9.0)
    assert alpha["socratteachllm"]["judge_mean_total"] == pytest.approx(5.0)
    assert alpha["glm49b"]["judge_mean_total"] == pytest.approx(7.5)
    assert all(c["n"] == 2 for c in alpha.values())
    # judge-beta totals: gemma31b 4, 5; socratteachllm 10, 9; glm49b 6, 6.
    beta = {s: ps[s]["judges"]["judge-beta"] for s in ps}
    assert beta["gemma31b"]["judge_mean_total"] == pytest.approx(4.5)
    assert beta["socratteachllm"]["judge_mean_total"] == pytest.approx(9.5)
    order = res["system_order"]
    assert order["judge-alpha"]["human_order"] == ["gemma31b", "glm49b", "socratteachllm"]
    assert order["judge-alpha"]["same_order"] is True
    assert order["judge-beta"]["judge_order"] == ["socratteachllm", "glm49b", "gemma31b"]
    assert order["judge-beta"]["same_order"] is False


def test_totals_agreement_by_hand():
    res = hs.score(make_worksheet(), make_export(), n_boot=200)
    t = res["totals"]["judge-alpha"]
    # human 10, 4, 6, 8, 6, 7; judge 9, 6, 8, 9, 4, 7; judge minus human -1, 2, 2, 1, -2, 0.
    assert t["n"] == 6
    assert t["mean_abs_diff"] == pytest.approx(8 / 6)
    assert t["mean_signed_diff_judge_minus_human"] == pytest.approx(2 / 6)
    # Ranks (average for ties): human 6, 1, 2.5, 5, 2.5, 4; judge 5.5, 2, 4, 5.5, 1, 3.
    hr, jr = [6, 1, 2.5, 5, 2.5, 4], [5.5, 2, 4, 5.5, 1, 3]
    mh, mj = sum(hr) / 6, sum(jr) / 6
    cov = sum((a - mh) * (b - mj) for a, b in zip(hr, jr, strict=True))
    var = (sum((a - mh) ** 2 for a in hr) * sum((b - mj) ** 2 for b in jr)) ** 0.5
    assert t["spearman"] == pytest.approx(cov / var)


def test_partial_exports_use_only_complete_items():
    scores = make_export()["scores"]
    scores["h002"] = {**scores["h002"], "question_form": None}  # axis cleared
    del scores["h003"]["advancement"]  # axis absent
    scores["h007"] = {"note": "could not judge", "translation_unclear": True}  # nothing scored
    res = hs.score(make_worksheet(), make_export(scores), n_boot=200)
    assert res["n_items_in_export"] == 7
    assert res["n_complete"] == 4
    assert res["n_partial_or_empty_in_export"] == 3
    assert res["per_system"]["socratteachllm"]["n"] == 1
    assert res["per_system"]["socratteachllm"]["human_mean_total"] == pytest.approx(6.0)
    assert res["per_system"]["glm49b"]["human_mean_total"] == pytest.approx(7.0)
    assert res["totals"]["judge-alpha"]["n"] == 4
    assert res["per_axis"]["advancement"]["judge-beta"]["n"] == 4
    assert res["translation_unclear"]["n_flagged_in_export"] == 1
    assert res["translation_unclear"]["n_flagged_complete"] == 0
    report = hs.format_report(res)
    assert "4 of 7" in report and "(n=4)" in report


def test_empty_export_gives_no_numbers():
    res = hs.score(make_worksheet(), make_export({}), n_boot=50)
    assert res["n_complete"] == 0
    assert res["per_system"]["gemma31b"]["human_mean_total"] is None
    assert res["totals"]["judge-alpha"] == {
        "n": 0,
        "spearman": None,
        "mean_abs_diff": None,
        "mean_signed_diff_judge_minus_human": None,
    }
    assert res["system_order"]["judge-alpha"]["same_order"] is None
    assert res["bootstrap_human_total"][0]["ci95"] is None
    assert "n/a" in hs.format_report(res)


def test_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    cases = [
        ([0, 1, 2, 3, 3, 2, 1, 0, 2, 2], [0, 2, 2, 3, 2, 2, 0, 1, 3, 2], 3),
        ([0, 1, 2, 2, 1, 0, 2], [2, 1, 0, 2, 1, 1, 2], 2),
        ([3, 3, 3, 2], [3, 2, 3, 3], 3),  # categories 0 and 1 never used
        ([0, 0, 1, 1], [1, 1, 0, 0], 2),  # negative kappa
    ]
    for a, b, top in cases:
        want = cohen_kappa_score(a, b, weights="quadratic", labels=list(range(top + 1)))
        assert hs.weighted_kappa(a, b, top) == pytest.approx(want)
    assert hs.weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 3) == pytest.approx(1.0)


def test_kappa_by_hand():
    # Two categories: quadratic weights reduce to plain Cohen's kappa.
    # Observed agreement 3/4; chance agreement (2/4)(3/4) + (2/4)(1/4) = 1/2; kappa = 1/2.
    assert hs.weighted_kappa([0, 0, 1, 1], [0, 0, 0, 1], 1) == pytest.approx(0.5)


def test_kappa_is_undefined_when_both_raters_are_constant():
    # Both raters gave 2 to every item: chance disagreement is zero, kappa is 0/0.
    assert hs.weighted_kappa([2, 2, 2, 2], [2, 2, 2, 2], 2) is None
    assert hs.weighted_kappa([], [], 3) is None
    # One constant rater is defined (kappa 0), not undefined.
    assert hs.weighted_kappa([2, 2, 2, 2], [2, 1, 2, 0], 2) == pytest.approx(0.0)
    # In a full report the undefined case is None and prints as n/a, never 1.0.
    ws = make_worksheet()
    for item in ws["items"]:
        item["_judge"]["judge-alpha"]["age_appropriateness"] = 2
    scores = make_export()["scores"]
    for entry in scores.values():
        entry["age_appropriateness"] = 2
    res = hs.score(ws, make_export(scores), n_boot=50)
    cell = res["per_axis"]["age_appropriateness"]["judge-alpha"]
    assert cell["exact_agreement"] == pytest.approx(1.0)
    assert cell["quadratic_weighted_kappa"] is None
    assert "exact 100.0%, kappa n/a (n=6)" in hs.format_report(res)


def test_per_axis_exact_agreement():
    res = hs.score(make_worksheet(), make_export(), n_boot=50)
    # question_form: human 2, 1, 1, 2, 1, 1; judge-alpha 1 everywhere -> 4 of 6 equal.
    cell = res["per_axis"]["question_form"]["judge-alpha"]
    assert cell["n"] == 6
    assert cell["exact_agreement"] == pytest.approx(4 / 6)
    assert cell["quadratic_weighted_kappa"] == pytest.approx(0.0)


def test_bootstrap_resamples_dialogues():
    res = hs.score(make_worksheet(), make_export(), n_boot=2000, seed=0)
    pairs = {(b["a"], b["b"]): b for b in res["bootstrap_human_total"]}
    assert set(pairs) == {
        ("gemma31b", "glm49b"),
        ("gemma31b", "socratteachllm"),
        ("glm49b", "socratteachllm"),
    }
    b = pairs[("gemma31b", "socratteachllm")]
    # Dialogue 4: 10 - 4 = 6. Dialogue 9: 8 - 6 = 2. A resample of two dialogues gives 6, 4 or 2.
    assert b["n_dialogues"] == 2 and b["n_a"] == 2 and b["n_b"] == 2
    assert b["diff_a_minus_b"] == pytest.approx(4.0)
    assert b["ci95"] == pytest.approx([2.0, 6.0])
    again = hs.score(make_worksheet(), make_export(), n_boot=2000, seed=0)
    assert again["bootstrap_human_total"] == res["bootstrap_human_total"]


def test_bootstrap_uses_only_dialogues_with_both_systems():
    scores = make_export()["scores"]
    scores["h007"] = {**scores["h001"]}  # gemma31b, dialogue 12, which no other system has
    res = hs.score(make_worksheet(), make_export(scores), n_boot=200)
    assert res["per_system"]["gemma31b"]["n"] == 3
    b = res["bootstrap_human_total"][0]
    assert (b["a"], b["n_dialogues"], b["n_a"]) == ("gemma31b", 2, 2)


def test_flagged_items_are_counted_and_excluded():
    scores = make_export()["scores"]
    scores["h002"]["translation_unclear"] = True
    res = hs.score(make_worksheet(), make_export(scores), n_boot=50)
    tu = res["translation_unclear"]
    assert tu["n_flagged_in_export"] == 1 and tu["n_flagged_complete"] == 1
    assert tu["n_complete_without_flagged"] == 5
    without = tu["totals_without_flagged"]["judge-alpha"]
    # Without h002: judge minus human is -1, 2, 1, -2, 0.
    assert without["n"] == 5
    assert without["mean_abs_diff"] == pytest.approx(6 / 5)
    assert without["mean_signed_diff_judge_minus_human"] == pytest.approx(0.0)
    assert res["totals"]["judge-alpha"]["n"] == 6


def test_score_command_prints_report_and_writes_json(tmp_path, capsys):
    ws_path = tmp_path / "worksheet.json"
    ws_path.write_text(json.dumps(make_worksheet(), ensure_ascii=False), encoding="utf-8")
    export = tmp_path / "human_scores.tester.json"
    export.write_text(json.dumps(make_export()), encoding="utf-8")
    out = tmp_path / "agreement.json"
    hs.main(
        ["score", "--worksheet", str(ws_path), str(export), "--out", str(out), "--n-boot", "100"]
    )
    printed = capsys.readouterr().out
    for heading in ("1. Mean total", "2. Human vs judge", "3. Per axis", "4. Difference", "5. Tr"):
        assert heading in printed
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["n_complete"] == 6 and saved["fingerprint"] == FINGERPRINT


def test_page_asks_for_the_scorers_name_on_first_launch(tmp_path):
    html = (Path(__file__).parent.parent / "scripts" / "human_scoring_page.py").read_text()
    # The name dialog is created at boot and cannot be dismissed with an empty name.
    assert "function askName()" in html
    assert "askName();" in html.split("function boot()")[1]
    assert "dlg.addEventListener('cancel', e => e.preventDefault())" in html
