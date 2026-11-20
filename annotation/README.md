# Human scoring of teacher replies

A person scores AI teacher replies on the same four axes as the LLM judge
(`scripts/llm_judge_eval.py`), without knowing which system wrote each reply.
The scores are then compared with the judge's. One script does both steps:
`scripts/human_scoring_page.py`.

## 1. Build the page

Input is a private worksheet, `results/human_scoring/worksheet.json`
(`results/` is gitignored). The script docstring gives its shape.

```bash
uv run python scripts/human_scoring_page.py build \
    --worksheet results/human_scoring/worksheet.json \
    --out results/human_scoring/human_scoring_ui.html
```

The output is one HTML file with no external assets. It makes no network
request and works when opened from disk (`file://`). Send that file to the
annotator; nothing else is needed.

## 2. Score in the browser

- Each item shows the stage, the topic statement, the conversation so far, the
  current student turn, a reference reply, and the AI teacher reply to score.
- The text is an English machine translation. *Show Chinese* (per item, or for
  all items in the header) puts the Chinese original next to it.
- Four button groups, one per axis, with the judge's anchor wording. An item
  counts as scored when all four are set. The note and the *Translation
  unclear* flag are optional.
- Answers are saved in the browser's localStorage, under a key that contains the
  worksheet fingerprint. They stay on that browser and that machine until
  exported. A private window or cleared site data loses them, so export often.
- *Export scores* downloads `human_scores.<annotator>.json`. It works at any
  time; the header shows how many items are not fully scored. *Import…* loads
  an earlier export of the same page and replaces the answers in the browser.
- Keys: `→`/`←` next/previous, `0`–`3` set the outlined axis, `Backspace`
  clears the last axis set, `U` first unscored item, `Z` Chinese on/off, `F`
  translation-unclear flag.

Export shape (unset axes are `null`; items with nothing entered are left out):

```json
{
  "kind": "kele_human_scoring",
  "fingerprint": "<worksheet fingerprint>",
  "annotator": "name",
  "exported_utc": "2026-11-20T18:00:00.000Z",
  "scores": {
    "h001": {
      "socratic_validity": 3, "advancement": 2, "age_appropriateness": 2,
      "question_form": 1, "note": "", "translation_unclear": false
    }
  }
}
```

## 3. Compare with the judge

```bash
uv run python scripts/human_scoring_page.py score \
    --worksheet results/human_scoring/worksheet.json \
    human_scores.NAME.json --out results/human_scoring/agreement.NAME.json
```

The export must carry the worksheet's fingerprint, or the command stops. Only
items with all four axes set are used, and n is printed beside every number:

1. Per system: mean human total and mean judge total over the same items, and
   whether the two put the systems in the same order.
2. Item totals, human against each judge: Spearman rank correlation, mean
   absolute difference, mean signed difference (judge minus human).
3. Per axis: exact agreement rate and quadratically weighted Cohen's kappa
   (n/a when both raters gave every item the same single score).
4. Difference in mean human total between each pair of systems, with a 95%
   interval from 10,000 bootstrap resamples of dialogues (fixed seed).
5. The number of items flagged *translation unclear*, and item 2 without them.

## What stays private

- The worksheet. Its underscore keys (`_system`, `_run`, `_dialogue_file`,
  `_dialogue_id`, `_turn_idx`, `_judge`) say which system wrote each reply and
  what the judge gave it. Keep it under `results/` and do not send it to the
  annotator.
- The page holds only the public fields of each item. The build copies them one
  by one, then searches the finished HTML for every private key name and every
  private string value (system, run, file and judge names) and refuses to write
  the page if one is found.
- The page and the exports contain dialogue text. Keep the exports and the
  `score` output under `results/` too. Section 1 of the output names the systems.

## Limits

- One annotator: this measures that person's agreement with the judge, not
  agreement among humans.
- The annotator reads translations. A translation error can change a score.
- Blinding hides the system name, not the writing style of a system.
