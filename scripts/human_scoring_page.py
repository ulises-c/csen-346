#!/usr/bin/env python3
"""Human scoring of teacher replies: a blind page, then agreement with the LLM judge.

``build`` turns a private worksheet into one self-contained HTML page. A person
scores each AI teacher reply on the four axes of the LLM judge
(scripts/llm_judge_eval.py) without knowing which system wrote it. English
machine translations are shown; the Chinese original is behind a toggle. The
page needs no server and makes no network request: answers are kept in the
browser's localStorage under the worksheet fingerprint, and *Export* downloads
``human_scores.<annotator>.json``.

``score`` compares an export with the judge scores stored in the worksheet.

Worksheet (private, under gitignored results/):
    {"seed": 42, "translator_model": "...", "fingerprint": "<hex>", "items": [ITEM, ...]}
Each ITEM has public fields (item_id, stage, question_*, history, student_*,
reference_*, reply_*) and private fields whose key starts with an underscore
(_system, _run, _dialogue_file, _dialogue_id, _turn_idx, _judge). Only the
public fields are copied into the page, field by field. After rendering, the
build checks that no private key name and no private string value occurs
anywhere in the HTML, and refuses to write the page otherwise.

What ``score`` reports (only items with all four axes set are used; n is given
beside every number):

1. Per system: mean human total and mean total of each judge over the same
   items, and whether human and judge put the systems in the same order.
2. Human against each judge on item totals: Spearman rank correlation, mean
   absolute difference, mean signed difference (judge minus human).
3. Per axis: exact agreement rate and quadratically weighted Cohen's kappa.
   Kappa is undefined (reported as n/a) when both raters gave every item the
   same single score, because chance disagreement is then zero.
4. For each pair of systems, the difference in mean human total with a 95%
   percentile interval from a bootstrap that resamples dialogues
   (``_dialogue_id``), not single items. Only dialogues with at least one scored
   item of both systems are used, and each resample draws the same dialogues
   for both. Means are item-weighted inside each resample.
5. How many items were flagged "translation unclear", and section 2 again
   without them.

Limits: one annotator, so this measures agreement of that person with the
judge, not human consensus. The annotator reads translations, so translation
errors can lower agreement; section 5 gives a rough check. The bootstrap
interval covers the choice of dialogues only.

Usage:
    uv run python scripts/human_scoring_page.py build \\
        --worksheet results/human_scoring/worksheet.json \\
        --out results/human_scoring/human_scoring_ui.html
    uv run python scripts/human_scoring_page.py score \\
        --worksheet results/human_scoring/worksheet.json \\
        human_scores.NAME.json --out results/human_scoring/agreement.NAME.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

KIND = "kele_human_scoring"
STORE_PREFIX = "kele-human-scoring-v1:"

# The judge's rubric, verbatim from JUDGE_SYSTEM_PROMPT in scripts/llm_judge_eval.py
# (tests/test_human_scoring_page.py checks that the two stay identical).
AXES: list[dict] = [
    {
        "key": "socratic_validity",
        "label": "Socratic validity",
        "max": 3,
        "question": "Is this a valid Socratic teaching move at the current stage?",
        "anchors": {"3": "textbook", "2": "acceptable", "1": "marginal", "0": "wrong stage/move"},
    },
    {
        "key": "advancement",
        "label": "Advancement",
        "max": 3,
        "question": "Does this response advance the student's reasoning toward the answer?",
        "anchors": {
            "3": "clearly advances",
            "2": "neutral but valid",
            "1": "stalls",
            "0": "confuses/regresses",
        },
    },
    {
        "key": "age_appropriateness",
        "label": "Age appropriateness",
        "max": 2,
        "question": "Is vocabulary and complexity right for elementary school?",
        "anchors": {"2": "well-calibrated", "1": "borderline", "0": "too complex or too simple"},
    },
    {
        "key": "question_form",
        "label": "Question form",
        "max": 2,
        "question": "Is the response a single Socratic question?",
        "anchors": {
            "2": "clean single question",
            "1": "multiple questions or hint",
            "0": "statement only or over-leading",
        },
    },
]
AXIS_KEYS = [a["key"] for a in AXES]
AXIS_MAX = {a["key"]: a["max"] for a in AXES}
STAGES: dict[str, dict[str, str]] = {
    "a": {"name": "questioning", "text": "student poses a question"},
    "b": {"name": "concept probing", "text": "probe student concepts"},
    "c": {"name": "inductive reasoning", "text": "help student reason through misconceptions"},
    "d": {"name": "rule construction", "text": "help student build correct rule and apply it"},
    "e": {"name": "closure", "text": "summarize and confirm understanding"},
}
REFERENCE_LABEL = "Reference response (one valid Socratic move, but not the only valid one)"
N_BOOT = 10_000
BOOT_SEED = 0


def rubric_line(axis: dict) -> str:
    """One axis as the judge prompt writes it: ``<question> 3=..., 2=..., ...``."""
    anchors = ", ".join(f"{k}={axis['anchors'][str(k)]}" for k in range(axis["max"], -1, -1))
    return f"{axis['question']} {anchors}"


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human scoring — Socratic teacher replies</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f4f6fb; --panel: #ffffff; --panel-soft: #f8f9fc; --ink: #182034;
  --muted: #667085; --faint: #98a2b3; --line: #e4e7ec; --line-strong: #d0d5dd;
  --accent: #4f46e5; --accent-strong: #3730a3; --accent-bg: #eef2ff;
  --solid-ink: #ffffff;
  --warn: #c2415d; --ok: #087f5b; --warn-bg: #fff1f3; --ok-bg: #ecfdf3;
  --target: #b45309; --target-bg: #fffbeb;
  --shadow-sm: 0 1px 2px rgb(16 24 40 / 0.05);
  --shadow-lg: 0 16px 40px rgb(16 24 40 / 0.08);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #10131c; --panel: #181c27; --panel-soft: #1e2330; --ink: #f2f4f7;
    --muted: #a6adbb; --faint: #7d8597; --line: #2d3444; --line-strong: #414a5e;
    --accent: #9b98ff; --accent-strong: #c4c2ff; --accent-bg: #26274a;
    --solid-ink: #10131c;
    --warn: #fda4af; --ok: #6ce9b4; --warn-bg: #44232b; --ok-bg: #153a2d;
    --target: #fbbf24; --target-bg: #3a2f14;
    --shadow-sm: 0 1px 2px rgb(0 0 0 / 0.25);
    --shadow-lg: 0 18px 48px rgb(0 0 0 / 0.28);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.6 Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
header {
  position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--panel) 94%, transparent);
  border-bottom: 1px solid var(--line); box-shadow: var(--shadow-sm); backdrop-filter: blur(14px);
}
.wrap { max-width: 980px; margin: 0 auto; padding: 0 24px; }
.bar { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; min-height: 68px; }
.brand { display: flex; align-items: center; gap: 11px; }
.brand-mark {
  display: grid; place-items: center; width: 34px; height: 34px; border-radius: 10px;
  background: var(--accent); color: var(--solid-ink); font-size: 12px; font-weight: 750; letter-spacing: 0.04em;
  box-shadow: 0 6px 16px color-mix(in srgb, var(--accent) 25%, transparent);
}
.brand-copy { display: flex; flex-direction: column; line-height: 1.15; }
h1 { font-size: 15px; margin: 0; font-weight: 700; letter-spacing: -0.01em; }
.eyebrow { color: var(--muted); font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.spacer { flex: 1; }
.who { display: flex; align-items: center; gap: 8px; }
.header-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
input[type=text] {
  font: inherit; padding: 8px 11px; border: 1px solid var(--line-strong); border-radius: 9px;
  background: var(--panel-soft); color: var(--ink); width: 156px;
}
input[type=text]:focus, textarea:focus { outline: 3px solid color-mix(in srgb, var(--accent) 18%, transparent); border-color: var(--accent); }
button {
  font: inherit; min-height: 36px; padding: 7px 12px; border: 1px solid var(--line-strong); border-radius: 9px;
  background: var(--panel); color: var(--ink); cursor: pointer; transition: 120ms ease;
}
button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent-strong); background: var(--accent-bg); }
button:focus-visible, summary:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--accent) 24%, transparent); outline-offset: 2px;
}
button:disabled { opacity: 0.4; cursor: not-allowed; }
button[aria-pressed=true] { background: var(--accent-bg); border-color: var(--accent); color: var(--accent-strong); }
.progress-panel { padding: 12px 0 15px; border-top: 1px solid var(--line); }
.progress-copy { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.progress-title { font-size: 12px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.progress { height: 6px; background: var(--line); border-radius: 99px; overflow: hidden; margin-top: 9px; }
.progress > div { height: 100%; border-radius: inherit; background: var(--accent); width: 0; transition: width 0.25s ease; }
.tally { font-size: 13px; color: var(--muted); }
.tally strong { color: var(--ink); }
.missing { font-size: 13px; color: var(--warn); font-weight: 650; }
.missing.none { color: var(--ok); }
.save-state { margin-left: auto; display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }
.save-state::before { content: ""; width: 7px; height: 7px; border-radius: 99px; background: var(--ok); flex: none; }
.save-state.error::before { background: var(--warn); }
.dots { display: grid; grid-template-columns: repeat(30, minmax(4px, 1fr)); gap: 4px; margin-top: 10px; }
.dot { min-height: 9px; height: 9px; border-radius: 99px; background: var(--line); border: 0; padding: 0; cursor: pointer; }
.dot:hover { outline: 2px solid var(--accent); background: var(--accent-bg); }
.dot.partial { background: color-mix(in srgb, var(--accent) 40%, var(--line)); }
.dot.done { background: var(--accent); }
.dot.flag { box-shadow: inset 0 0 0 2px var(--warn); }
.dot.at { outline: 2px solid var(--accent); outline-offset: 2px; }
main { max-width: 980px; margin: 0 auto; padding: 28px 24px 0; }
details.guide {
  background: var(--accent-bg); border: 1px solid color-mix(in srgb, var(--accent) 20%, var(--line));
  border-radius: 14px; padding: 14px 18px; margin-bottom: 20px;
}
details.guide summary { cursor: pointer; font-weight: 700; color: var(--accent-strong); }
.guide-hint { margin-left: 8px; color: var(--muted); font-size: 13px; font-weight: 450; }
details.guide ul { margin: 8px 0 0; padding-left: 20px; }
details.guide li { margin-bottom: 6px; }
details.guide h3 { font-size: 14px; margin: 18px 0 6px; }
.guide-start { display: flex; justify-content: flex-end; margin-top: 14px; }
.guide-start button { background: var(--accent); border-color: var(--accent); color: var(--solid-ink); font-weight: 700; }
.guide-start button:hover:not(:disabled) { background: var(--accent-strong); color: var(--solid-ink); }
table.rubric { width: 100%; border-collapse: collapse; font-size: 13.5px; background: var(--panel); border-radius: 10px; overflow: hidden; }
table.rubric th, table.rubric td { text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid var(--line); }
table.rubric th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
table.rubric code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
kbd {
  font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--panel);
  border: 1px solid var(--line-strong); border-bottom-width: 2px; border-radius: 5px; padding: 2px 6px;
}
.card {
  background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--line);
  border-radius: 18px; padding: 24px; box-shadow: var(--shadow-lg);
}
.card.done { border-top-color: var(--accent); }
.card-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding-bottom: 4px; }
.idx {
  color: var(--muted); font-size: 12px; font-weight: 650; font-variant-numeric: tabular-nums;
  background: var(--panel-soft); border: 1px solid var(--line); border-radius: 99px; padding: 4px 9px;
}
.stage {
  font-size: 13px; background: var(--accent-bg); color: var(--accent-strong);
  border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--line)); border-radius: 99px; padding: 4px 12px;
}
.stage strong { font-weight: 750; }
.case-status { margin-left: auto; font-size: 12px; font-weight: 700; color: var(--muted); }
.case-status.done { color: var(--accent-strong); }
.zh-toggle { min-height: 30px; padding: 3px 10px; font-size: 13px; }
.q { margin-top: 20px; }
.q-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); font-weight: 750; margin-bottom: 8px; }
.q-sub { text-transform: none; letter-spacing: 0; font-weight: 500; }
.block {
  background: var(--panel-soft); border: 1px solid var(--line); border-left: 3px solid var(--line-strong);
  border-radius: 12px; padding: 12px 16px;
}
.block + .block { margin-top: 8px; }
.block.student { border-left-color: var(--accent); margin-right: 12%; }
.block.teacher { border-left-color: var(--ok); margin-left: 12%; }
.block.reference { border-style: dashed; border-left-style: solid; }
.block.target {
  background: var(--target-bg); border: 2px solid var(--target); border-left-width: 6px;
  padding: 16px 18px; font-size: 16px;
}
.target-label { color: var(--target); }
.who-tag { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--faint); margin-bottom: 2px; }
.block.student .who-tag { color: var(--accent-strong); }
.block.teacher .who-tag { color: var(--ok); }
.pair { display: grid; grid-template-columns: 1fr; gap: 14px; }
.pair.zh-on { grid-template-columns: 1fr 1fr; }
.txt { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }
.txt.zh { color: var(--muted); border-left: 1px dashed var(--line-strong); padding-left: 14px; }
.empty { color: var(--faint); font-style: italic; }
.score-section { border-top: 1px solid var(--line-strong); padding-top: 18px; margin-top: 24px; }
.axis {
  margin-top: 12px; padding: 12px 14px; border: 1px solid var(--line-strong);
  border-left: 4px solid var(--line-strong); border-radius: 10px; background: var(--panel-soft);
}
.axis.set { border-left-color: var(--accent); }
.axis.active { outline: 2px solid color-mix(in srgb, var(--accent) 45%, transparent); outline-offset: 1px; }
.axis-head { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.axis-name { font-weight: 750; font-size: 14px; }
.axis-range { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.axis-q { font-size: 13px; color: var(--muted); margin: 2px 0 9px; }
.choices { display: flex; gap: 8px; flex-wrap: wrap; }
.choice { flex: 1; min-width: 120px; text-align: left; padding: 8px 12px; line-height: 1.3; }
.choice .num { font-weight: 750; font-size: 16px; margin-right: 8px; font-variant-numeric: tabular-nums; }
.choice .anchor { font-size: 13px; }
.choice[aria-pressed=true] { font-weight: 680; box-shadow: inset 0 0 0 1px var(--accent); }
.total-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
.total { font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; }
.total .of { color: var(--muted); font-weight: 500; }
.needs { font-size: 13px; font-weight: 650; color: var(--warn); }
.flag[aria-pressed=true] { background: var(--warn-bg); border-color: var(--warn); color: var(--warn); }
textarea.note-box {
  display: block; width: 100%; min-height: 56px; font: inherit; padding: 8px 10px; background: var(--panel-soft);
  color: var(--ink); border: 1px solid var(--line-strong); border-radius: 8px; resize: vertical;
}
.nav {
  display: flex; gap: 10px; align-items: center; margin-top: 16px; padding: 12px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow-sm);
}
.nav .where { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
footer { max-width: 980px; margin: 0 auto; padding: 24px 24px 60px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.note { color: var(--muted); font-size: 13px; }
.done-banner {
  background: var(--ok-bg); border: 1px solid var(--ok); color: var(--ok);
  border-radius: 12px; padding: 13px 16px; margin-bottom: 18px; font-weight: 650;
}
.resume {
  display: flex; align-items: center; gap: 8px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 12px; padding: 10px 14px; margin-bottom: 14px; font-size: 13px; color: var(--muted);
}
@media (max-width: 720px) {
  .wrap { padding: 0 16px; }
  main { padding: 18px 12px 0; }
  .card { padding: 18px 14px; }
  .pair.zh-on { grid-template-columns: 1fr; }
  .block.student, .block.teacher { margin-left: 0; margin-right: 0; }
}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="bar">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">HS</span>
        <span class="brand-copy">
          <span class="eyebrow">Blind human scoring</span>
          <h1>Socratic teacher replies · four axes</h1>
        </span>
      </div>
      <div class="spacer"></div>
      <div class="who">
        <label for="annotator" class="note">Annotator</label>
        <input type="text" id="annotator" placeholder="Your name" autocomplete="off" spellcheck="false">
      </div>
      <div class="header-actions">
        <button id="zh-all" type="button" aria-pressed="false">Show Chinese (all)</button>
        <button id="import" type="button">Import…</button>
        <button id="export" type="button">Export scores</button>
      </div>
    </div>
    <div class="progress-panel">
      <div class="progress-copy">
        <span class="progress-title">Progress</span>
        <span class="tally" id="tally" aria-live="polite"></span>
        <span class="missing" id="missing" aria-live="polite"></span>
        <span class="save-state" id="save-state" aria-live="polite"></span>
      </div>
      <div class="progress" id="progress" role="progressbar" aria-label="Items scored" aria-valuemin="0">
        <div id="bar"></div>
      </div>
      <div class="dots" id="dots" aria-label="Item navigator"></div>
    </div>
  </div>
</header>
<main>
  <details class="guide" id="guide">
    <summary>Scoring guide and rubric
      <span class="guide-hint">Read this before you begin; it stays available here</span></summary>
    <p class="note">
      An AI teacher gives Socratic replies to an elementary-school student. Each item shows the topic statement,
      the conversation so far, the student's current turn, a reference reply, and one AI teacher reply.
      Score the AI teacher reply on the four axes below. The page does not say which system wrote a reply.
      The text is a machine translation of the Chinese original; use <em>Show Chinese</em> to see the original
      next to it, and mark <em>Translation unclear</em> when the English cannot be judged with confidence.
    </p>
    <h3>Rubric (the same wording the LLM judge receives)</h3>
    <table class="rubric" id="rubric"></table>
    <p class="note">Total is the sum of the four axes, 0–10.</p>
    <h3>The five stages</h3>
    <ul id="stages"></ul>
    <h3>How the page works</h3>
    <ul>
      <li>The reference reply is one valid Socratic move, but not the only valid one. A reply that differs from it
        can still earn full marks.</li>
      <li>An item counts as scored when all four axes are set. Click a selected score again to clear it.
        The note is optional.</li>
      <li>Every change is saved in this browser at once. <em>Export scores</em> downloads a JSON file and works
        at any time, also when some items are not scored; the header shows how many are missing.
        <em>Import…</em> loads an earlier export of this same page and replaces the answers in the browser.</li>
      <li>Keys: <kbd>→</kbd>/<kbd>←</kbd> next / previous item; <kbd>0</kbd>–<kbd>3</kbd> set the outlined axis
        (the first one without a score); <kbd>Backspace</kbd> clears the last axis set; <kbd>U</kbd> first
        unscored item; <kbd>Z</kbd> Chinese on/off for this item; <kbd>F</kbd> translation-unclear flag.
        Keys are off while typing in a text box (<kbd>Esc</kbd> leaves the box).</li>
    </ul>
    <div class="guide-start">
      <button id="start" type="button">Close the guide and start scoring</button>
    </div>
  </details>
  <div id="review">
    <div id="banner"></div>
    <div id="card"></div>
    <div class="nav">
      <button id="prev" type="button">← Previous</button>
      <button id="next" type="button">Next →</button>
      <span class="where" id="where"></span>
      <span class="spacer"></span>
      <button id="skip" type="button">First unscored</button>
    </div>
  </div>
</main>
<footer id="foot"></footer>
<input type="file" id="file" accept="application/json,.json" hidden>
<script id="items" type="application/json">__ITEMS_JSON__</script>
<script id="meta" type="application/json">__META_JSON__</script>
<script>
'use strict';
const ITEMS = JSON.parse(document.getElementById('items').textContent);
const META = JSON.parse(document.getElementById('meta').textContent);
const AXES = META.axes;
const STAGES = META.stages;
const STORE = META.store_key;
const MAX_TOTAL = AXES.reduce((s, a) => s + a.max, 0);
const INDEX = {};
ITEMS.forEach((it, i) => { INDEX[it.item_id] = i; });

let state = load();
let current = 0;
let storageOk = true;
const zhOverride = {};  // per-item display choice; not saved, not exported

function emptyState() {
  return { kind: META.kind, fingerprint: META.fingerprint, annotator: '', scores: {}, position: 0,
           show_zh: false, guide_read: false };
}

// One saved or imported entry, reduced to the known fields; null when a value is out of range.
function cleanEntry(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const out = { note: '', translation_unclear: false };
  for (const a of AXES) {
    const v = raw[a.key];
    if (v === null || v === undefined) { out[a.key] = null; continue; }
    if (!Number.isInteger(v) || v < 0 || v > a.max) return null;
    out[a.key] = v;
  }
  if (raw.note !== undefined && raw.note !== null) {
    if (typeof raw.note !== 'string') return null;
    out.note = raw.note;
  }
  if (raw.translation_unclear !== undefined && raw.translation_unclear !== null) {
    if (typeof raw.translation_unclear !== 'boolean') return null;
    out.translation_unclear = raw.translation_unclear;
  }
  return out;
}

// All entries of a saved or imported score map; null when any entry is unknown or invalid.
function cleanScores(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const out = {};
  for (const [id, entry] of Object.entries(raw)) {
    if (!(id in INDEX)) return null;
    const clean = cleanEntry(entry);
    if (!clean) return null;
    out[id] = clean;
  }
  return out;
}

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
    if (saved && saved.kind === META.kind && saved.fingerprint === META.fingerprint) {
      const scores = cleanScores(saved.scores);
      if (scores) {
        return { ...emptyState(), annotator: typeof saved.annotator === 'string' ? saved.annotator : '',
                 scores, position: Number.isInteger(saved.position) ? saved.position : 0,
                 show_zh: !!saved.show_zh, guide_read: !!saved.guide_read, saved_utc: saved.saved_utc };
      }
    }
  } catch (e) { /* absent, corrupt or blocked storage */ }
  return emptyState();
}

function save() {
  state.saved_utc = new Date().toISOString();
  try {
    localStorage.setItem(STORE, JSON.stringify(state));
    storageOk = true;
    setSaveState('', 'Saved in this browser ' + new Date().toLocaleTimeString() + ' — export before you finish');
  } catch (e) {
    storageOk = false;
    setSaveState('error', 'This browser refused to store the answers — export now to keep them');
  }
}

function setSaveState(kind, text) {
  const box = document.getElementById('save-state');
  box.className = 'save-state' + (kind ? ' ' + kind : '');
  box.textContent = text;
}

function slugify(name) {
  return (name || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'unnamed';
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function entry(i) { return state.scores[ITEMS[i].item_id] || null; }
function ensure(i) {
  const id = ITEMS[i].item_id;
  if (!state.scores[id]) {
    const fresh = { note: '', translation_unclear: false };
    AXES.forEach(a => { fresh[a.key] = null; });
    state.scores[id] = fresh;
  }
  return state.scores[id];
}
function axesSet(i) {
  const e = entry(i);
  return e ? AXES.filter(a => Number.isInteger(e[a.key])).length : 0;
}
function isDone(i) { return axesSet(i) === AXES.length; }
function hasContent(e) {
  return !!e && (AXES.some(a => Number.isInteger(e[a.key])) || !!e.note || !!e.translation_unclear);
}
function total(i) {
  const e = entry(i);
  return e ? AXES.reduce((s, a) => s + (Number.isInteger(e[a.key]) ? e[a.key] : 0), 0) : 0;
}
function firstUnscored() {
  for (let i = 0; i < ITEMS.length; i++) if (!isDone(i)) return i;
  return -1;
}
function zhShown(i) {
  const id = ITEMS[i].item_id;
  return id in zhOverride ? zhOverride[id] : !!state.show_zh;
}

function textPair(en, zh, showZh) {
  const pair = el('div', 'pair' + (showZh ? ' zh-on' : ''));
  const enNode = el('div', 'txt en' + (en ? '' : ' empty'), en || '(empty)');
  enNode.lang = 'en';
  pair.append(enNode);
  if (showZh) {
    const zhNode = el('div', 'txt zh' + (zh ? '' : ' empty'), zh || '(empty)');
    zhNode.lang = 'zh';
    pair.append(zhNode);
  }
  return pair;
}

function section(label, sub) {
  const box = el('div', 'q');
  const head = el('div', 'q-label', label);
  if (sub) head.append(' ', el('span', 'q-sub', sub));
  box.append(head);
  return box;
}

function renderCard() {
  const i = current;
  const item = ITEMS[i];
  const e = entry(i);
  const showZh = zhShown(i);
  const done = isDone(i);
  const card = el('div', 'card' + (done ? ' done' : ''));

  const head = el('div', 'card-head');
  head.append(el('span', 'idx', 'Item ' + (i + 1) + ' of ' + ITEMS.length));
  const st = STAGES[item.stage];
  const stage = el('span', 'stage');
  stage.append(el('strong', null, 'Stage ' + item.stage));
  stage.append(st ? ' (' + st.name + '): ' + st.text : ' (not one of the five stages)');
  head.append(stage);
  const zhBtn = el('button', 'zh-toggle', showZh ? 'Hide Chinese' : 'Show Chinese');
  zhBtn.type = 'button';
  zhBtn.setAttribute('aria-pressed', String(showZh));
  zhBtn.addEventListener('click', toggleZh);
  head.append(zhBtn);
  const set = axesSet(i);
  head.append(el('span', 'case-status' + (done ? ' done' : ''),
    done ? 'Scored' : set ? set + ' of ' + AXES.length + ' axes set' : 'Not scored'));
  card.append(head);

  const topic = section('Topic statement');
  const topicBlock = el('div', 'block');
  topicBlock.append(textPair(item.question_en, item.question_zh, showZh));
  topic.append(topicBlock);
  card.append(topic);

  const hist = section('Conversation so far');
  if (!item.history.length) {
    hist.append(el('div', 'note', 'Start of the dialogue: there are no earlier turns.'));
  }
  item.history.forEach(turn => {
    const teacher = turn.role === 'teacher';
    const block = el('div', 'block ' + (teacher ? 'teacher' : 'student'));
    block.append(el('div', 'who-tag', teacher ? 'Teacher' : 'Student'));
    block.append(textPair(turn.en, turn.zh, showZh));
    hist.append(block);
  });
  card.append(hist);

  const stu = section('Current student turn');
  const stuBlock = el('div', 'block student');
  stuBlock.append(el('div', 'who-tag', 'Student'));
  stuBlock.append(textPair(item.student_en, item.student_zh, showZh));
  stu.append(stuBlock);
  card.append(stu);

  const ref = section(META.reference_label);
  const refBlock = el('div', 'block reference');
  refBlock.append(textPair(item.reference_en, item.reference_zh, showZh));
  ref.append(refBlock);
  card.append(ref);

  const tgt = section('AI teacher reply to score');
  tgt.firstChild.classList.add('target-label');
  const tgtBlock = el('div', 'block target');
  tgtBlock.append(textPair(item.reply_en, item.reply_zh, showZh));
  tgt.append(tgtBlock);
  card.append(tgt);

  const scoreBox = el('div', 'score-section');
  scoreBox.append(el('div', 'q-label', 'Your scores for the AI teacher reply'));
  const active = AXES.find(a => !e || !Number.isInteger(e[a.key]));
  AXES.forEach(a => {
    const value = e ? e[a.key] : null;
    const isSet = Number.isInteger(value);
    const box = el('div', 'axis' + (isSet ? ' set' : '') + (active && active.key === a.key ? ' active' : ''));
    const ah = el('div', 'axis-head');
    ah.append(el('span', 'axis-name', a.label), el('span', 'axis-range', '0–' + a.max));
    box.append(ah);
    box.append(el('div', 'axis-q', a.question));
    const row = el('div', 'choices');
    row.setAttribute('role', 'group');
    row.setAttribute('aria-label', a.label);
    for (let v = a.max; v >= 0; v--) {
      const b = el('button', 'choice');
      b.type = 'button';
      b.dataset.axis = a.key;
      b.dataset.value = String(v);
      b.title = a.label + ' ' + v + ': ' + a.anchors[String(v)];
      b.setAttribute('aria-pressed', String(value === v));
      b.append(el('span', 'num', String(v)), el('span', 'anchor', a.anchors[String(v)]));
      b.addEventListener('click', () => setAxis(a.key, v));
      row.append(b);
    }
    box.append(row);
    scoreBox.append(box);
  });

  const totalRow = el('div', 'total-row');
  const tot = el('span', 'total');
  tot.id = 'total';
  if (set) {
    tot.append('Total: ' + total(i) + ' ', el('span', 'of', '/ ' + MAX_TOTAL));
  } else {
    tot.append('Total: — ', el('span', 'of', '/ ' + MAX_TOTAL));
  }
  totalRow.append(tot);
  if (!done) {
    const left = AXES.filter(a => !e || !Number.isInteger(e[a.key])).map(a => a.label.toLowerCase());
    totalRow.append(el('span', 'needs', (set ? 'So far. ' : '') + 'Still to set: ' + left.join(', ')));
  }
  totalRow.append(el('span', 'spacer'));
  const flag = el('button', 'flag', 'Translation unclear');
  flag.type = 'button';
  flag.id = 'flag';
  flag.setAttribute('aria-pressed', String(!!(e && e.translation_unclear)));
  flag.addEventListener('click', toggleFlag);
  totalRow.append(flag);
  scoreBox.append(totalRow);

  const noteBox = section('Note', '(optional)');
  const ta = el('textarea', 'note-box');
  ta.id = 'note';
  ta.placeholder = 'Anything worth recording about this reply or its translation';
  ta.value = e ? e.note : '';
  ta.addEventListener('input', () => {
    if (!ta.value && !entry(current)) return;
    ensure(current).note = ta.value;
    save();
    tally();
  });
  noteBox.append(ta);
  scoreBox.append(noteBox);
  card.append(scoreBox);

  document.getElementById('card').replaceChildren(card);
  document.getElementById('where').textContent = (i + 1) + ' / ' + ITEMS.length;
  document.getElementById('prev').disabled = i === 0;
  document.getElementById('next').disabled = i === ITEMS.length - 1;
}

function setAxis(key, value) {
  const e = ensure(current);
  e[key] = e[key] === value ? null : value;
  save(); renderCard(); tally();
}
function toggleFlag() {
  const e = ensure(current);
  e.translation_unclear = !e.translation_unclear;
  save(); renderCard(); tally();
}
function toggleZh() {
  zhOverride[ITEMS[current].item_id] = !zhShown(current);
  renderCard();
}
function toggleZhAll() {
  state.show_zh = !state.show_zh;
  Object.keys(zhOverride).forEach(id => delete zhOverride[id]);
  save(); renderCard(); tally();
}

function go(i, scroll) {
  current = Math.min(Math.max(0, i), ITEMS.length - 1);
  state.position = current;
  try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { /* reported by save() */ }
  renderCard();
  tally();
  if (scroll !== false) window.scrollTo(0, 0);
}

function buildDots() {
  const dots = document.getElementById('dots');
  ITEMS.forEach((it, i) => {
    const d = el('button', 'dot');
    d.type = 'button';
    d.addEventListener('click', () => go(i));
    dots.append(d);
  });
}

function tally() {
  let done = 0, partial = 0, flagged = 0;
  const dots = document.getElementById('dots').children;
  ITEMS.forEach((it, i) => {
    const d = isDone(i), set = axesSet(i), e = entry(i);
    if (d) done++; else if (set) partial++;
    if (e && e.translation_unclear) flagged++;
    dots[i].className = 'dot' + (d ? ' done' : set ? ' partial' : '') +
      (e && e.translation_unclear ? ' flag' : '') + (i === current ? ' at' : '');
    dots[i].title = 'Item ' + (i + 1) + ': ' + (d ? 'scored' : set ? set + ' of ' + AXES.length + ' axes set' : 'not scored') +
      (e && e.translation_unclear ? ', translation unclear' : '');
    dots[i].setAttribute('aria-label', dots[i].title);
  });
  const t = document.getElementById('tally');
  t.replaceChildren(el('strong', null, done + ' / ' + ITEMS.length), ' scored' +
    (partial ? ' · ' + partial + ' partly scored' : '') + (flagged ? ' · ' + flagged + ' flagged translation unclear' : ''));
  const missing = ITEMS.length - done;
  const m = document.getElementById('missing');
  m.className = 'missing' + (missing ? '' : ' none');
  m.textContent = missing ? missing + ' not fully scored — export is still possible' : 'All items scored';
  document.getElementById('bar').style.width = (ITEMS.length ? 100 * done / ITEMS.length : 0) + '%';
  const p = document.getElementById('progress');
  p.setAttribute('aria-valuemax', String(ITEMS.length));
  p.setAttribute('aria-valuenow', String(done));
  const zh = document.getElementById('zh-all');
  zh.setAttribute('aria-pressed', String(!!state.show_zh));
  zh.textContent = state.show_zh ? 'Hide Chinese (all)' : 'Show Chinese (all)';
  const banner = document.getElementById('banner');
  banner.replaceChildren();
  if (!missing && ITEMS.length) {
    banner.append(el('div', 'done-banner', 'All ' + ITEMS.length + ' items are scored. Use Export scores to download the file.'));
  }
}

// The export file. Items with nothing entered are left out; unset axes are null.
function buildPayload() {
  const scores = {};
  ITEMS.forEach(it => {
    const e = state.scores[it.item_id];
    if (!hasContent(e)) return;
    const row = {};
    AXES.forEach(a => { row[a.key] = Number.isInteger(e[a.key]) ? e[a.key] : null; });
    row.note = e.note || '';
    row.translation_unclear = !!e.translation_unclear;
    scores[it.item_id] = row;
  });
  return { kind: META.kind, fingerprint: META.fingerprint, annotator: (state.annotator || '').trim(),
           exported_utc: new Date().toISOString(), scores };
}

function exportScores() {
  const annotator = (state.annotator || '').trim();
  if (!annotator) {
    document.getElementById('annotator').focus();
    alert('Put your name in the Annotator box first: the export file is named after it.');
    return;
  }
  const payload = buildPayload();
  if (!Object.keys(payload.scores).length) { alert('Nothing has been scored yet.'); return; }
  let done = 0;
  ITEMS.forEach((it, i) => { if (isDone(i)) done++; });
  const name = 'human_scores.' + slugify(annotator) + '.json';
  const blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  const missing = ITEMS.length - done;
  setSaveState('', 'Exported ' + name + ': ' + done + ' of ' + ITEMS.length + ' scored' +
    (missing ? ', ' + missing + ' not fully scored' : ''));
}

function importScores(file) {
  const reader = new FileReader();
  reader.onload = () => {
    let data;
    try { data = JSON.parse(reader.result); } catch (e) { alert('Not valid JSON.'); return; }
    if (!data || data.kind !== META.kind) { alert('This file is not an export of this scoring page.'); return; }
    if (data.fingerprint !== META.fingerprint) {
      alert('This export belongs to a different worksheet (fingerprint differs). Nothing was changed.');
      return;
    }
    const scores = cleanScores(data.scores);
    if (!scores) { alert('The export holds an unknown item or an out-of-range score. Nothing was changed.'); return; }
    const have = Object.values(state.scores).filter(hasContent).length;
    // An import replaces the answers; it never merges two annotators' partial work.
    if (have && !confirm('Replace the ' + have + ' answers in this browser with the ' +
        Object.keys(scores).length + ' in the file?')) return;
    state.scores = scores;
    state.annotator = typeof data.annotator === 'string' ? data.annotator : '';
    state.guide_read = true;
    document.getElementById('annotator').value = state.annotator;
    document.getElementById('guide').open = false;
    save();
    const next = firstUnscored();
    go(next === -1 ? 0 : next);
  };
  reader.readAsText(file);
}

function resumeNotice() {
  const have = Object.values(state.scores).filter(hasContent).length;
  if (!have) return;
  const when = state.saved_utc ? new Date(state.saved_utc).toLocaleString() : 'earlier';
  const note = el('div', 'resume', 'Resumed: ' + have + ' items with answers, last saved ' + when + '.');
  const reset = el('button', null, 'Start over');
  reset.type = 'button';
  reset.addEventListener('click', () => {
    if (!confirm('Delete every answer stored in this browser for this page?')) return;
    state = { ...emptyState(), annotator: state.annotator, show_zh: state.show_zh, guide_read: true };
    save(); go(0); note.remove();
  });
  note.append(' ', reset);
  document.getElementById('banner').before(note);
}

function renderGuide() {
  const table = document.getElementById('rubric');
  const head = el('tr');
  ['Axis', 'Range', 'Question and anchors'].forEach(h => head.append(el('th', null, h)));
  table.append(head);
  AXES.forEach(a => {
    const tr = el('tr');
    const name = el('td');
    name.append(el('code', null, a.key));
    tr.append(name, el('td', null, '0-' + a.max), el('td', null, a.line));
    table.append(tr);
  });
  const stages = document.getElementById('stages');
  Object.entries(STAGES).forEach(([letter, s]) => {
    stages.append(el('li', null, letter + ' (' + s.name + '): ' + s.text));
  });
}

document.getElementById('annotator').addEventListener('input', e => { state.annotator = e.target.value; save(); });
document.getElementById('export').addEventListener('click', exportScores);
document.getElementById('import').addEventListener('click', () => document.getElementById('file').click());
document.getElementById('file').addEventListener('change', e => {
  if (e.target.files[0]) importScores(e.target.files[0]);
  e.target.value = '';
});
document.getElementById('zh-all').addEventListener('click', toggleZhAll);
document.getElementById('prev').addEventListener('click', () => go(current - 1));
document.getElementById('next').addEventListener('click', () => go(current + 1));
document.getElementById('skip').addEventListener('click', () => {
  const i = firstUnscored();
  if (i === -1) alert('Every item is scored.'); else go(i);
});
document.getElementById('start').addEventListener('click', () => {
  document.getElementById('guide').open = false;
  window.scrollTo(0, 0);
});
document.getElementById('guide').addEventListener('toggle', e => {
  if (!e.target.open && !state.guide_read) { state.guide_read = true; save(); }
});

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = document.activeElement ? document.activeElement.tagName : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') { if (e.key === 'Escape') document.activeElement.blur(); return; }
  const cur = entry(current);
  if (e.key === 'ArrowRight') go(current + 1);
  else if (e.key === 'ArrowLeft') go(current - 1);
  else if (e.key === 'u' || e.key === 'U') { const i = firstUnscored(); if (i !== -1) go(i); }
  else if (e.key === 'z' || e.key === 'Z') toggleZh();
  else if (e.key === 'f' || e.key === 'F') toggleFlag();
  else if (e.key === 'Backspace') {
    const last = AXES.filter(a => cur && Number.isInteger(cur[a.key])).pop();
    if (!last) return;
    cur[last.key] = null;
    save(); renderCard(); tally();
  } else if (/^[0-9]$/.test(e.key)) {
    const axis = AXES.find(a => !cur || !Number.isInteger(cur[a.key]));
    const v = Number(e.key);
    if (!axis || v > axis.max) return;
    ensure(current)[axis.key] = v;
    save(); renderCard(); tally();
  } else return;
  e.preventDefault();
});

function boot() {
  renderGuide();
  buildDots();
  document.getElementById('annotator').value = state.annotator || '';
  const started = state.guide_read || Object.values(state.scores).some(hasContent);
  document.getElementById('guide').open = !started;
  const first = firstUnscored();
  go(Number.isInteger(state.position) ? state.position : (first === -1 ? 0 : first), false);
  resumeNotice();
  setSaveState('', 'Answers are saved in this browser — export before you finish');
  document.getElementById('foot').textContent =
    'Generated ' + META.generated_utc + ' · ' + ITEMS.length + ' items · worksheet fingerprint ' +
    META.fingerprint + ' · answers stay in this browser until you export.';
}
boot();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- build


def load_worksheet(path: Path) -> dict:
    """Read and shape-check a worksheet."""
    ws = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ws, dict) or not isinstance(ws.get("items"), list) or not ws["items"]:
        raise ValueError(f"{path}: expected an object with a non-empty 'items' list")
    fp = ws.get("fingerprint")
    if not isinstance(fp, str) or not fp.strip():
        raise ValueError(f"{path}: missing 'fingerprint'")
    ids = [it.get("item_id") for it in ws["items"]]
    if any(not isinstance(i, str) or not i for i in ids):
        raise ValueError(f"{path}: every item needs a string item_id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item_id")
    return ws


def public_item(item: dict) -> dict:
    """The fields of an item that the page may show, copied one by one."""

    def text(key: str) -> str:
        value = item.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"item {item.get('item_id')}: {key} must be a string")
        return value

    history = []
    for turn in item.get("history") or []:
        if turn.get("role") not in ("student", "teacher"):
            raise ValueError(f"item {item.get('item_id')}: history role must be student or teacher")
        history.append(
            {"role": turn["role"], "en": str(turn.get("en", "")), "zh": str(turn.get("zh", ""))}
        )
    return {
        "item_id": item["item_id"],
        "stage": text("stage"),
        "question_en": text("question_en"),
        "question_zh": text("question_zh"),
        "history": history,
        "student_en": text("student_en"),
        "student_zh": text("student_zh"),
        "reference_en": text("reference_en"),
        "reference_zh": text("reference_zh"),
        "reply_en": text("reply_en"),
        "reply_zh": text("reply_zh"),
    }


def _private_strings(value) -> set[str]:
    """Every string inside a private value: dict keys and string leaves."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        out = {k for k in value if isinstance(k, str)}
        for v in value.values():
            out |= _private_strings(v)
        return out
    if isinstance(value, list):
        return set().union(*(_private_strings(v) for v in value)) if value else set()
    return set()


def private_needles(worksheet: dict) -> set[str]:
    """Strings that must not occur in the page: private key names and private string values.

    Rubric axis names and "total" are left out: they are public and the page shows them.
    Numbers (dialogue id, turn index, judge scores) cannot be searched for as text; they
    are kept out by ``public_item`` copying named fields only.
    """
    needles: set[str] = set()
    for item in worksheet["items"]:
        for key, value in item.items():
            if key.startswith("_"):
                needles.add(key)
                needles |= _private_strings(value)
    return {n for n in needles if n.strip()} - set(AXIS_KEYS) - {"total"}


def _inline(payload: str) -> str:
    """Make JSON safe inside a <script> block: no ``</script>`` or ``<!--`` can form."""
    return payload.replace("<", "\\u003c")


def render(worksheet: dict) -> str:
    """Render the page. Raises ValueError if any private key or value would be in it."""
    items = [public_item(it) for it in worksheet["items"]]
    meta = {
        "kind": KIND,
        "fingerprint": worksheet["fingerprint"],
        "store_key": STORE_PREFIX + worksheet["fingerprint"],
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "axes": [{**a, "line": rubric_line(a)} for a in AXES],
        "stages": STAGES,
        "reference_label": REFERENCE_LABEL,
    }
    html = _TEMPLATE.replace("__ITEMS_JSON__", _inline(json.dumps(items, ensure_ascii=False)))
    html = html.replace("__META_JSON__", _inline(json.dumps(meta, ensure_ascii=False)))
    leaks = sorted(n for n in private_needles(worksheet) if n in html)
    if leaks:
        raise ValueError(
            "private worksheet content would appear in the page: "
            + ", ".join(repr(n) for n in leaks[:8])
            + ". If a private value (for example a run name) also occurs in the visible "
            "text of an item, change the worksheet; the page is not written."
        )
    return html


def cmd_build(args: argparse.Namespace) -> None:
    worksheet = load_worksheet(args.worksheet)
    try:
        html = render(worksheet)
    except ValueError as exc:
        raise SystemExit(f"build refused: {exc}") from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}: {len(worksheet['items'])} items, {len(html) / 1024:.0f} KiB")
    print(f"fingerprint {worksheet['fingerprint']}")
    print("no private key or private string value occurs in the page")


# --------------------------------------------------------------------------- score


def weighted_kappa(a: list[int], b: list[int], max_score: int) -> float | None:
    """Quadratically weighted Cohen's kappa over the categories 0..max_score.

    Returns None when it is undefined: no items, or both raters used one and the
    same category for every item (expected disagreement is zero).
    """
    if len(a) != len(b):
        raise ValueError("rating lists differ in length")
    n = len(a)
    if n == 0:
        return None
    k = max_score + 1
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[x, y] += 1
    idx = np.arange(k)
    weights = (idx[:, None] - idx[None, :]) ** 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    denom = float((weights * expected).sum())
    if denom == 0:
        return None
    return 1.0 - float((weights * observed).sum()) / denom


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation; None with fewer than 3 items or a constant input."""
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    rho = float(spearmanr(a, b)[0])  # type: ignore[arg-type]
    return None if math.isnan(rho) else rho


def validate_export(worksheet: dict, export: dict) -> dict[str, dict]:
    """Check kind, fingerprint, item ids and score ranges; return the score map."""
    if not isinstance(export, dict) or export.get("kind") != KIND:
        raise ValueError(f"not a {KIND} export")
    if export.get("fingerprint") != worksheet["fingerprint"]:
        raise ValueError(
            f"fingerprint mismatch: export has {export.get('fingerprint')!r}, "
            f"worksheet has {worksheet['fingerprint']!r}; the export belongs to another page"
        )
    scores = export.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("export has no 'scores' object")
    known = {it["item_id"] for it in worksheet["items"]}
    unknown = sorted(set(scores) - known)
    if unknown:
        raise ValueError(f"export has items that are not in the worksheet: {unknown[:5]}")
    for item_id, entry in scores.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{item_id}: entry is not an object")
        for axis in AXIS_KEYS:
            v = entry.get(axis)
            if v is None:
                continue
            if type(v) is not int or not 0 <= v <= AXIS_MAX[axis]:
                raise ValueError(f"{item_id}: {axis}={v!r} is outside 0-{AXIS_MAX[axis]}")
    return scores


def _is_complete(entry: dict) -> bool:
    return all(type(entry.get(axis)) is int for axis in AXIS_KEYS)


def _judge_entry(item: dict, judge: str) -> dict | None:
    """A judge's scores for an item, with total recomputed if absent; None if incomplete."""
    raw = (item.get("_judge") or {}).get(judge)
    if not isinstance(raw, dict) or any(not isinstance(raw.get(a), int | float) for a in AXIS_KEYS):
        return None
    total = raw.get("total")
    if not isinstance(total, int | float):
        total = sum(raw[a] for a in AXIS_KEYS)
    return {**{a: raw[a] for a in AXIS_KEYS}, "total": total}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _order(means: dict[str, float | None]) -> list[str] | None:
    """Systems from highest to lowest mean; None if a mean is missing or two are equal."""
    vals = list(means.values())
    if len(vals) < 2 or any(v is None for v in vals) or len(set(vals)) != len(vals):
        return None
    return sorted(means, key=lambda s: means[s], reverse=True)  # type: ignore[arg-type,return-value]


def totals_agreement(rows: list[dict], judges: list[str]) -> dict[str, dict]:
    """Section 2: human against each judge on item totals."""
    out = {}
    for judge in judges:
        pairs = [
            (r["human"]["total"], r["judge"][judge]["total"]) for r in rows if judge in r["judge"]
        ]
        human = [float(h) for h, _ in pairs]
        other = [float(j) for _, j in pairs]
        diffs = [j - h for h, j in zip(human, other, strict=True)]
        out[judge] = {
            "n": len(pairs),
            "spearman": spearman(human, other),
            "mean_abs_diff": _mean([abs(d) for d in diffs]),
            "mean_signed_diff_judge_minus_human": _mean(diffs),
        }
    return out


def bootstrap_pairs(
    rows: list[dict], systems: list[str], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> list[dict]:
    """Section 4: difference in mean human total for each pair of systems.

    Resamples dialogues with replacement. A pair uses the dialogues in which both
    systems have at least one scored item; the same draw serves both systems.
    """
    out = []
    for a, b in itertools.combinations(systems, 2):
        per: dict = {}
        for r in rows:
            if r["system"] in (a, b):
                cell = per.setdefault(r["dialogue"], {a: [0.0, 0], b: [0.0, 0]})[r["system"]]
                cell[0] += r["human"]["total"]
                cell[1] += 1
        common = sorted((d for d, c in per.items() if c[a][1] and c[b][1]), key=str)
        result: dict = {
            "a": a,
            "b": b,
            "n_dialogues": len(common),
            "n_a": sum(per[d][a][1] for d in common),
            "n_b": sum(per[d][b][1] for d in common),
            "diff_a_minus_b": None,
            "ci95": None,
            "n_boot": n_boot,
            "seed": seed,
        }
        if common:
            sums = np.array([[per[d][a][0], per[d][b][0]] for d in common])
            counts = np.array([[per[d][a][1], per[d][b][1]] for d in common], dtype=float)
            point = sums.sum(axis=0) / counts.sum(axis=0)
            result["diff_a_minus_b"] = float(point[0] - point[1])
        if len(common) >= 2:
            rng = np.random.default_rng(seed)
            draws = rng.multinomial(len(common), np.full(len(common), 1 / len(common)), size=n_boot)
            means = (draws @ sums) / (draws @ counts)
            lo, hi = np.percentile(means[:, 0] - means[:, 1], [2.5, 97.5])
            result["ci95"] = [float(lo), float(hi)]
        out.append(result)
    return out


def score(worksheet: dict, export: dict, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Compare an export with the worksheet's judge scores. See the module docstring."""
    scores = validate_export(worksheet, export)
    items = worksheet["items"]
    rows = []
    for item in items:
        entry = scores.get(item["item_id"])
        if entry is None or not _is_complete(entry):
            continue
        human = {a: entry[a] for a in AXIS_KEYS}
        human["total"] = sum(human.values())
        judge = {}
        for name in item.get("_judge") or {}:
            j = _judge_entry(item, name)
            if j is not None:
                judge[name] = j
        rows.append(
            {
                "item_id": item["item_id"],
                "system": item["_system"],
                "dialogue": item["_dialogue_id"],
                "human": human,
                "judge": judge,
                "flagged": bool(entry.get("translation_unclear")),
            }
        )
    judges = sorted({name for it in items for name in (it.get("_judge") or {})})
    systems = sorted({it["_system"] for it in items})

    per_system: dict[str, dict] = {}
    for system in systems:
        mine = [r for r in rows if r["system"] == system]
        per_system[system] = {
            "n_in_worksheet": sum(it["_system"] == system for it in items),
            "n": len(mine),
            "human_mean_total": _mean([r["human"]["total"] for r in mine]),
            "judges": {},
        }
        for judge in judges:
            both = [r for r in mine if judge in r["judge"]]
            per_system[system]["judges"][judge] = {
                "n": len(both),
                "judge_mean_total": _mean([r["judge"][judge]["total"] for r in both]),
                "human_mean_total_same_items": _mean([r["human"]["total"] for r in both]),
            }
    ordering: dict[str, dict] = {}
    for judge in judges:
        cells = {s: per_system[s]["judges"][judge] for s in systems}
        human_order = _order({s: c["human_mean_total_same_items"] for s, c in cells.items()})
        judge_order = _order({s: c["judge_mean_total"] for s, c in cells.items()})
        ordering[judge] = {
            "human_order": human_order,
            "judge_order": judge_order,
            "same_order": (
                None if human_order is None or judge_order is None else human_order == judge_order
            ),
        }

    per_axis: dict[str, dict] = {}
    for axis in AXIS_KEYS:
        per_axis[axis] = {}
        for judge in judges:
            both = [r for r in rows if judge in r["judge"]]
            human = [r["human"][axis] for r in both]
            other = [r["judge"][judge][axis] for r in both]
            # Kappa needs integer categories inside the axis range.
            usable = all(float(v).is_integer() and 0 <= v <= AXIS_MAX[axis] for v in other)
            per_axis[axis][judge] = {
                "n": len(both),
                "exact_agreement": _mean(
                    [float(h == j) for h, j in zip(human, other, strict=True)]
                ),
                "quadratic_weighted_kappa": (
                    weighted_kappa(human, [int(v) for v in other], AXIS_MAX[axis])
                    if usable
                    else None
                ),
            }

    unflagged = [r for r in rows if not r["flagged"]]
    return {
        "kind": "kele_human_scoring_agreement",
        "fingerprint": worksheet["fingerprint"],
        "annotator": export.get("annotator"),
        "exported_utc": export.get("exported_utc"),
        "n_items_in_worksheet": len(items),
        "n_items_in_export": len(scores),
        "n_complete": len(rows),
        "n_partial_or_empty_in_export": len(scores) - len(rows),
        "judges": judges,
        "systems": systems,
        "per_system": per_system,
        "system_order": ordering,
        "totals": totals_agreement(rows, judges),
        "per_axis": per_axis,
        "bootstrap_human_total": bootstrap_pairs(rows, systems, n_boot=n_boot, seed=seed),
        "translation_unclear": {
            "n_flagged_in_export": sum(bool(e.get("translation_unclear")) for e in scores.values()),
            "n_flagged_complete": len(rows) - len(unflagged),
            "n_complete_without_flagged": len(unflagged),
            "totals_without_flagged": totals_agreement(unflagged, judges),
        },
    }


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def format_report(res: dict) -> str:
    """The printed report: plain text, n beside every number."""
    out = [
        f"annotator: {res['annotator']}   exported: {res['exported_utc']}",
        f"items with all four axes scored: {res['n_complete']} of {res['n_items_in_worksheet']} "
        f"({res['n_partial_or_empty_in_export']} partly scored or empty entries in the export "
        f"are not used; {res['n_items_in_worksheet'] - res['n_items_in_export']} items are not "
        "in the export)",
        "",
        "1. Mean total (0-10) per system",
    ]
    for system, s in res["per_system"].items():
        out.append(
            f"  {system}: human {_fmt(s['human_mean_total'])} (n={s['n']} of {s['n_in_worksheet']})"
        )
        for judge, j in s["judges"].items():
            out.append(
                f"      judge {judge}: {_fmt(j['judge_mean_total'])}, human on the same items "
                f"{_fmt(j['human_mean_total_same_items'])} (n={j['n']})"
            )
    for judge, o in res["system_order"].items():
        if o["same_order"] is None:
            verdict = "not comparable (a system has no scored item, or two means are equal)"
        else:
            verdict = "same order" if o["same_order"] else "DIFFERENT order"
        human = " > ".join(o["human_order"]) if o["human_order"] else "n/a"
        other = " > ".join(o["judge_order"]) if o["judge_order"] else "n/a"
        out.append(f"  order, human vs {judge}: {verdict}. human: {human}; judge: {other}")

    def totals_block(block: dict) -> list[str]:
        return [
            f"  {judge}: Spearman {_fmt(t['spearman'], '.3f')}, mean absolute difference "
            f"{_fmt(t['mean_abs_diff'])}, mean signed difference (judge minus human) "
            f"{_fmt(t['mean_signed_diff_judge_minus_human'], '+.2f')} (n={t['n']})"
            for judge, t in block.items()
        ]

    out += ["", "2. Human vs judge on item totals"] + totals_block(res["totals"])
    out += ["", "3. Per axis: exact agreement and quadratically weighted kappa"]
    for axis, by_judge in res["per_axis"].items():
        for judge, a in by_judge.items():
            out.append(
                f"  {axis} vs {judge}: exact {_fmt(a['exact_agreement'], '.1%')}, "
                f"kappa {_fmt(a['quadratic_weighted_kappa'], '.3f')} (n={a['n']})"
            )
    out += ["", "4. Difference in mean human total between systems (dialogue bootstrap)"]
    for b in res["bootstrap_human_total"]:
        ci = "n/a" if b["ci95"] is None else f"[{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]"
        out.append(
            f"  {b['a']} minus {b['b']}: {_fmt(b['diff_a_minus_b'], '+.2f')}, 95% interval {ci} "
            f"({b['n_dialogues']} dialogues with both; n={b['n_a']} and {b['n_b']} items; "
            f"{b['n_boot']} resamples, seed {b['seed']})"
        )
    tu = res["translation_unclear"]
    out += [
        "",
        f"5. Translation unclear: {tu['n_flagged_in_export']} items flagged in the export, "
        f"{tu['n_flagged_complete']} of them fully scored. Without those "
        f"(n={tu['n_complete_without_flagged']}):",
    ] + totals_block(tu["totals_without_flagged"])
    return "\n".join(out)


def cmd_score(args: argparse.Namespace) -> None:
    worksheet = load_worksheet(args.worksheet)
    export = json.loads(args.export.read_text(encoding="utf-8"))
    try:
        result = score(worksheet, export, n_boot=args.n_boot, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"{args.export}: {exc}") from exc
    print(format_report(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")
        print(f"\nwrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build the scoring page from a worksheet")
    b.add_argument("--worksheet", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True, help="HTML file to write")
    s = sub.add_parser("score", help="Compare an export with the judge scores in the worksheet")
    s.add_argument("export", type=Path, help="human_scores.<annotator>.json from the page")
    s.add_argument("--worksheet", type=Path, required=True)
    s.add_argument("--out", type=Path, help="Also write the full result as JSON")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--seed", type=int, default=BOOT_SEED)
    args = ap.parse_args(argv)
    {"build": cmd_build, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
