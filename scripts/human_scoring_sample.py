#!/usr/bin/env python3
"""Draw the sample for human scoring and build the private worksheet.

Three steps, so that the only step that calls a model can run on another machine:

    # 1. Sample turns and attach the English dataset text and the judges' scores.
    uv run python scripts/human_scoring_sample.py sample \\
        --run gemma31b=<run dir> --run socratteachllm=<run dir> --run glm49b=<run dir> \\
        --judge qwen36=judge_summary --judge bedrock_sonnet46=judge_summary_bedrock-claude-sonnet-4-6 \\
        --out results/human_scoring/worksheet.json
    # 2. Translate the systems' Chinese replies (Amazon Bedrock; needs boto3 and AWS credentials).
    uv run --with boto3 python scripts/human_scoring_sample.py translate \\
        --worksheet results/human_scoring/worksheet.json --model <Bedrock model id>
    # Later, to add or refresh a judge's scores: the `judges` subcommand.
    # 3. Build the page: scripts/human_scoring_page.py build.

Sampling: N dialogue turns, split evenly over stages b to e by ground-truth state, at most one
turn per dialogue, and only turns that every run actually produced (runs stop early at different
points). Each sampled turn yields one item per system, so the same turns are scored for every
system. Items are then shuffled. Student turns and reference replies come from the English
machine translation of the dataset (ulises-c/SocratDataset-EN); a system's own replies, in the
history and the one to score, are Chinese and are translated in step 2.

Keys starting with an underscore are private: system, run, dialogue and judge scores. They must
not reach the scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

AXES = ("socratic_validity", "advancement", "age_appropriateness", "question_form")

TRANSLATE_SYSTEM = (
    "You translate a Chinese elementary-school teacher's reply into English for a human rater. "
    "Translate literally and completely. Keep the same number of sentences and questions, the same "
    "level of vocabulary, and any mistakes, repetition, odd phrasing or off-topic content; do not "
    "improve, shorten, explain or correct anything. If the text is already English, return it "
    "unchanged. Output only the translation."
)


def _dialogue_files(run_dir: Path) -> dict[str, Path]:
    return {
        f.name: f
        for f in sorted((run_dir / "dialogues").glob("*.json"))
        if not f.name.startswith(".")
    }


def _judge_scores(run_dir: Path, stem: str, fname: str, turn_idx: int) -> dict | None:
    path = run_dir / "judge_scores" / stem / fname
    if not path.is_file():
        return None
    for s in json.loads(path.read_text())["scores"]:
        if s["turn_idx"] == turn_idx:
            return {a: s[a] for a in AXES} | {"total": s["total"]}
    return None


def build_sample(
    runs: dict[str, Path],
    judges: dict[str, str],
    english: dict[int, dict],
    n_turns: int,
    stages: str,
    seed: int,
) -> list[dict]:
    """Sampled items, shuffled. ``english`` maps a dialogue id to its English dataset record."""
    files = {name: _dialogue_files(d) for name, d in runs.items()}
    shared = sorted(set.intersection(*(set(f) for f in files.values())))
    loaded = {name: {f: json.loads(files[name][f].read_text()) for f in shared} for name in runs}

    # Candidate turns per stage: present in every run, with English text available.
    by_stage: dict[str, list[tuple[str, int]]] = {s: [] for s in stages}
    for fname in shared:
        dialogues = [loaded[name][fname] for name in runs]
        en = english.get(dialogues[0]["id"])
        if en is None:
            continue
        n_common = min(len(d["dialogue"]) for d in dialogues)
        for i in range(min(n_common, len(en["dialogue"]))):
            stage = dialogues[0]["dialogue"][i].get("ground_truth_state", "?")[:1]
            if stage in by_stage:
                by_stage[stage].append((fname, i))

    rng = random.Random(seed)
    per_stage = {
        s: n_turns // len(stages) + (1 if k < n_turns % len(stages) else 0)
        for k, s in enumerate(stages)
    }
    used: set[str] = set()
    picked: list[tuple[str, int, str]] = []
    # Rarest stage first, so the one-turn-per-dialogue rule does not starve it.
    for stage in sorted(stages, key=lambda s: len(by_stage[s])):
        pool = list(by_stage[stage])
        rng.shuffle(pool)
        taken = 0
        for fname, i in pool:
            if taken == per_stage[stage]:
                break
            if fname in used:
                continue
            used.add(fname)
            picked.append((fname, i, stage))
            taken += 1

    items = []
    for fname, i, stage in picked:
        for name, run_dir in runs.items():
            d = loaded[name][fname]
            en = english[d["id"]]
            turns = d["dialogue"]
            history = []
            for k in range(i):
                history.append(
                    {
                        "role": "student",
                        "en": en["dialogue"][k]["student"],
                        "zh": turns[k]["student"],
                    }
                )
                history.append(
                    {"role": "teacher", "en": "", "zh": turns[k]["teacher_response"].strip()}
                )
            items.append(
                {
                    "stage": stage,
                    "question_en": str(en.get("question", "")),
                    "question_zh": str(d.get("question", "")),
                    "history": history,
                    "student_en": en["dialogue"][i]["student"],
                    "student_zh": turns[i]["student"],
                    "reference_en": en["dialogue"][i]["teacher"],
                    "reference_zh": turns[i]["ground_truth_teacher"],
                    "reply_en": "",
                    "reply_zh": turns[i]["teacher_response"].strip(),
                    "_system": name,
                    "_run": run_dir.name,
                    "_dialogue_file": fname,
                    "_dialogue_id": d["id"],
                    "_turn_idx": i,
                    "_ground_truth_state": turns[i].get("ground_truth_state", "?"),
                    "_judge": {
                        j: sc
                        for j, stem in judges.items()
                        if (sc := _judge_scores(run_dir, stem, fname, i)) is not None
                    },
                }
            )
    rng.shuffle(items)
    for k, item in enumerate(items, 1):
        item["item_id"] = f"h{k:03d}"
    return items


def fingerprint(items: list[dict]) -> str:
    """Hash of what the scorer sees, so an export can be tied to this worksheet."""
    public = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
    return hashlib.sha256(
        json.dumps(public, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def pending_translations(items: list[dict]) -> list[str]:
    """Distinct Chinese texts that still need English."""
    todo: dict[str, None] = {}
    for it in items:
        if not it["reply_en"]:
            todo[it["reply_zh"]] = None
        for h in it["history"]:
            if h["role"] == "teacher" and not h["en"]:
                todo[h["zh"]] = None
    return list(todo)


def apply_translations(items: list[dict], table: dict[str, str]) -> None:
    for it in items:
        if not it["reply_en"] and it["reply_zh"] in table:
            it["reply_en"] = table[it["reply_zh"]]
        for h in it["history"]:
            if h["role"] == "teacher" and not h["en"] and h["zh"] in table:
                h["en"] = table[h["zh"]]


def _translate_bedrock(client, model: str, text: str, max_retries: int = 6) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = client.converse(
                modelId=model,
                system=[{"text": TRANSLATE_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": text}]}],
                # No temperature: newer Claude models on Bedrock reject the parameter.
                inferenceConfig={"maxTokens": 1500},
            )
            return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()
        except Exception as e:
            print(f"  bedrock error (attempt {attempt + 1}): {str(e)[:200]}", file=sys.stderr)
            time.sleep(min(2**attempt, 30))
    return None


def _load_english(repo: str) -> dict[int, dict]:
    from src.project.kele import load_dataset

    return {d["id"]: d for d in load_dataset(split="test", hf_repo=repo)}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", action="append", required=True, metavar="SYSTEM=DIR")
    s.add_argument("--judge", action="append", default=[], metavar="NAME=SUMMARY_STEM")
    s.add_argument("--english-repo", default="ulises-c/SocratDataset-EN")
    s.add_argument("--turns", type=int, default=50, help="Turns sampled; items = turns x systems")
    s.add_argument("--stages", default="bcde")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("translate")
    t.add_argument("--worksheet", type=Path, required=True)
    t.add_argument("--model", required=True, help="Bedrock model or inference-profile id")
    j = sub.add_parser("judges", help="Refresh the private judge scores from the run folders")
    j.add_argument("--worksheet", type=Path, required=True)
    j.add_argument(
        "--runs-root", type=Path, action="append", required=True, help="Folder holding run dirs"
    )
    j.add_argument("--judge", action="append", required=True, metavar="NAME=SUMMARY_STEM")
    args = ap.parse_args()

    if args.cmd == "judges":
        sheet = json.loads(args.worksheet.read_text())
        judges = dict(x.split("=", 1) for x in args.judge)
        missing = 0
        for it in sheet["items"]:
            run_dir = next(
                (r / it["_run"] for r in args.runs_root if (r / it["_run"]).is_dir()), None
            )
            for name, stem in judges.items():
                sc = run_dir and _judge_scores(run_dir, stem, it["_dialogue_file"], it["_turn_idx"])
                if sc:
                    it["_judge"][name] = sc
                else:
                    missing += 1
        args.worksheet.write_text(json.dumps(sheet, ensure_ascii=False, indent=1))
        print(f"judge scores refreshed for {len(sheet['items'])} items; {missing} missing")
        return 0

    if args.cmd == "sample":
        runs = {k: Path(v) for k, v in (r.split("=", 1) for r in args.run)}
        judges = dict(j.split("=", 1) for j in args.judge)
        items = build_sample(
            runs, judges, _load_english(args.english_repo), args.turns, args.stages, args.seed
        )
        sheet = {
            "seed": args.seed,
            "english_repo": args.english_repo,
            "translator_model": None,
            "fingerprint": None,
            "items": items,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(sheet, ensure_ascii=False, indent=1))
        counts: dict[str, int] = {}
        for it in items:
            counts[it["stage"]] = counts.get(it["stage"], 0) + 1
        print(f"{len(items)} items, by stage {dict(sorted(counts.items()))}; ")
        print(f"{len(pending_translations(items))} distinct replies to translate -> {args.out}")
        return 0

    import boto3  # type: ignore[import-not-found]  # optional: only this step needs it

    sheet = json.loads(args.worksheet.read_text())
    todo = pending_translations(sheet["items"])
    client = boto3.client("bedrock-runtime")
    table: dict[str, str] = {}
    for k, text in enumerate(todo, 1):
        out = _translate_bedrock(client, args.model, text)
        if out:
            table[text] = out
        if k % 25 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)} translated", flush=True)
            apply_translations(sheet["items"], table)
            args.worksheet.write_text(json.dumps(sheet, ensure_ascii=False, indent=1))
    apply_translations(sheet["items"], table)
    left = len(pending_translations(sheet["items"]))
    sheet["translator_model"] = args.model
    sheet["fingerprint"] = fingerprint(sheet["items"]) if left == 0 else None
    args.worksheet.write_text(json.dumps(sheet, ensure_ascii=False, indent=1))
    print(f"{len(todo) - left} translated, {left} left; fingerprint {sheet['fingerprint']}")
    return 0 if left == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
