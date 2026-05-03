#!/usr/bin/env python3
"""translate_dataset.py — Translate SocratDataset.json (Chinese → English).

Calls any OpenAI-compatible local LLM (vLLM, Ollama, llama.cpp).
Saves a local checkpoint every LOCAL_CHECKPOINT_EVERY records and uploads
it to HuggingFace every HF_CHECKPOINT_EVERY records, so the run can be
resumed from HF if the local machine dies.

Auth: run `hf auth login` once (or set HF_TOKEN env var) before starting.

Usage:
    # Smoke test (10 records):
    poetry run python -m src.project.translate_dataset --smoke-test 10 --no-push

    # Full overnight run (checkpoints + final push all go to HF automatically):
    poetry run python -m src.project.translate_dataset

    # Resume from a previously uploaded HF checkpoint:
    poetry run python -m src.project.translate_dataset --restore-from-hub

    # Skip upload for this run without editing the file:
    poetry run python -m src.project.translate_dataset --no-push
"""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

# ── Job Configuration — edit here before running ──────────────────────────────
#
# Set PUSH_TO_HUB=True to auto-upload the completed dataset and periodic
# checkpoints to HuggingFace. Requires `hf auth login` or HF_TOKEN env var.
PUSH_TO_HUB: bool = True
HF_REPO: str = "ulises-c/SocratDataset-EN"

# Upload a checkpoint to HF every N completed records (0 to disable).
# The checkpoint is uploaded as a file to the same HF repo so the run
# can be recovered from HF if the local machine loses power or crashes.
HF_CHECKPOINT_EVERY: int = 50

# Local checkpoint is always saved every N records regardless of HF setting.
LOCAL_CHECKPOINT_EVERY: int = 5

# Local LLM server — swap freely, the script is API-agnostic.
# Recommended: Qwen/Qwen3.5-27B at Q4_K_M (~16.5 GB VRAM on R9700)
MODEL: str = "Qwen3.5-9B-UD-Q4_K_XL.gguf"
BASE_URL: str = "http://localhost:8000/v1"

# Total tokens the model may generate per call (thinking + output combined).
# Match this to the server's -c (context) value. At 32768 context the model
# will never actually generate that many for a translation — it's a ceiling.
MAX_TOKENS: int = 32768

# Tokens carved out of MAX_TOKENS for Qwen3 chain-of-thought reasoning.
# The model thinks for up to this many tokens, then writes the translation.
# Set 0 to disable thinking entirely (chat_template_kwargs enable_thinking=false).
# Rough guide (translation is not complex reasoning, so keep this modest):
#   0    — fastest, no reasoning, ~500-800 output tokens per record
#   1024 — light check
#   4096 — generous budget; model reasons deeply before translating
THINKING_BUDGET: int = 0

# HuggingFace source dataset (loaded via the datasets library at runtime).
# references/ is read-only; never write output files there.
HF_INPUT_REPO: str = "ulises-c/SocratDataset"

# Append all log lines to this file in addition to stdout. Set "" to disable.
LOG_PATH: str = "data/translate.log"

# All write paths land in data/ (gitignored; final dataset is uploaded to HF).
OUTPUT_PATH: str = "data/SocratDataset-EN.json"
CHECKPOINT_PATH: str = "data/translate_checkpoint.json"

# ──────────────────────────────────────────────────────────────────────────────

_log_fh = None  # file handle opened in main() when LOG_PATH is set


def _log(msg: str) -> None:
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


# ── Categorical lookup tables (no LLM needed) ─────────────────────────────────

MISSION_MAP = {
    "选择题": "multiple_choice",
    "判断题": "true_false",
}

GRADE_MAP = {
    "1小学一年级上册": "Grade 1 Vol. 1",
    "1小学一年级下册": "Grade 1 Vol. 2",
    "2小学二年级上册": "Grade 2 Vol. 1",
    "2小学二年级下册": "Grade 2 Vol. 2",
    "3小学三年级上册": "Grade 3 Vol. 1",
    "3小学三年级下册": "Grade 3 Vol. 2",
    "4小学四年级上册": "Grade 4 Vol. 1",
    "4小学四年级下册": "Grade 4 Vol. 2",
    "5小学五年级上册": "Grade 5 Vol. 1",
    "5小学五年级下册": "Grade 5 Vol. 2",
    "6小学六年级上册": "Grade 6 Vol. 1",
    "6小学六年级下册": "Grade 6 Vol. 2",
}

# State codes (a1, b2 … e34), literal "None", and empty string pass through.
# All other action values are Chinese strings handled via the lookup cache.
_ACTION_PASSTHROUGH = re.compile(r"^[a-e]\d+$|^None$|^$")

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a professional translator specializing in Chinese elementary school \
science education. Translate all Chinese text in the JSON to English.

Rules:
- Preserve exact JSON structure and all keys unchanged.
- Translate values only, never keys.
- Keep the Socratic / pedagogical tone in dialogue fields.
- Student and teacher voices should sound natural for elementary school age.
- Never use double-quote characters (\") inside translated text values. Use single quotes (') if quoting is needed.
- Return ONLY valid JSON — no markdown fences, no commentary.\
"""

_ONE_SHOT = """\
Example input:
{"question":"下列哪种动物是哺乳动物？","options":["鲨鱼","蝙蝠","鳄鱼"],\
"newHint":"想想哪种动物用乳汁哺育幼崽。",\
"newKnowledgePoint":"哺乳动物是温血动物，用乳汁哺育幼崽，身体覆盖毛发或皮毛。",\
"newAnalyze":"鲨鱼是鱼类，鳄鱼是爬行动物，两者都是冷血动物。蝙蝠是哺乳动物。答案是蝙蝠。",\
"dialogue":[{"student":"下列哪种动物是哺乳动物？A.鲨鱼 B.蝙蝠 C.鳄鱼",\
"evaluation":"学生初始回答",\
"teacher":"让我们思考一下，哺乳动物有什么特征呢？"}]}

Example output:
{"question":"Which of the following animals is a mammal?","options":["Shark","Bat","Crocodile"],\
"newHint":"Think about which animal feeds its young with milk.",\
"newKnowledgePoint":"Mammals are warm-blooded animals that nurse their young with milk and have bodies covered in hair or fur.",\
"newAnalyze":"Sharks are fish and crocodiles are reptiles — both cold-blooded. Bats are mammals. The answer is Bat.",\
"dialogue":[{"student":"Which of the following animals is a mammal? A. Shark  B. Bat  C. Crocodile",\
"evaluation":"Student initial response",\
"teacher":"Let us think — what are the characteristics of a mammal?"}]}\
"""


# ── Core translation helpers ──────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text)


def _parse_json(text: str) -> dict:
    """Parse JSON with recovery for truncated responses (missing closing braces)."""
    text = _strip_fences(text)
    for suffix in ("", "}", "}]}", "]}", "}]}"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
    json.loads(text)  # re-raise original error
    return {}  # unreachable


# Matches inline option arrays embedded in student text, e.g.:
#   ["缺水", "缺阳光", "缺肥料"] or ['Yes', 'No']
# The model converts single-quoted versions back to double-quoted in its output,
# so stripping the array entirely is safer. Options are already in the options field.
_INLINE_ARRAY_RE = re.compile(r'\s*\[(?:"[^"]*"|\'[^\']*\')(?:,\s*(?:"[^"]*"|\'[^\']*\'))*\]')


def _safe_quotes(obj: object) -> object:
    """Strip inline option arrays and remaining ASCII double-quotes from string values."""
    if isinstance(obj, str):
        return _INLINE_ARRAY_RE.sub("", obj).replace('"', "'").replace("“", "'").replace("”", "'")
    if isinstance(obj, list):
        return [_safe_quotes(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _safe_quotes(v) for k, v in obj.items()}
    return obj


def _build_payload(record: dict) -> str:
    payload = _safe_quotes(
        {
            "question": record["question"],
            "options": record["options"],
            "newHint": record["newHint"],
            "newKnowledgePoint": record["newKnowledgePoint"],
            "newAnalyze": record["newAnalyze"],
            "dialogue": [
                {
                    "student": t["student"],
                    "evaluation": t["evaluation"],
                    "teacher": t["teacher"],
                }
                for t in record["dialogue"]
            ],
        }
    )
    return _ONE_SHOT + "\n\nNow translate:\n" + json.dumps(payload, ensure_ascii=False)


_ESCAPE_REMINDER = (
    "\n\nIMPORTANT: your previous response contained invalid JSON. "
    'If any translated text contains double-quote characters, escape them as \\" in the JSON output.'
)


def _merge_split_turns(turns: list[dict], expected: int) -> list[dict]:
    """Merge split turns when the model separates student/teacher into individual turns.

    The model sometimes outputs 2*N turns where student turns and teacher turns alternate
    instead of N combined turns. Detect and merge back into N proper turns.
    """
    if len(turns) != 2 * expected:
        return turns
    # Pattern A: even=student-only, odd=teacher-only
    is_a = all(
        "student" in turns[2 * i]
        and "teacher" not in turns[2 * i]
        and "teacher" in turns[2 * i + 1]
        and "student" not in turns[2 * i + 1]
        for i in range(expected)
    )
    # Pattern B: even=teacher-only, odd=student-only
    is_b = all(
        "teacher" in turns[2 * i]
        and "student" not in turns[2 * i]
        and "student" in turns[2 * i + 1]
        and "teacher" not in turns[2 * i + 1]
        for i in range(expected)
    )
    if not (is_a or is_b):
        return turns
    merged = []
    for i in range(expected):
        s_turn = turns[2 * i] if is_a else turns[2 * i + 1]
        t_turn = turns[2 * i + 1] if is_a else turns[2 * i]
        merged.append(
            {
                "student": s_turn.get("student", ""),
                "evaluation": s_turn.get("evaluation", t_turn.get("evaluation", "")),
                "teacher": t_turn.get("teacher", ""),
            }
        )
    return merged


def translate_record(client: OpenAI, model: str, record: dict, retries: int = 3) -> dict:
    prompt = _build_payload(record)
    extra: dict = {}
    if THINKING_BUDGET == 0:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    last_err: Exception | None = None
    for attempt in range(retries):
        # After the first JSON failure, remind the model to escape quotes.
        user_prompt = prompt if attempt == 0 else prompt + _ESCAPE_REMINDER
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_TOKENS,
                extra_body=extra,
            )
            result = _parse_json(resp.choices[0].message.content)
            expected = len(record["dialogue"])
            result["dialogue"] = _merge_split_turns(result.get("dialogue", []), expected)
            got = len(result.get("dialogue", []))
            if got < expected:
                raise ValueError(f"dialogue truncated: got {got} turns, expected {expected}")
            return result
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"id={record['id']} failed after {retries} attempts: {last_err}")


_ACTION_ESCAPE_REMINDER = (
    " IMPORTANT: your previous response contained invalid JSON. "
    'Escape any double-quote characters inside string values as \\".'
)
_ACTION_SYSTEM = (
    "Translate each Chinese string in the JSON array to English. "
    "Return ONLY a JSON array of translated strings in the same order."
)


def _translate_action_chunk(
    client: OpenAI, model: str, chunk: list[str], retries: int = 3
) -> list[str]:
    """Translate one chunk of action strings; falls back to per-string on persistent failure."""
    user_content = json.dumps(chunk, ensure_ascii=False)
    extra: dict = {}
    if THINKING_BUDGET == 0:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(retries):
        content = user_content if attempt == 0 else user_content + _ACTION_ESCAPE_REMINDER
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ACTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=MAX_TOKENS,
            extra_body=extra,
        )
        raw = resp.choices[0].message.content or ""
        if not raw.strip():
            _log(f"  [action cache] empty response (attempt {attempt + 1}/{retries}), retrying…")
            time.sleep(2**attempt)
            continue
        try:
            result = json.loads(_strip_fences(raw))
            if isinstance(result, list) and len(result) == len(chunk):
                return result
            _log(f"  [action cache] wrong shape (attempt {attempt + 1}/{retries}). raw={raw!r}")
        except json.JSONDecodeError:
            _log(
                f"  [action cache] JSON parse failed (attempt {attempt + 1}/{retries}). raw={raw!r}"
            )
        time.sleep(2**attempt)

    # Chunk failed — retry each string individually so one bad string doesn't poison the rest.
    _log(
        f"  [action cache] chunk failed; falling back to per-string translation ({len(chunk)} strings)"
    )
    results = []
    for s in chunk:
        single = _translate_action_chunk(client, model, [s], retries=retries)
        results.append(single[0])
    return results


def build_action_cache(
    client: OpenAI, model: str, all_actions: list[str], chunk_size: int = 20
) -> dict[str, str]:
    """Translate unique Chinese action strings in chunks; falls back gracefully on error."""
    to_translate = [a for a in all_actions if not _ACTION_PASSTHROUGH.match(a)]
    if not to_translate:
        return {}
    cache: dict[str, str] = {}
    for i in range(0, len(to_translate), chunk_size):
        chunk = to_translate[i : i + chunk_size]
        translated = _translate_action_chunk(client, model, chunk)
        cache.update(zip(chunk, translated))
    return cache


def apply_translation(original: dict, translated: dict, action_cache: dict, model: str) -> dict:
    row = dict(original)
    row["grade"] = GRADE_MAP.get(original["grade"], original["grade"])
    row["mission"] = MISSION_MAP.get(original["mission"], original["mission"])
    row["question"] = translated["question"]
    row["options"] = translated["options"]
    row["newHint"] = translated["newHint"]
    row["newKnowledgePoint"] = translated["newKnowledgePoint"]
    row["newAnalyze"] = translated["newAnalyze"]

    translated_turns = translated.get("dialogue", [])
    new_dialogue = []
    for i, turn in enumerate(original["dialogue"]):
        new_turn = dict(turn)
        if i < len(translated_turns):
            t = translated_turns[i]
            new_turn["student"] = t.get("student", turn["student"])
            new_turn["evaluation"] = t.get("evaluation", turn["evaluation"])
            new_turn["teacher"] = t.get("teacher", turn["teacher"])
        action = turn["action"]
        new_turn["action"] = (
            action if _ACTION_PASSTHROUGH.match(action) else action_cache.get(action, action)
        )
        new_dialogue.append(new_turn)

    row["dialogue"] = new_dialogue
    row["translation_meta"] = {
        "model": model,
        "translated_at": datetime.now(UTC).isoformat(),
    }
    return row


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def _save_checkpoint(
    path: Path,
    translated_ids: set[int],
    results: list[dict],
    action_cache: dict[str, str],
) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "translated_ids": list(translated_ids),
                "results": results,
                "action_cache": action_cache,
            },
            f,
        )


def upload_checkpoint_to_hf(checkpoint_path: Path, hf_repo: str, done: int) -> None:
    """Upload the local checkpoint JSON to the HF dataset repo as a file."""
    from huggingface_hub import create_repo, upload_file  # type: ignore

    create_repo(hf_repo, repo_type="dataset", exist_ok=True)
    upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="translate_checkpoint.json",
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message=f"checkpoint: {done} records translated",
    )
    _log(f"  [HF checkpoint] {done} records → {hf_repo}/translate_checkpoint.json")


def restore_checkpoint_from_hf(checkpoint_path: Path, hf_repo: str) -> bool:
    """Download checkpoint from HF repo if it exists. Returns True if restored."""
    from huggingface_hub import hf_hub_download  # type: ignore
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError  # type: ignore

    try:
        hf_hub_download(
            repo_id=hf_repo,
            filename="translate_checkpoint.json",
            repo_type="dataset",
            local_dir=str(checkpoint_path.parent),
        )
        _log(f"Restored checkpoint from HuggingFace: {hf_repo}")
        return True
    except (EntryNotFoundError, RepositoryNotFoundError):
        _log(f"No checkpoint found on HuggingFace ({hf_repo}) — starting fresh")
        return False


# ── HuggingFace dataset upload ────────────────────────────────────────────────


def push_dataset_to_hf(results: list[dict], hf_repo: str, model: str) -> None:
    from datasets import Dataset  # type: ignore
    from huggingface_hub import create_repo  # type: ignore

    create_repo(hf_repo, repo_type="dataset", exist_ok=True)
    ds = Dataset.from_list(results)
    ds.push_to_hub(
        hf_repo,
        commit_message=f"Add {len(results)} translated records ({model})",
    )
    _log(f"Pushed {len(results)} records to https://huggingface.co/datasets/{hf_repo}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate SocratDataset.json (Chinese → English)")
    parser.add_argument(
        "--input-repo",
        default=HF_INPUT_REPO,
        help="HuggingFace source dataset repo (default: %(default)s)",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--model", default=MODEL, help="Model name as served by the local API")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        metavar="N",
        help="Translate only N records then exit (sanity check)",
    )
    parser.add_argument(
        "--ids",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[],
        metavar="ID[,ID...]",
        help="Translate only these record IDs (comma-separated), ignoring checkpoint",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip all HuggingFace uploads for this run (overrides PUSH_TO_HUB=True)",
    )
    parser.add_argument(
        "--restore-from-hub",
        action="store_true",
        help="Download checkpoint from HuggingFace before starting (to resume after machine failure)",
    )
    parser.add_argument("--hf-repo", default=HF_REPO)
    args = parser.parse_args()

    push = PUSH_TO_HUB and not args.no_push

    global _log_fh
    if LOG_PATH:
        log_path = Path(LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(log_path, "a")  # noqa: SIM115
        _log(f"=== translate_dataset started {datetime.now(UTC).isoformat()} ===")

    if args.restore_from_hub:
        restore_checkpoint_from_hf(Path(args.checkpoint), args.hf_repo)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    _log(f"Loading source dataset from HuggingFace: {args.input_repo} …")
    from datasets import load_dataset as _load_dataset  # type: ignore

    dataset: list[dict] = _load_dataset(args.input_repo, split="train").to_list()
    _log(f"Loaded {len(dataset)} records.")

    if args.ids:
        id_set = set(args.ids)
        dataset = [r for r in dataset if r["id"] in id_set]
        _log(f"ID filter: {sorted(id_set)}")
    elif args.smoke_test:
        dataset = dataset[: args.smoke_test]
        _log(f"Smoke-test mode: {args.smoke_test} records")

    # Resume from local checkpoint
    cp = Path(args.checkpoint)
    if cp.exists():
        with open(cp) as f:
            state = json.load(f)
        translated_ids: set[int] = set(state["translated_ids"])
        results: list[dict] = state["results"]
        action_cache: dict[str, str] = state.get("action_cache", {})
        _log(f"Resuming from checkpoint: {len(translated_ids)} records already done")
    else:
        translated_ids = set()
        results = []
        action_cache = {}

    # Build action lookup table; on resume, catch any strings missing from the checkpoint cache.
    all_actions = sorted({t["action"] for row in dataset for t in row["dialogue"]})
    missing = [a for a in all_actions if not _ACTION_PASSTHROUGH.match(a) and a not in action_cache]
    if missing:
        _log(f"Building action lookup table ({len(missing)} new strings) …")
        action_cache.update(build_action_cache(client, args.model, missing))
        _log(f"  {len(action_cache)} unique actions in cache")

    total = len(dataset)
    start = time.time()
    done_at_start = len(translated_ids)  # checkpoint records; excluded from rate
    last_hf_push = done_at_start
    errors = 0

    for record in dataset:
        if not args.ids and record["id"] in translated_ids:
            continue
        try:
            translated = translate_record(client, args.model, record)
            results.append(apply_translation(record, translated, action_cache, args.model))
            translated_ids.add(record["id"])
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            errors += 1
            continue

        done = len(translated_ids)
        elapsed = time.time() - start
        new_this_run = done - done_at_start
        rate = new_this_run / elapsed if elapsed > 0 else 0
        eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
        if eta_min == float("inf"):
            eta_str = "∞"
        elif eta_min >= 60:
            eta_str = f"{eta_min / 60:.1f}h"
        else:
            eta_str = f"{eta_min:.0f}m"
        _log(f"[{done:>5}/{total}] id={record['id']:>5}  {rate * 3600:.1f} rec/hr  ETA {eta_str}")

        if done % LOCAL_CHECKPOINT_EVERY == 0:
            _save_checkpoint(cp, translated_ids, results, action_cache)

        if push and HF_CHECKPOINT_EVERY > 0 and done % HF_CHECKPOINT_EVERY == 0:
            _save_checkpoint(cp, translated_ids, results, action_cache)
            upload_checkpoint_to_hf(cp, args.hf_repo, done)
            push_dataset_to_hf(results, args.hf_repo, args.model)
            last_hf_push = done

    # Final local save
    _save_checkpoint(cp, translated_ids, results, action_cache)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    _log(f"\nWrote {len(results)} records to {out}  ({errors} errors)")

    if not push:
        _log(f"Skipped HuggingFace upload (PUSH_TO_HUB={PUSH_TO_HUB}, --no-push={args.no_push})")
    elif len(results) != last_hf_push:
        push_dataset_to_hf(results, args.hf_repo, args.model)

    if _log_fh is not None:
        _log_fh.close()


if __name__ == "__main__":
    main()
