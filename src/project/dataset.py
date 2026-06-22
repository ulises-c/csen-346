"""
Unified training data loader for SocratTeachLLM v2.

Normalises all datasets to HF `messages` format, ready for
TRL SFTTrainer + apply_chat_template.

Each record:
    {
        "id":                  str,
        "source":              str,  # see _SOURCE_LOADERS for valid keys
        "messages":            [{"role": "system"|"user"|"assistant", "content": str}, ...],
        "ground_truth_states": list[str] | None,  # only for KELE state-annotated sources
    }

Stage 1 — General instruction (ulises-c/general-instruction-dataset-sft-stage-1 collection):
    "openhermes"          teknium/OpenHermes-2.5
    "ultrachat"           HuggingFaceH4/ultrachat_200k
    "slimorca"            Open-Orca/slimorca-deduped-cleaned-corrected

Stage 2 — Socratic teaching (ulises-c/socratic-teaching-datasets-sft-stage-2 collection):
    "socrat-zh"           ulises-c/SocratDataset          (original Chinese, state+action annotated)
    "socrat-en"           ulises-c/SocratDataset-EN       (English translation, state+action annotated)
    "socrat-synthetic"    ulises-c/SocratDataset-SYNTHETIC (Chinese synthetic, state annotated) — eval only
    "socrat-synthetic-en" ulises-c/SocratDataset-SYNTHETIC-EN (English synthetic, state annotated) — eval only
    "socrateach-multi"    ulises-c/SocraTeach_Multi
    "socrateach-single"   ulises-c/SocraTeach_Single
    "socratic-math"       ulises-c/SocraticMATH
    "socratic-math-sol"   ulises-c/SocraticMATH-sol

Usage:
    from src.project.dataset import load_training_data
    # Stage 1
    records = load_training_data(["openhermes", "ultrachat", "slimorca"])
    # Stage 2 full mix (socrat-synthetic / socrat-synthetic-en are eval-only)
    records = load_training_data(["socrat-zh", "socrat-en", "socrateach-multi",
                                   "socrateach-single", "socratic-math", "socratic-math-sol"])
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

_SOCRAT_ZH_SYSTEM = (
    "你是一位苏格拉底式教师。通过启发性问题引导学生自己发现答案——永远不要直接告诉学生答案。"
)

_SOCRAT_EN_SYSTEM = (
    "You are a Socratic teacher. Guide the student to discover the answer through "
    "heuristic questions — never give away the answer directly."
)

# SocraTeach datasets were generated without SocRule state annotations; softer framing
# ("guiding questions") avoids implying state-conditioned behaviour to the model.
_SOCRATEACH_SYSTEM = (
    "You are a Socratic teacher. Guide the student through this problem using "
    "guiding questions — never give away the answer directly."
)


def _strip_quotes(s: str) -> str:
    """Strip surrounding single or double quotes present in the HF SocratDataset upload."""
    s = s.strip()
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        return s[1:-1]
    return s


def _format_options(opts: list[str]) -> str:
    """Format a list of answer options as '(A) opt1 (B) opt2 ...'."""
    return " ".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))


def _split(records: list, split: str, seed: int) -> list:
    """Return the train or test subset of *records* using a seeded 90/10 shuffle-split.

    The shuffle is deterministic for a given (len(records), seed) pair, matching the
    split used by kele.load_dataset so training and evaluation sets never overlap.
    """
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split_point = int(len(records) * 0.9)
    chosen = indices[:split_point] if split == "train" else indices[split_point:]
    return [records[i] for i in sorted(chosen)]


# ---------------------------------------------------------------------------
# SocratDataset (original Chinese) — socrat-zh
# ---------------------------------------------------------------------------


def _socrat_zh_to_messages(record: dict) -> dict:
    """Convert one original Chinese SocratDataset record to messages format.

    Same schema as SocratDataset-EN but in Chinese. The local JSON serialises
    state/action/evaluation with extra surrounding quotes (e.g. "'a1'") — these
    are stripped before use. The HF version (ulises-c/SocratDataset) is clean.
    """
    q = record["question"]
    opts = record.get("options") or []
    hint = record.get("newHint") or ""
    knowledge = record.get("newKnowledgePoint") or ""

    system_parts = [_SOCRAT_ZH_SYSTEM, f"问题：{q}"]
    if opts:
        system_parts.append(f"选项：{_format_options(opts)}")
    if hint:
        system_parts.append(f"提示：{hint}")
    if knowledge:
        system_parts.append(f"知识点：{knowledge}")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    states: list[str] = []

    for turn in record.get("dialogue", []):
        state = _strip_quotes(turn.get("state", ""))
        action = _strip_quotes(turn.get("action", ""))
        user_content = turn["student"]
        if state and action:
            user_content += (
                f"\n\n苏格拉底教学顾问评估结果: 学生处于 {state} 状态\n"
                f"苏格拉底教学顾问建议的操作: {action}"
            )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": turn["teacher"]})
        if state:
            states.append(state)

    return {
        "id": str(record["id"]),
        "source": "socrat-zh",
        "messages": messages,
        "ground_truth_states": states if states else None,
    }


def load_socrat_zh(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset",
) -> list[dict]:
    """Load the original Chinese SocratDataset from HuggingFace and convert to messages format."""
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]
    converted = [_socrat_zh_to_messages(r) for r in raw]
    if split == "all":
        return converted
    return _split(converted, split, seed)


# ---------------------------------------------------------------------------
# SocratDataset-EN
# ---------------------------------------------------------------------------


def _socrat_en_to_messages(record: dict) -> dict:
    """Convert one SocratDataset-EN record to the unified messages format.

    The consultant's evaluation and action are prepended to each teacher turn
    so the fine-tuned model learns state-conditioned response generation.
    At inference time these are supplied by the live consultant agent.
    """
    q = record["question"]
    opts = record.get("options") or []
    hint = record.get("newHint") or ""
    knowledge = record.get("newKnowledgePoint") or ""

    system_parts = [_SOCRAT_EN_SYSTEM, f"Problem: {q}"]
    if opts:
        system_parts.append(f"Options: {_format_options(opts)}")
    if hint:
        system_parts.append(f"Hint: {hint}")
    if knowledge:
        system_parts.append(f"Knowledge: {knowledge}")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    states: list[str] = []

    for turn in record.get("dialogue", []):
        # HF upload of SocratDataset-EN is pre-cleaned — no surrounding quotes in
        # state/action. If testing against raw local JSON, apply _strip_quotes here too.
        state = turn.get("state", "")
        action = turn.get("action", "")
        user_content = turn["student"]
        if state and action:
            user_content += (
                f"\n\n苏格拉底教学顾问评估结果: 学生处于 {state} 状态\n"
                f"苏格拉底教学顾问建议的操作: {action}"
            )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": turn["teacher"]})
        if state:
            states.append(state)

    return {
        "id": str(record["id"]),
        "source": "socrat-en",
        "messages": messages,
        "ground_truth_states": states if states else None,
    }


def load_socrat_en(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset-EN",
) -> list[dict]:
    """Load SocratDataset-EN from HuggingFace and convert to messages format.

    Uses the same 90/10 split logic as kele.load_dataset so train/test sets
    are identical between the evaluation and training pipelines.
    """
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]
    converted = [_socrat_en_to_messages(r) for r in raw]
    if split == "all":
        return converted
    return _split(converted, split, seed)


# ---------------------------------------------------------------------------
# SocraTeach_Multi
# ---------------------------------------------------------------------------


def _socrateach_multi_record_to_messages(problem: dict, dlg: dict) -> dict:
    """Convert one SocraTeach_Multi dialogue to messages format.

    Each problem has multiple dialogues (one per student persona / path).
    We generate one training record per dialogue, not one per problem.

    Turn structure: teacher asks first (system field), student responds (user field).
    We map: system → assistant (teacher), user → user (student).
    """
    q = problem["question"]
    system_content = f"{_SOCRATEACH_SYSTEM}\nProblem: {q}"

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for turn in dlg.get("turns", []):
        teacher_q = turn.get("system", "")
        student_a = turn.get("user", "")
        if teacher_q:
            messages.append({"role": "assistant", "content": teacher_q})
        if student_a:
            messages.append({"role": "user", "content": student_a})

    return {
        "id": dlg["dialogue_id"],
        "source": "socrateach-multi",
        "messages": messages,
        "ground_truth_states": None,
    }


def load_socrateach_multi(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocraTeach_Multi",
) -> list[dict]:
    """Load SocraTeach_Multi from HuggingFace and convert to messages format.

    Returns one record per (problem, dialogue) pair.
    """
    from datasets import load_dataset as hf_load

    hf_ds = hf_load(hf_repo, split="train")

    records: list[dict] = []
    for problem in hf_ds:
        problem = dict(problem)
        for dlg in problem.get("dialogues", []):
            if isinstance(dlg, dict):
                records.append(_socrateach_multi_record_to_messages(problem, dlg))

    if split == "all":
        return records
    return _split(records, split, seed)


# ---------------------------------------------------------------------------
# SocraTeach_Single
# ---------------------------------------------------------------------------


def _socrateach_single_to_messages(record: dict) -> dict:
    """Convert one SocraTeach_Single record to messages format.

    History is a list of [user, assistant] pairs. The last exchange
    is (prompt, response). The first history entry often contains the
    system instruction as the user message — we extract it as the system role.
    """
    student_type = record.get("student_type", "")
    history = record.get("history") or []
    prompt = record.get("prompt", "")
    response = record.get("response", "")

    messages: list[dict] = []
    system_parts = [_SOCRATEACH_SYSTEM]
    if student_type:
        system_parts.append(f"Student type: {student_type}")

    # The first history pair typically contains the meta-instruction as the
    # user message (e.g. "You are a Socratic teacher, please guide me...").
    # Treat it as additional system context rather than a training turn.
    start_idx = 0
    if history and isinstance(history[0], (list, tuple)) and len(history[0]) >= 1:
        first_user = history[0][0] if history[0] else ""
        if isinstance(first_user, str) and first_user.startswith("You are a Socratic teacher"):
            system_parts.append(f"Task: {first_user[:300]}")
            start_idx = 1

    messages.append({"role": "system", "content": "\n".join(system_parts)})

    for pair in history[start_idx:]:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})

    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})

    return {
        "id": record["id"],
        "source": "socrateach-single",
        "messages": messages,
        "ground_truth_states": None,
    }


def load_socrateach_single(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocraTeach_Single",
) -> list[dict]:
    """Load SocraTeach_Single from HuggingFace and convert to messages format."""
    from datasets import load_dataset as hf_load

    hf_ds = hf_load(hf_repo, split="train")
    records = [_socrateach_single_to_messages(dict(r)) for r in hf_ds]
    if split == "all":
        return records
    return _split(records, split, seed)


# ---------------------------------------------------------------------------
# SocratDataset-SYNTHETIC (Chinese, state-annotated, no action field)
# ---------------------------------------------------------------------------


def load_socrat_synthetic(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset-SYNTHETIC",
) -> list[dict]:
    """Load SocratDataset-SYNTHETIC from HuggingFace (both configs, combined).

    Merges the `default` (37 records) and `n75_extension` (38 records) configs
    into a single pool of 75 records before splitting.
    """
    from datasets import load_dataset as hf_load

    raw_default = [dict(r) for r in hf_load(hf_repo, split="train")]
    raw_ext = [dict(r) for r in hf_load(hf_repo, name="n75_extension", split="train")]

    def _convert(r: dict) -> dict:
        q = r["question"]
        system_content = f"{_SOCRAT_ZH_SYSTEM}\n问题：{q}"
        messages: list[dict] = [{"role": "system", "content": system_content}]
        states: list[str] = []
        for turn in r.get("dialogue", []):
            messages.append({"role": "user", "content": turn["student"]})
            state = turn.get("state", "")
            teacher_content = turn["teacher"]
            if state:
                teacher_content = f"[State: {state}]\n" + teacher_content
            messages.append({"role": "assistant", "content": teacher_content})
            if state:
                states.append(state)
        return {
            "id": str(r["id"]),
            "source": "socrat-synthetic",
            "messages": messages,
            "ground_truth_states": states if states else None,
        }

    converted = [_convert(r) for r in raw_default + raw_ext]
    if split == "all":
        return converted
    return _split(converted, split, seed)


# ---------------------------------------------------------------------------
# SocratDataset-SYNTHETIC-EN (English, state-annotated, no action field)
# ---------------------------------------------------------------------------


def load_socrat_synthetic_en(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset-SYNTHETIC-EN",
) -> list[dict]:
    """Load SocratDataset-SYNTHETIC-EN from HuggingFace (75 records)."""
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]

    def _convert(r: dict) -> dict:
        q = r["question"]
        system_content = f"{_SOCRAT_EN_SYSTEM}\nProblem: {q}"
        messages: list[dict] = [{"role": "system", "content": system_content}]
        states: list[str] = []
        for turn in r.get("dialogue", []):
            messages.append({"role": "user", "content": turn["student"]})
            state = turn.get("state", "")
            teacher_content = turn["teacher"]
            if state:
                teacher_content = f"[State: {state}]\n" + teacher_content
            messages.append({"role": "assistant", "content": teacher_content})
            if state:
                states.append(state)
        return {
            "id": str(r["id"]),
            "source": "socrat-synthetic-en",
            "messages": messages,
            "ground_truth_states": states if states else None,
        }

    converted = [_convert(r) for r in raw]
    if split == "all":
        return converted
    return _split(converted, split, seed)


# ---------------------------------------------------------------------------
# SocraticMATH / SocraticMATH-sol
# ---------------------------------------------------------------------------

_SOCRATIC_MATH_SYSTEM = (
    "You are a Socratic mathematics teacher. Guide the student to discover the answer "
    "through guiding questions — never give away the answer directly."
)


def _socratic_math_to_messages(record: dict, source: str) -> dict:
    """Convert one SocraticMATH record to messages format.

    Both SocraticMATH and SocraticMATH-sol share the same `conversations` schema:
    each turn has `from` ("user" or "assistant") and `value`.  The -sol variant
    prefixes the first assistant turn with a full solution (【解析】:).
    """
    messages: list[dict] = [{"role": "system", "content": _SOCRATIC_MATH_SYSTEM}]
    for turn in record.get("conversations", []):
        role = "user" if turn["from"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return {
        "id": str(record["id"]),
        "source": source,
        "messages": messages,
        "ground_truth_states": None,
    }


def _load_socratic_math_all(hf_repo: str, source: str) -> list[dict]:
    """Load all three HF splits and combine into a single pool."""
    from datasets import load_dataset as hf_load

    all_records: list[dict] = []
    for hf_split in ("train", "validation", "test"):
        all_records.extend(
            [_socratic_math_to_messages(dict(r), source) for r in hf_load(hf_repo, split=hf_split)]
        )
    return all_records


def load_socratic_math(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocraticMATH",
) -> list[dict]:
    """Load SocraticMATH (6,846 records) from HuggingFace."""
    converted = _load_socratic_math_all(hf_repo, "socratic-math")
    if split == "all":
        return converted
    return _split(converted, split, seed)


def load_socratic_math_sol(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocraticMATH-sol",
) -> list[dict]:
    """Load SocraticMATH-sol (6,846 records with prepended solutions) from HuggingFace."""
    converted = _load_socratic_math_all(hf_repo, "socratic-math-sol")
    if split == "all":
        return converted
    return _split(converted, split, seed)


# ---------------------------------------------------------------------------
# Stage 1: General instruction datasets (ShareGPT / chat format)
# ---------------------------------------------------------------------------


def _sharegpt_to_messages(record: dict, source: str, idx: int) -> dict:
    """Convert a ShareGPT-style record (from/value turns) to messages format.

    Handles the standard ShareGPT schema used by OpenHermes-2.5 and SlimOrca:
        {"conversations": [{"from": "system"|"human"|"gpt", "value": "..."}]}
    """
    messages: list[dict] = []
    for turn in record.get("conversations", []):
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(turn.get("from", ""), "")
        if role:
            messages.append({"role": role, "content": turn.get("value", "")})
    return {
        "id": str(record.get("id", f"{source}-{idx}")),
        "source": source,
        "messages": messages,
        "ground_truth_states": None,
    }


def load_openhermes(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "teknium/OpenHermes-2.5",
) -> list[dict]:
    """Load OpenHermes-2.5 (~1M records) from HuggingFace."""
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]
    converted = [
        _sharegpt_to_messages(r, "openhermes", i)
        for i, r in enumerate(raw)
        if r.get("conversations")
    ]
    if split == "all":
        return converted
    return _split(converted, split, seed)


def load_slimorca(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "Open-Orca/slimorca-deduped-cleaned-corrected",
) -> list[dict]:
    """Load SlimOrca-deduped (~500k records) from HuggingFace."""
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]
    converted = [
        _sharegpt_to_messages(r, "slimorca", i) for i, r in enumerate(raw) if r.get("conversations")
    ]
    if split == "all":
        return converted
    return _split(converted, split, seed)


def load_ultrachat(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "HuggingFaceH4/ultrachat_200k",
) -> list[dict]:
    """Load UltraChat-200k from HuggingFace.

    Uses train_sft + test_sft HF splits as the combined pool; our own _split
    then carves out the 90/10 train/eval partition.
    """
    from datasets import load_dataset as hf_load

    raw: list[dict] = []
    for hf_split in ("train_sft", "test_sft"):
        raw.extend([dict(r) for r in hf_load(hf_repo, split=hf_split)])

    converted = [
        {
            "id": str(r.get("id", f"ultrachat-{i}")),
            "source": "ultrachat",
            "messages": r["messages"],
            "ground_truth_states": None,
        }
        for i, r in enumerate(raw)
        if r.get("messages")
    ]
    if split == "all":
        return converted
    return _split(converted, split, seed)


# ---------------------------------------------------------------------------
# Inference-matching SFT sources (socrat-zh-sft, socrat-en-sft)
# ---------------------------------------------------------------------------
# These produce one record per dialogue *turn* in the exact single-message format
# that socratic_teaching_system.py:socrates_teacher sends at inference, fixing the
# train/serve schema drift that caused output collapse on the 5090-trained model.
#
# Differences from the multi-turn socrat-zh/en sources:
#   - System prompt: the 7-line inference rules block, not _SOCRAT_ZH/EN_SYSTEM
#   - Problem context (问题/选项/提示/知识点) NOT included — inference never sends it
#   - Message structure: one (system, user, assistant) triple per turn
#   - User message: 历史对话记录 blob + 当前学生输入 label, matching socrates_teacher()
#   - History format: "学生: ...\n老师: ...\n" pairs, matching get_formatted_history()

_TEACHER_INFERENCE_SYSTEM = """\
你是一位使用苏格拉底教学法的小学科学教师，擅长启发式教学。
接下来你会收到历史对话记录、当前学生输入和苏格拉底教学顾问对当前教学对话的评估及建议操作；
你的任务是遵循建议的操作并参考评估结果对学生提问以完成苏格拉底式教学。
以下是你需要遵守的规则：
- 每次只能提出一个问题（输出时请检查问题数量，如超出请删去多余问题）
- 提出的问题必须与解题直接相关（输出时请检查问题是否偏离解题，如偏题请重新输出与解题直接相关的问题）
- 请确保提问符合小学阶段学生的知识水平，避免过于困难
- 语气应该非常亲切并具有鼓励性
- 除非苏格拉底教学顾问建议的操作要求，否则不能给出过于明显的提示
- 如果接收到的建议操作为：对题目进行总结，则总结题目且不再提出问题"""


def _build_inference_user_message(
    history_turns: list[tuple[str, str]],
    student_input: str,
    evaluation: str,
    action: str,
) -> str:
    """Build the user message exactly as socrates_teacher() does at inference.

    history_turns: list of (student, teacher) pairs for completed turns before this one.
    evaluation/action are the consultant fields as inference sends them: the
    free-form consultant prose and get_action_for_state(state) respectively.
    """
    if history_turns:
        history_lines = []
        for s, t in history_turns:
            history_lines.append(f"学生: {s}")
            history_lines.append(f"老师: {t}")
        formatted_history = "\n".join(history_lines)
    else:
        formatted_history = ""

    return (
        f"\n历史对话记录:\n{formatted_history}\n\n"
        f"当前学生输入: {student_input}\n\n"
        f"苏格拉底教学顾问评估结果: {evaluation}\n"
        f"苏格拉底教学顾问建议的操作: {action}\n"
    )


def _socrat_zh_to_sft_records(record: dict) -> list[dict]:
    """One SocratDataset-ZH dialogue → N per-turn SFT records in inference format."""
    from src.project.socratic_teaching_system import get_action_for_state

    dialogue = record.get("dialogue", [])
    records = []
    history: list[tuple[str, str]] = []

    for i, turn in enumerate(dialogue):
        state = _strip_quotes(turn.get("state", ""))
        if not (state and _strip_quotes(turn.get("action", ""))):
            history.append((turn["student"], turn["teacher"]))
            continue

        action = get_action_for_state(state)
        evaluation = _strip_quotes(turn.get("evaluation", "")) or f"学生处于 {state} 状态"
        user_msg = _build_inference_user_message(history, turn["student"], evaluation, action)
        records.append(
            {
                "id": f"{record['id']}_{i}",
                "source": "socrat-zh-sft",
                "messages": [
                    {"role": "system", "content": _TEACHER_INFERENCE_SYSTEM},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": turn["teacher"]},
                ],
                "ground_truth_states": [state],
            }
        )
        history.append((turn["student"], turn["teacher"]))

    return records


def _socrat_en_to_sft_records(record: dict) -> list[dict]:
    """One SocratDataset-EN dialogue → N per-turn SFT records in inference format."""
    from src.project.socratic_teaching_system import get_action_for_state

    dialogue = record.get("dialogue", [])
    records = []
    history: list[tuple[str, str]] = []

    for i, turn in enumerate(dialogue):
        state = turn.get("state", "")
        if not (state and turn.get("action", "")):
            history.append((turn["student"], turn["teacher"]))
            continue

        action = get_action_for_state(state)
        evaluation = turn.get("evaluation", "") or f"学生处于 {state} 状态"
        user_msg = _build_inference_user_message(history, turn["student"], evaluation, action)
        records.append(
            {
                "id": f"{record['id']}_{i}",
                "source": "socrat-en-sft",
                "messages": [
                    {"role": "system", "content": _TEACHER_INFERENCE_SYSTEM},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": turn["teacher"]},
                ],
                "ground_truth_states": [state],
            }
        )
        history.append((turn["student"], turn["teacher"]))

    return records


def load_socrat_zh_sft(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset",
) -> list[dict]:
    """Load SocratDataset-ZH as per-turn inference-matching SFT records.

    Splits at the dialogue level before expanding to per-turn records so that
    all turns of a dialogue stay in the same partition. Splitting on the flat
    per-turn list would scatter a dialogue's turns across train/test, causing
    history blobs in test records to contain teacher responses seen in training.
    """
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]
    if split == "all":
        converted: list[dict] = []
        for r in raw:
            converted.extend(_socrat_zh_to_sft_records(r))
        return converted
    split_raw = _split(raw, split, seed)
    converted = []
    for r in split_raw:
        converted.extend(_socrat_zh_to_sft_records(r))
    return converted


def load_socrat_en_sft(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset-EN",
) -> list[dict]:
    """Load SocratDataset-EN as per-turn inference-matching SFT records.

    Splits at the dialogue level before expanding to per-turn records — same
    reasoning as load_socrat_zh_sft.
    """
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]
    if split == "all":
        converted: list[dict] = []
        for r in raw:
            converted.extend(_socrat_en_to_sft_records(r))
        return converted
    split_raw = _split(raw, split, seed)
    converted = []
    for r in split_raw:
        converted.extend(_socrat_en_to_sft_records(r))
    return converted


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------

_SOURCE_LOADERS = {
    # Stage 1 — general instruction
    "openhermes": load_openhermes,
    "ultrachat": load_ultrachat,
    "slimorca": load_slimorca,
    # Stage 2 — Socratic teaching (multi-turn, legacy format)
    "socrat-zh": load_socrat_zh,
    "socrat-en": load_socrat_en,
    # Stage 2 — inference-matching per-turn SFT format (use these for SFT training)
    "socrat-zh-sft": load_socrat_zh_sft,
    "socrat-en-sft": load_socrat_en_sft,
    "socrat-synthetic": load_socrat_synthetic,
    "socrat-synthetic-en": load_socrat_synthetic_en,
    "socrateach-multi": load_socrateach_multi,
    "socrateach-single": load_socrateach_single,
    "socratic-math": load_socratic_math,
    "socratic-math-sol": load_socratic_math_sol,
}


def _validate_sources(sources: list[str]) -> None:
    unknown = set(sources) - set(_SOURCE_LOADERS)
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}. Valid: {set(_SOURCE_LOADERS)}")


def load_training_data(
    sources: list[str] | None = None,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Load and combine training records from one or more sources.

    Downloads each source concurrently; result order matches `sources`.

    Args:
        sources: List of source keys to include. Defaults to all known sources.
            See module docstring for the full list of valid keys.
        split: "train", "test", or "all". Applied independently to each source.
        seed: Random seed for reproducible splits (same seed across all sources).

    Returns:
        Combined list of records in messages format, one record per dialogue.
    """
    if sources is None:
        sources = list(_SOURCE_LOADERS.keys())
    _validate_sources(sources)

    def _load(src: str) -> tuple[str, list[dict]]:
        return src, _SOURCE_LOADERS[src](split=split, seed=seed)  # type: ignore[call-arg]

    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        for src, records in (
            f.result() for f in as_completed({pool.submit(_load, s): s for s in sources})
        ):
            results[src] = records

    return [r for src in sources for r in results[src]]


def load_split_pair(
    sources: list[str] | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load train and eval splits in a single HF download pass per source.

    Equivalent to calling load_training_data twice with split="train" and "test",
    but downloads each dataset once, halving network I/O when both splits are needed.

    Returns:
        (train_records, eval_records) with the same per-source 90/10 split as
        load_training_data. Result order within each list matches `sources`.
    """
    if sources is None:
        sources = list(_SOURCE_LOADERS.keys())
    _validate_sources(sources)

    def _load(src: str) -> tuple[str, list[dict], list[dict]]:
        all_recs = _SOURCE_LOADERS[src](split="all", seed=seed)  # type: ignore[call-arg]
        return src, _split(all_recs, "train", seed), _split(all_recs, "test", seed)

    train_by_src: dict[str, list[dict]] = {}
    eval_by_src: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        for src, train_recs, eval_recs in (
            f.result() for f in as_completed({pool.submit(_load, s): s for s in sources})
        ):
            train_by_src[src] = train_recs
            eval_by_src[src] = eval_recs

    train_records = [r for src in sources for r in train_by_src[src]]
    eval_records = [r for src in sources for r in eval_by_src[src]]
    return train_records, eval_records
