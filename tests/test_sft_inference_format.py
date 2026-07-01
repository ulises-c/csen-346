"""Train/serve format-parity tests for the inference-matching SFT sources.

Background (PR #101, issue #94): the first Stage-2b SFT model collapsed into a
non-terminating repetition loop at eval. Root cause was a train/serve schema
mismatch — the multi-turn `socrat-zh/en` training format looked nothing like the
single-message prompt `SocraticTeachingSystem.socrates_teacher()` builds at
inference. The `socrat-zh-sft` / `socrat-en-sft` sources were added to emit one
`(system, user, assistant)` triple per dialogue turn in that exact inference
shape.

These tests are the durable, faithful version of the ad-hoc "render-diff" check
quoted in the PR. They capture the *real* prompt the inference path constructs
(by intercepting the teacher call) and assert the SFT record renders identically
under the training chat template — so any future drift on either side breaks CI
instead of silently re-triggering the collapse.

Chat-template note: the Gemma 4 training template applies `{{ content | trim }}`
to every message (scripts/train_sft.py), and the serving template trims too.
`jinja2.trim` == Python `str.strip()`, so two messages render identically iff
their `(role, content.strip())` match. We model render-equivalence with that
rule — no tokenizer / model download required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock data builders (mirror tests/test_dataset.py conventions — no HF network)
# ---------------------------------------------------------------------------


def _mock_hf(records: list[dict]) -> MagicMock:
    m = MagicMock()
    m.__iter__ = MagicMock(side_effect=lambda: iter(records))
    return m


def _dialogue_record(id_: int, n_turns: int = 2) -> dict:
    """SocratDataset-shaped record with `n_turns` fully-annotated turns."""
    return {
        "id": id_,
        "question": f"Q{id_}",
        "options": ["opt1", "opt2"],
        "newHint": "hint",
        "newKnowledgePoint": "kp",
        "dialogue": [
            {
                "student": f"学生输入{t}",
                "teacher": f"老师回复{t}",
                "state": ("a1", "b2", "c8", "d33")[t % 4],
                "action": f"action_{t}",
                "evaluation": f"评估{t}",
            }
            for t in range(n_turns)
        ],
    }


# ---------------------------------------------------------------------------
# Faithful inference-prompt capture
# ---------------------------------------------------------------------------


def _capture_inference_prompt(
    history_pairs: list[tuple[str, str]],
    student_input: str,
    evaluation: str,
    state: str,
    action: str,
) -> dict:
    """Return the (system, user) prompt socrates_teacher() actually sends.

    Drives the real production code path: builds the same conversation history,
    then intercepts `call_teacher_wrapped` (the seam right after apply_pre_call,
    so the captured strings are exactly what hits the teacher model) instead of
    making a network call.
    """
    from src.project.socratic_teaching_system import SocraticTeachingSystem

    system = SocraticTeachingSystem(
        consultant_api_key="test",
        consultant_base_url="http://localhost:0",
        consultant_model_name="m",
        teacher_api_key="test",
        teacher_base_url="http://localhost:0",
        teacher_model_name="m",
    )
    for s, t in history_pairs:
        system.add_to_history("student", s)
        system.add_to_history("teacher", t)
    # The current student turn is added (unpaired) before the teacher call,
    # exactly as process_student_input does in production.
    system.add_to_history("student", student_input)
    system.current_state = state

    captured: dict = {}

    def fake_call(client, model, system_prompt, user_input, predicted_state):  # noqa: ANN001
        captured["system"] = system_prompt
        captured["user"] = user_input
        return "captured"

    with patch(
        "src.project.tournament_utilizations.call_teacher_wrapped",
        side_effect=fake_call,
    ):
        system.socrates_teacher(student_input, evaluation, action)

    return captured


def _render_eq(content_a: str, content_b: str) -> bool:
    """True iff the two message contents render identically under the chat
    template's per-message `| trim` (== Python str.strip())."""
    return content_a.strip() == content_b.strip()


# ===========================================================================
# Loader unit tests — socrat-zh-sft / socrat-en-sft
# ===========================================================================


@pytest.mark.parametrize(
    ("loader_name", "source"),
    [("load_socrat_zh_sft", "socrat-zh-sft"), ("load_socrat_en_sft", "socrat-en-sft")],
)
def test_sft_loader_one_record_per_turn(loader_name, source):
    import src.project.dataset as ds

    raw = [_dialogue_record(1, n_turns=3)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = getattr(ds, loader_name)(split="all")

    # 3 fully-annotated turns -> 3 per-turn records
    assert len(records) == 3
    for i, r in enumerate(records):
        assert r["source"] == source
        assert r["id"] == f"1_{i}"
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert r["ground_truth_states"] == [("a1", "b2", "c8", "d33")[i % 4]]


def test_sft_system_is_the_inference_rules_block():
    import src.project.dataset as ds

    raw = [_dialogue_record(1)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = ds.load_socrat_zh_sft(split="all")

    system = records[0]["messages"][0]["content"]
    assert system == ds._TEACHER_INFERENCE_SYSTEM
    # Hallmarks of the 7-line inference block — NOT the legacy 1-line system.
    assert system.startswith("你是一位使用苏格拉底教学法的小学科学教师")
    assert "以下是你需要遵守的规则：" in system
    # Problem context must NOT leak in — inference never sends it.
    assert "问题：Q1" not in system
    assert "选项：" not in system


def test_sft_user_message_structure_and_history_accumulation():
    import src.project.dataset as ds

    raw = [_dialogue_record(7, n_turns=3)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = ds.load_socrat_zh_sft(split="all")

    first_user = records[0]["messages"][1]["content"]
    second_user = records[1]["messages"][1]["content"]

    # Required inference labels, in order. Post Phase-0 the eval line carries the
    # dataset's free-form `evaluation` field and the action line carries the
    # canonical map output get_action_for_state("a1") — NOT the raw dataset action.
    assert "历史对话记录:" in first_user
    assert "当前学生输入: 学生输入0" in first_user
    assert "苏格拉底教学顾问评估结果: 评估0" in first_user
    assert "苏格拉底教学顾问建议的操作: 生成一个与解题相关的子问题" in first_user

    # Turn 0 has empty history; turn 1 carries turn 0's student+teacher verbatim.
    assert "学生: 学生输入0" not in first_user
    assert "学生: 学生输入0" in second_user
    assert "老师: 老师回复0" in second_user
    assert "当前学生输入: 学生输入1" in second_user


def test_sft_assistant_target_is_clean_teacher_turn():
    import src.project.dataset as ds

    raw = [_dialogue_record(1)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = ds.load_socrat_zh_sft(split="all")

    for i, r in enumerate(records):
        asst = r["messages"][2]["content"]
        assert asst == f"老师回复{i}"
        # No consultant markers must leak into the loss target.
        assert "苏格拉底教学顾问" not in asst
        assert not asst.startswith("[State:")


def test_sft_skips_turns_without_state_action_but_keeps_them_in_history():
    import src.project.dataset as ds

    record = {
        "id": 1,
        "dialogue": [
            {"student": "S0", "teacher": "T0", "state": "a1", "action": "act0"},
            {"student": "S1", "teacher": "T1"},  # no state/action -> not a target
            {"student": "S2", "teacher": "T2", "state": "b2", "action": "act2"},
        ],
    }
    with patch("datasets.load_dataset", return_value=_mock_hf([record])):
        records = ds.load_socrat_zh_sft(split="all")

    # Only turns 0 and 2 become training targets.
    assert [r["id"] for r in records] == ["1_0", "1_2"]
    # But the skipped turn 1 still appears in turn 2's history blob.
    third_user = records[1]["messages"][1]["content"]
    assert "学生: S1" in third_user and "老师: T1" in third_user
    assert "当前学生输入: S2" in third_user


def test_sft_split_keeps_a_dialogue_whole_no_history_contamination():
    """Regression for the d04bc83 review fix: splitting the flat per-turn list
    scattered a dialogue's turns across train/test, leaking turn-N teacher
    targets into a test record's history blob. Split must be dialogue-level."""
    import src.project.dataset as ds

    raw = [_dialogue_record(i, n_turns=2) for i in range(40)]
    with patch("datasets.load_dataset", side_effect=lambda *a, **k: _mock_hf(raw)):
        train = ds.load_socrat_zh_sft(split="train", seed=42)
        test = ds.load_socrat_zh_sft(split="test", seed=42)

    def dlg_ids(records):
        return {r["id"].rsplit("_", 1)[0] for r in records}

    assert len(train) + len(test) == 80  # 40 dialogues x 2 turns
    assert dlg_ids(train).isdisjoint(dlg_ids(test)), (
        "a dialogue's turns must not span train/test — would contaminate eval history"
    )


def test_sft_sources_registered_in_unified_entry_point():
    import src.project.dataset as ds

    assert "socrat-zh-sft" in ds._SOURCE_LOADERS
    assert "socrat-en-sft" in ds._SOURCE_LOADERS


# ===========================================================================
# Render-diff gate — SFT record vs the REAL inference prompt
# ===========================================================================


def test_sft_record_renders_identically_to_inference_prompt():
    """The core fix: an SFT training record must render byte-for-byte identically
    (under the chat template's per-message trim) to the prompt socrates_teacher()
    sends at inference, when the consultant fields match.

    Post Phase-0 (PR #101) the action dimension is un-rigged: the loader sources
    the action from get_action_for_state(state) — exactly what process_student_input
    feeds socrates_teacher — instead of the raw dataset action. So when the dataset
    `evaluation` matches the live consultant eval, parity is total: system AND user
    render-equal, action line included, zero drift."""
    import src.project.dataset as ds
    from src.project.socratic_teaching_system import get_action_for_state

    s0, t0 = "学生输入0", "老师回复0"
    s1, state = "学生输入1", "b2"
    eval1 = "学生展示了部分理解，但需要进一步引导其推理。"

    record = {
        "id": 99,
        "dialogue": [
            {
                "student": s0,
                "teacher": t0,
                "state": "a1",
                "action": "action_0",
                "evaluation": "学生刚刚提出了问题。",
            },
            {
                "student": s1,
                "teacher": "老师回复1",
                "state": state,
                "action": "action_1",
                "evaluation": eval1,
            },
        ],
    }
    with patch("datasets.load_dataset", return_value=_mock_hf([record])):
        records = ds.load_socrat_zh_sft(split="all")

    sft_msgs = records[1]["messages"]  # the turn-1 training example
    sft_system = sft_msgs[0]["content"]
    sft_user = sft_msgs[1]["content"]

    # Inference sends the canonical action for the state and the live consultant's
    # free-form eval. Feed the same eval text the dataset carries so the only thing
    # under test is structural parity — including the now-matched action line.
    inf = _capture_inference_prompt(
        history_pairs=[(s0, t0)],
        student_input=s1,
        evaluation=eval1,
        state=state,
        action=get_action_for_state(state),
    )

    assert _render_eq(sft_system, inf["system"]), (
        f"system drift:\n  SFT: {sft_system!r}\n  INF: {inf['system']!r}"
    )
    assert _render_eq(sft_user, inf["user"]), (
        f"user drift:\n  SFT: {sft_user!r}\n  INF: {inf['user']!r}"
    )


def test_inference_system_block_is_trim_equivalent_despite_raw_whitespace():
    """The inference system string carries a leading newline + trailing indent
    that the SFT constant omits. Document that this is immaterial: the chat
    template trims it away, so the two are render-equivalent. (Guards against a
    'fix' that adds raw whitespace and introduces asymmetric drift.)"""
    inf = _capture_inference_prompt(
        history_pairs=[],
        student_input="S",
        evaluation="学生处于 a1 状态",
        state="a1",
        action="act",
    )
    import src.project.dataset as ds

    raw_inf_system = inf["system"]
    assert raw_inf_system != ds._TEACHER_INFERENCE_SYSTEM, "expected raw whitespace difference"
    assert _render_eq(raw_inf_system, ds._TEACHER_INFERENCE_SYSTEM), "trim must reconcile them"


def test_model_turn_terminator_is_inside_the_loss_mask():
    """Regression for the EOS-collapse root cause (PR #101 / issue #94).

    `assistant_only_loss=True` masks every token *outside* the
    `{% generation %}` block to -100. If the model-role terminator `<turn|>`
    renders outside that block it receives zero gradient and the model never
    learns to stop — the non-terminating repetition collapse seen on both the
    5090 and R9700 runs. TRL's own gemma3_training.jinja keeps `<end_of_turn>`
    inside the block for exactly this reason.

    Renders the real training template through transformers' jinja machinery
    (no tokenizer/model download) and asserts the assistant loss span covers the
    terminator. TRL's guard only checks that `{% generation %}` is *present*, not
    that it is placed correctly, so this is the test that actually pins it.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import train_sft
    from transformers.utils.chat_template_utils import render_jinja_template

    conversation = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "老师回复"},
    ]
    rendered, generation_indices = render_jinja_template(
        conversations=[conversation],
        chat_template=train_sft._GEMMA4_TRAINING_CHAT_TEMPLATE,
        return_assistant_tokens_mask=True,
        bos_token="<bos>",
    )
    text, spans = rendered[0], generation_indices[0]

    assert len(spans) == 1, f"expected exactly one assistant loss span, got {spans}"
    in_mask = text[spans[0][0] : spans[0][1]]
    assert in_mask == "老师回复<turn|>\n", (
        "the assistant loss span must be exactly content + terminator; "
        f"got {in_mask!r} — terminator masking is broken"
    )
    # The terminator must be trained...
    assert "<turn|>" in in_mask, (
        "model-turn terminator <turn|> is OUTSIDE the loss mask — the model "
        "will never learn to emit EOS (non-terminating repetition collapse)"
    )
    # ...and the prompt-cue header must NOT be (it is not generated by the model).
    assert "<|turn>model" not in in_mask


def test_evaluation_line_is_the_only_residual_drift_under_freeform_consultant():
    """The sole remaining residual after Phase 0 (PR #101): training carries the
    dataset's *stored* `evaluation` annotation, but at inference the live consultant
    regenerates free-form prose on that line, so the two seldom match token-for-token.
    The action line no longer drifts (both sides use get_action_for_state). This test
    pins the residual to *exactly one line* — if a future change reintroduces drift on
    any OTHER line (e.g. the action line), this breaks, distinguishing a new regression
    from the already-accepted, irreducible evaluation-line gap.
    """
    import src.project.dataset as ds
    from src.project.socratic_teaching_system import get_action_for_state

    s0, t0 = "学生输入0", "老师回复0"
    s1, state = "学生输入1", "b2"
    stored_eval = "数据集中标注的评估文本。"

    record = {
        "id": 99,
        "dialogue": [
            {
                "student": s0,
                "teacher": t0,
                "state": "a1",
                "action": "action_0",
                "evaluation": "学生刚刚提出了问题。",
            },
            {
                "student": s1,
                "teacher": "老师回复1",
                "state": state,
                "action": "action_1",
                "evaluation": stored_eval,
            },
        ],
    }
    with patch("datasets.load_dataset", return_value=_mock_hf([record])):
        records = ds.load_socrat_zh_sft(split="all")
    sft_user = records[1]["messages"][1]["content"]

    freeform_eval = "学生已理解基本概念，但在具体应用时仍有困难，建议通过追问巩固其推理过程。"
    inf = _capture_inference_prompt(
        history_pairs=[(s0, t0)],
        student_input=s1,
        evaluation=freeform_eval,
        state=state,
        action=get_action_for_state(state),
    )

    sft_lines = sft_user.strip().splitlines()
    inf_lines = inf["user"].strip().splitlines()
    assert len(sft_lines) == len(inf_lines), "line-count drift beyond the evaluation line"

    diffs = [(a, b) for a, b in zip(sft_lines, inf_lines, strict=True) if a != b]
    assert len(diffs) == 1, f"expected exactly one drifting line, got {diffs}"
    sft_line, inf_line = diffs[0]
    assert sft_line == f"苏格拉底教学顾问评估结果: {stored_eval}"
    assert inf_line == f"苏格拉底教学顾问评估结果: {freeform_eval}"
