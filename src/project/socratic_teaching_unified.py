"""
Unified Socratic teaching system — single LLM call per turn.

Replaces the two-call (consultant + teacher) flow with a single structured-output
call that emits {evaluation, state, teacher_response} together.

See docs/SOCRATIC_FUSION_PLAN.md for design and rationale. Empirically motivated
by docs/QWEN_LOCAL_EXPLORATION_LOG.md — the per-turn wall clock is dominated by
the count of LLM calls and the teacher generation, so collapsing the two-call
chain into one is the structural lever for ~33–40% throughput gains.

Falls back to the parent two-call path on schema/parse failure, so a unified
run never gets worse than the two-call run for any individual turn.
"""

import json
import os
import time
from typing import Any

# Optional few-shot exemplars to nudge the teacher toward terse, single-question
# Chinese phrasing matching the SocratDataset ground truth. Drawn from train-split
# dialogues 1, 2, 3 (all non-overlapping with the test split) for diversity
# across distinct question types; covers stages b, c, d.
_FEW_SHOT_TEACHER_BLOCK = """---

# 第四部分：教师风格示例
以下是优秀教师回复的示例。请注意它们的简洁、亲切、且每条仅含一个问题：

示例1（b阶段，state=b6）:
学生: 我觉得种子可能在花里面。
老师: 嗯，花是植物的重要部分，它确实和种子的产生有关。那么你还记得花的哪个部分会变成种子或者保护种子吗？

示例2（c阶段，state=c12）:
学生: 植物通过根从土壤中吸收水和养分。
老师: 植物除了从土壤中吸收养分外，还需要阳光来进行光合作用。你知道蘑菇是否需要阳光来获取养分吗？

示例3（d阶段，state=d33）:
学生: 光合作用可以让植物制造食物。
老师: 非常好！所以，结合我们讨论的内容，你认为植物的叶子从新芽到落叶，是一个生命历程吗？

请严格遵循这种风格：开头不要长篇铺垫，只问一个有针对性的问题，语气亲切。
"""

import openai

from src.project.socratic_teaching_system import SocraticTeachingSystem

# Strict JSON schema for the unified call's output.
# llama.cpp accepts this via response_format and uses GBNF-constrained
# decoding to make the structured output deterministic. State enum is the
# source of truth — derived from the keys of state_to_action.
SOCRATIC_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evaluation", "state", "teacher_response"],
    "properties": {
        "evaluation": {
            "type": "string",
            "minLength": 1,
        },
        "state": {
            "type": "string",
            "enum": [
                "a0",
                "a1",
                "b2",
                "b3",
                "b4",
                "b5",
                "b6",
                "b7",
                "c8",
                "c9",
                "c10",
                "c11",
                "c12",
                "c13",
                "c14",
                "c15",
                "c16",
                "c17",
                "c18",
                "c19",
                "c20",
                "c21",
                "c22",
                "c23",
                "c24",
                "c25",
                "c26",
                "c27",
                "c28",
                "c29",
                "d30",
                "d31",
                "d32",
                "d33",
                "e34",
            ],
        },
        "teacher_response": {
            "type": "string",
            "minLength": 1,
        },
    },
}


class SocraticTeachingSystemUnified(SocraticTeachingSystem):
    """Single-call variant of SocraticTeachingSystem.

    Inherits all session/history machinery from the parent. Overrides
    process_student_input to use a single structured-output LLM call
    that produces (evaluation, state, teacher_response) together.

    Phase regression prevention, teaching-round counter, and forced d33/e34
    transitions are applied post-call as Python glue (same as the parent).
    Per the fusion plan §6 Option A, when glue overrides the model's
    emitted state we trust the model's teacher_response unchanged and log
    the divergence.

    On any unified-call failure (schema parse, API error, empty output),
    falls back to the parent's two-call process_student_input — guaranteeing
    a unified run never produces a strictly worse turn than two-call.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Both teacher and consultant URLs/keys point at the same llama.cpp
        # server in the qwen-local configs, so picking one is arbitrary.
        # Use the consultant client — its model_name and max_tokens are the
        # ones we want for structured-output decoding.
        self.unified_client = self.consultant_client
        self._unified_fallback_count = 0  # tracked for reporting
        self._last_thinking_content: str | None = None  # set each turn; None when not thinking

    def _build_unified_system_prompt(self) -> str:
        """Assemble the three-section unified prompt.

        Section 1: consultant rules (5 stages, 34 states, transition rules)
        Section 2: state-action map (the "lookup" the model would otherwise
                   need a separate call for)
        Section 3: teacher generation rules
        Section 4: output schema description
        """
        state_action_lines = "\n".join(
            f"{state}：{action}" for state, action in self.state_to_action.items()
        )

        # Opt-in few-shot exemplars: gated on KELE_FEW_SHOT_TEACHER=1.
        # When unset, few_shot_block is empty and the prompt is identical to
        # the pre-existing fusion-think configuration (no behavior change).
        few_shot_block = (
            _FEW_SHOT_TEACHER_BLOCK if os.environ.get("KELE_FEW_SHOT_TEACHER") == "1" else ""
        )

        return f"""# 角色指令
你同时担任**苏格拉底教学顾问**和**苏格拉底教师**两个角色。
对每一轮学生输入，你需要：
  1. 作为顾问，分析当前对话状态并选定阶段和状态编号
  2. 作为教师，根据所选状态对应的教学操作生成回复

最终以单一的JSON输出同时呈现两个角色的工作。

---

# 第一部分：顾问职责

你作为苏格拉底教学顾问，需严格遵循五阶段苏格拉底教学法进行对话管理。每次响应必须完成：
1. 判断学生是否已提出明确问题（如未提出保持在a0状态）
2. 分析对话历史
   - 记录连续正确回答次数及相同状态持续次数
   - 追踪当前阶段已持续轮数
   - 关注当前教学总轮数(上限为{self.max_teaching_rounds}轮)
3. 确定当前教学阶段
4. 在对应教学阶段内评估学生状态
5. 检查响应结果是否符合阶段管理与转换规则

## 阶段管理与转换规则
▲ 基本规则：
   - 学生提出具体问题后（进入a1状态）才正式进入教学阶段
   - 教学阶段严格递进顺序：a → b → c → d → e（禁止跳级/回退）
   - 教学阶段的最大对话轮数：{self.max_teaching_rounds}轮（从a1开始计数）
   - 答案规范：仅在d阶段可要求答案，获得正确答案后必须进入e阶段

▲ 阶段推进规则（满足任一条件即可推进）：
   - 当学生连续两次正确回答问题时，必须考虑进入下一阶段
   - 当同一状态连续出现超过2轮对话时，应评估并推进至新状态
   - 在b阶段停留不应超过3轮对话，超过应进入c阶段
   - 在c阶段停留不应超过5轮对话，超过应进入d阶段
   - 在d阶段停留不应超过3轮对话，超过应进入e阶段
   - 当前阶段问题已充分探讨但学生未取得突破时，应转入下一阶段

▲ 阶段推进建议：
   - 优先考虑阶段推进，而非机械地停留在某状态
   - 在同一状态不得重复停留
   - 遇到边界情况，应倾向于推进到下一阶段而非反复探讨
   - b阶段和d阶段建议轮数为1至2轮

---

## 阶段详解

### 阶段a：学生提问（单回合）
**状态定义**
a0：学生尚未提出问题
a1：学生提出问题

**转换规则**
- a0 → a0：学生仍未提出明确问题
- a0 → a1：学生提出明确问题
- a1 → 自动转入阶段b（仅1轮对话）

---

### 阶段b：概念探查（了解学生概念掌握程度）
**状态评估规则**
必须遍历b2-b7，选择最匹配状态：

| 状态编号 | 触发条件 |
|----------|------------------------------|
| b2       | 没有可用策略且问题的调查不完整 |
| b3       | 没有可用策略且问题已经被调查 |
| b4       | 学生的某个概念严重错误且存在相关的子问题 |
| b5       | 想要验证学生是否真正理解该概念 |
| b6       | 学生的练习和回答是错误的 |
| b7       | 学生在已学过的概念上出错 |

---

### 阶段c：归纳推理（找出学生归纳的规则，分析规则的正确性，并确定原理，此为主要对话阶段）
**状态评估规则**
必须遍历c8-c29，选择最匹配状态：

| 状态编号 | 触发条件 |
|----------|------------------------------|
| c8       | 学生产生不完整或不一致的预测 |
| c9       | 学生的回答是错误的 |
| c10      | 学生的回答与他们已学过的概念不一致 |
| c11      | 学生提出不相关因素 |
| c12      | 学生的解释不完整 |
| c13      | 老师提出的误导性问题成功误导学生 |
| c14      | 出现新情境 |
| c15      | 练习的是学生已熟悉的概念 |
| c16      | 学生理解了自己的错误 |
| c17      | 学生忽略了一个关键点且存在可用的子问题 |
| c18      | 学生忽略了一个关键点但没有可用的子问题 |
| c19      | 学生的误解类型不明确 |
| c20      | 学生做出错误预测 |
| c21      | 学生无法预测 |
| c22      | 学生正确回答问题 |
| c23      | 学生形成了部分假设 |
| c24      | 学生提出了假设且有经验 |
| c25      | 学生提出了假设但没有经验 |
| c26      | 学生无法检验所提出的假设但有经验 |
| c27      | 学生无法检验所提出的假设且缺乏经验 |
| c28      | 学生检验所提出的假设出现错误但有经验 |
| c29      | 学生检验所提出的假设出现错误且缺乏经验 |

---

### 阶段d：规则建构（帮助学生建立新规则并要求学生应用这些规则）

| 状态编号 | 触发条件 |
|----------|------------------------------|
| d30      | 老师想检查学生是否真正理解 |
| d31      | 经过辩证过程后学生仍未理解某个概念 |
| d32      | 学生已经调查某个问题 |
| d33      | 所有概念都已被研究 |

**强制要求**
仅在本阶段可获取正确答案，学生给出正确答案后必须进入e阶段

---

### 阶段e：老师总结
**状态定义**
e34：学生正确给出题目答案

---

# 第二部分：状态-操作映射
（你判断的state决定了你作为教师必须执行的教学操作）

{state_action_lines}

---

# 第三部分：教师职责

你作为苏格拉底教学法的小学科学教师，需根据所选state对应的教学操作生成对学生的回复，遵循以下规则：
- 每次只能提出一个问题（输出时请检查问题数量，如超出请删去多余问题）
- 提出的问题必须与解题直接相关（输出时请检查问题是否偏离解题）
- 请确保提问符合小学阶段学生的知识水平，避免过于困难
- 语气应该非常亲切并具有鼓励性
- 除非所选状态对应的操作要求，否则不能给出过于明显的提示
- 如果state=e34（操作为：对题目进行总结），则总结题目且不再提出问题
{few_shot_block}
---

# 输出格式
仅输出一个JSON对象，包含以下三个字段：
{{
    "evaluation": "顾问分析：判断当前对话所处的阶段（a/b/c/d/e）和学生状态及原因",
    "state": "34个状态编号之一（a0/a1/b2-b7/c8-c29/d30-d33/e34）",
    "teacher_response": "教师对学生的回复，符合state对应的教学操作"
}}
"""

    def unified_socratic_step(self, student_input: str) -> dict[str, Any] | None:
        """Single LLM call: classify state + generate teacher response.

        Returns the parsed JSON on success. Returns None on parse / schema /
        API failure — caller should fall back to two-call path.
        """
        system_prompt = self._build_unified_system_prompt()
        user_input = (
            f"历史对话记录:\n{self.get_full_formatted_history()}\n\n当前学生输入: {student_input}\n"
        )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "socratic_step",
                "schema": SOCRATIC_STEP_SCHEMA,
                "strict": True,
            },
        }

        # When thinking is enabled with a budget, max_tokens must cover both
        # the thinking block and the JSON output. Otherwise use a 16K floor.
        if self.consultant_thinking_budget > 0:
            max_tokens = self.consultant_thinking_budget + max(self.consultant_max_tokens, 4096)
        else:
            max_tokens = max(self.consultant_max_tokens, 16384)

        extra: dict = {}
        if self.consultant_disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.consultant_thinking_budget > 0:
            extra["chat_template_kwargs"] = {"thinking_budget": self.consultant_thinking_budget}

        # Retry on rate limits (mirrors socratic_teaching_consultant pattern)
        response = None
        for attempt in range(6):
            try:
                response = self.unified_client.chat.completions.create(
                    model=self.consultant_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    response_format=response_format,  # type: ignore[reportArgumentType]
                    max_tokens=max_tokens,
                    extra_body=extra or None,
                )
                break
            except openai.RateLimitError as rle:
                retry_after: float | None = None
                try:
                    ra = (
                        rle.response.headers.get("retry-after")
                        if getattr(rle, "response", None)
                        else None
                    )
                    if ra:
                        retry_after = float(ra)
                except Exception:
                    pass
                wait = retry_after if retry_after is not None else min(2**attempt, 30)
                print(f"Unified call rate-limited (attempt {attempt + 1}/6), sleeping {wait:.1f}s")
                time.sleep(wait)
            except Exception as exc:
                print(f"Unified call API error: {exc}")
                return None

        if response is None:
            print("Unified call exhausted retries on rate limits")
            return None

        raw_content = response.choices[0].message.content
        if not raw_content:
            print("Unified call returned empty content")
            return None

        # Extract thinking content: llama.cpp may expose it in reasoning_content
        # (separate field) or inline as <think>...</think> before the JSON.
        thinking_content: str | None = None
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            thinking_content = reasoning
        elif raw_content.lstrip().startswith("<think>"):
            end_tag = raw_content.find("</think>")
            if end_tag != -1:
                start_tag = raw_content.index("<think>") + len("<think>")
                thinking_content = raw_content[start_tag:end_tag].strip()
                raw_content = raw_content[end_tag + len("</think>") :].strip()

        # Strip markdown code fences if the server wrapped them despite the
        # schema (defensive — shouldn't happen with json_schema mode but free
        # to add).
        stripped = raw_content.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            stripped = stripped[len("```json") : -len("```")].strip()
        elif stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped[len("```") : -len("```")].strip()

        try:
            result = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(f"Unified call JSON parse failure: {exc}")
            print(f"Raw content (first 500 chars): {raw_content[:500]}")
            return None

        # Defensive validation on top of the schema — guard against API quirks
        if not isinstance(result.get("state"), str):
            print(f"Unified call: state field missing or non-string: {result.get('state')!r}")
            return None
        if not isinstance(result.get("teacher_response"), str) or not result["teacher_response"]:
            print("Unified call: teacher_response missing or empty")
            return None
        if not isinstance(result.get("evaluation"), str):
            result["evaluation"] = ""

        result["thinking_content"] = thinking_content
        return result

    def process_student_input(self, student_input: str) -> str:
        """Override: try unified call first; fall back to parent's two-call on failure.

        Glue logic (regression prevention, round counter, forced d33/e34) is
        replicated from the parent — applied post-call. Per fusion plan §6
        Option A, if glue overrides the model's emitted state, the
        teacher_response stays as the model emitted it (we trust the response
        even if it nominally mismatches the corrected action).
        """
        # IMPORTANT: do not modify history before unified attempt — failure
        # path delegates to super().process_student_input which will add the
        # student utterance itself.
        unified_result: dict[str, Any] | None = None
        try:
            unified_result = self.unified_socratic_step(student_input)
        except Exception as exc:
            print(f"Unified call unexpected exception: {exc}")
            unified_result = None

        if unified_result is None:
            self._last_thinking_content = None
            self._unified_fallback_count += 1
            print(
                f"Unified call failed — falling back to two-call "
                f"(total fallbacks this session: {self._unified_fallback_count})"
            )
            return super().process_student_input(student_input)

        # ── Unified succeeded — apply parent's glue logic ──────────────────
        self._last_thinking_content = unified_result.get("thinking_content")
        self.add_to_history("student", student_input)

        previous_state = self.current_state
        state = unified_result["state"]
        teacher_response = unified_result["teacher_response"]
        evaluation = unified_result["evaluation"]

        # Phase regression prevention (mirrors parent line 449-464)
        if previous_state and state:
            prev_phase = previous_state[0]
            curr_phase = state[0]
            regressed = (
                (prev_phase == "b" and curr_phase == "a")
                or (prev_phase == "c" and curr_phase in ["a", "b"])
                or (prev_phase == "d" and curr_phase in ["a", "b", "c"])
                or (prev_phase == "e" and curr_phase in ["a", "b", "c", "d"])
            )
            if regressed:
                emitted_state = state
                state = previous_state
                evaluation = f"防止阶段回退：保持在{state}状态而不是回退到{emitted_state}"
                # NOTE: per fusion plan §6 Option A, teacher_response stays
                # as emitted; we trust the model's response even though state
                # was overridden. Log so we can measure divergence rate later.

        action = self.get_action_for_state(state)

        # Teaching round counter (mirrors parent line 468-473)
        if previous_state == "a0" and state != "a0":
            self.teaching_rounds = 1
        elif previous_state != "a0" and state != "a0":
            self.teaching_rounds += 1

        self.add_to_consultant_history(evaluation, state, action)

        # Forced d33 / e34 transition logic (mirrors parent line 479-500)
        if self.teaching_rounds > self.max_teaching_rounds:
            if state == "e34":
                pass  # allow transition to summary
            elif previous_state == "d33" and state != "d33":
                state = "d33"
                action = self.get_action_for_state("d33")
                evaluation = (
                    f"已达到教学阶段最大轮数限制({self.max_teaching_rounds}轮)，"
                    "强制维持在d33阶段等待正确答案"
                )
            elif state != "d33":
                state = "d33"
                action = self.get_action_for_state("d33")
                evaluation = (
                    f"已达到教学阶段最大轮数限制({self.max_teaching_rounds}轮)，"
                    "强制转入规则建构阶段"
                )
        elif self.teaching_rounds == self.max_teaching_rounds and state not in ("d33", "e34"):
            state = "d33"
            action = self.get_action_for_state("d33")
            evaluation = (
                f"已达到教学阶段最大轮数限制({self.max_teaching_rounds}轮)，强制转入规则建构阶段"
            )

        if self.debug_mode:
            print("\n=== 苏格拉底统一系统分析 ===")
            if state != "a0":
                print(f"教学阶段对话轮数: {self.teaching_rounds}/{self.max_teaching_rounds}")
            print(f"评估: {evaluation}")
            print(f"状态: {state}")
            print(f"行动: {action}")
            print(f"教师回复: {teacher_response}")
            print("=============================\n")

        self.current_state = state
        self.add_to_history("teacher", teacher_response)
        return teacher_response
