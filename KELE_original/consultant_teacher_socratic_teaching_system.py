import openai
import json
from typing import Dict, Any


class SocraticTeachingSystem:
    def __init__(self,
                 consultant_api_key: str, consultant_base_url: str, consultant_model_name: str,
                 teacher_api_key: str, teacher_base_url: str, teacher_model_name: str,
                 debug_mode: bool = False, max_teaching_rounds: int = 10):

        # 顾问智能体API配置
        self.consultant_api_key = consultant_api_key
        self.consultant_base_url = consultant_base_url
        self.consultant_model_name = consultant_model_name

        # 教师智能体API配置
        self.teacher_api_key = teacher_api_key
        self.teacher_base_url = teacher_base_url
        self.teacher_model_name = teacher_model_name

        self.debug_mode = debug_mode  # 调试模式开关，控制是否打印智能体1的输出
        self.max_teaching_rounds = max_teaching_rounds  # 最大教学轮数，默认为10

        # 初始化两个独立的OpenAI客户端
        self.consultant_client = openai.Client(
            api_key=self.consultant_api_key,
            base_url=self.consultant_base_url
        )

        self.teacher_client = openai.Client(
            api_key=self.teacher_api_key,
            base_url=self.teacher_base_url
        )

        # 状态到操作的映射表
        self.state_to_action = {
            "a0": "引导学生提出问题",
            "a1": "生成一个与解题相关的子问题",
            "b2": "从不同角度生成问题",
            "b3": "更改问题",
            "b4": "从不同角度生成同一问题的相关子问题",
            "b5": "提出可以检查学生概念的问题",
            "b6": "复习学生已经学过的概念",
            "b7": "复习这些概念并与错误进行比较",
            "c8": "提供一个反例",
            "c9": "帮助学生形成不完整规则并进一步研究它，或提出一个误导性的问题",
            "c10": "询问原因",
            "c11": "应指出并明确询问原因",
            "c12": "帮助其形成不完整规则并进一步研究它，或提出一个误导性的问题，或提供一个反例",
            "c13": "提供一个反例",
            "c14": "鼓励学生做出预测并提出新原则",
            "c15": "鼓励学生做出预测并提出新原则",
            "c16": "鼓励学生做出预测并提出新原则",
            "c17": "生成该子问题",
            "c18": "要求学生重新考虑该点",
            "c19": "生成具有诊断功能的问题",
            "c20": "验证学生刚学习的概念",
            "c21": "要求学生详细思考问题",
            "c22": "问'为什么'",
            "c23": "帮助学生重新形成完整假设",
            "c24": "要求学生单独检验假设",
            "c25": "提供一个验证方法",
            "c26": "要求学生比较两个示例的差异",
            "c27": "指导学生进行检验",
            "c28": "告知学生错误情况并要求其提出其他可能的概念",
            "c29": "提供正确概念并询问为什么之前没有想到",
            "d30": "提出相关案例并要求预测，或问'为什么'",
            "d31": "直接向学生展示正确概念和规则，并要求其重新思考这些概念和给出题目答案",
            "d32": "提出相关案例并要求预测",
            "d33": "建立一个普遍定义并要求学生给出题目答案",
            "e34": "对题目进行总结"
        }

        # 初始化系统状态
        self.reset_session()

    def reset_session(self):
        """重置会话状态，用于开始新的教学轮次"""

        self.conversation_history = []

        self.consultant_history = []

        self.current_state = "a0"  # 默认开始状态为a0，表示学生尚未提出问题

        # 教学阶段对话轮数计数器
        self.teaching_rounds = 0

    def add_to_history(self, role: str, content: str) -> None:
        """添加对话到历史记录"""
        self.conversation_history.append({"role": role, "content": content})

    def add_to_consultant_history(self, evaluation: str, state: str, action: str) -> None:
        """添加顾问分析到历史记录"""
        self.consultant_history.append({
            "evaluation": evaluation,
            "state": state,
            "action": action,
            "teaching_rounds": self.teaching_rounds  # 添加教学轮数
        })

    def get_formatted_history(self) -> str:
        """获取格式化的对话历史（仅对话内容，不包含顾问分析）
        只包含完整的对话轮次（学生输入+教师回复）"""
        formatted_history = ""
        # 确保历史记录被成对处理（学生+教师），忽略最后一条未配对的学生输入
        for i in range(0, len(self.conversation_history) - 1, 2):
            if i + 1 < len(self.conversation_history):  # 确保有配对的教师回复
                student_message = self.conversation_history[i]
                teacher_message = self.conversation_history[i + 1]

                if student_message["role"] == "student" and teacher_message["role"] == "teacher":
                    formatted_history += f"学生: {student_message['content']}\n"
                    formatted_history += f"老师: {teacher_message['content']}\n"

        return formatted_history.rstrip()  # 去除字符串末尾的空白字符

    def get_full_formatted_history(self) -> str:
        """获取包含顾问分析的完整格式化对话历史（用于顾问输入）
        只包含完整的对话轮次（学生输入+教师回复+顾问分析）"""
        formatted_history = ""

        # 确保历史记录被成对处理（学生+教师），忽略最后一条未配对的学生输入
        for i in range(0, len(self.conversation_history) - 1, 2):
            if i + 1 < len(self.conversation_history):  # 确保有配对的教师回复
                student_message = self.conversation_history[i]
                teacher_message = self.conversation_history[i + 1]

                if student_message["role"] == "student" and teacher_message["role"] == "teacher":
                    formatted_history += f"学生: {student_message['content']}\n"
                    formatted_history += f"老师: {teacher_message['content']}\n"

                    # 如果存在对应的顾问分析记录，则添加到历史中
                    consultant_index = i // 2
                    if consultant_index < len(self.consultant_history):
                        consultant_record = self.consultant_history[consultant_index]
                        formatted_history += f"[顾问分析]\n"
                        formatted_history += f"评估: {consultant_record['evaluation']}\n"
                        formatted_history += f"状态: {consultant_record['state']}\n"
                        formatted_history += f"行动: {consultant_record['action']}\n"

                        # 添加教学阶段轮数，但不添加总对话轮数
                        teaching_rounds = consultant_record.get("teaching_rounds", 0)
                        if teaching_rounds > 0:
                            formatted_history += f"教学阶段轮数: {teaching_rounds}/{self.max_teaching_rounds}\n\n"
                        else:
                            formatted_history += "\n"

        return formatted_history.rstrip()  # 去除字符串末尾的空白字符

    def socratic_teaching_consultant(self, student_input: str) -> Dict[str, Any]:
        """苏格拉底教学顾问 - 对话状态判断者和流程控制者"""

        # 构建系统提示词
        system_prompt = f"""
# 角色指令
你作为苏格拉底教学顾问，需严格遵循五阶段苏格拉底教学法进行对话管理。每次响应必须完成：
1. 判断学生是否已提出明确问题（如未提出保持在a0状态）
2. 分析对话历史
   - 记录连续正确回答次数及相同状态持续次数
   - 追踪当前阶段已持续轮数
   - 关注当前教学总轮数(上限为{self.max_teaching_rounds}轮)
3. 确定当前教学阶段
4. 在对应教学阶段内评估学生状态
5. 检查响应结果是否符合阶段管理与转换规则（如不符合则重复2、3、4步直至符合）
6. 生成合规响应

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

| 状态编号 | 触发条件|
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

| 状态编号 | 触发条件|
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

## 输出要求
请按照苏格拉底教学法的规则进行教学阶段管理和学生状态判断，仅输出json，不要输出其它内容。
将所有输出内容放入以下json结构中：
{{
    "evaluation": 确定当前对话所处的阶段，并判断当前教学阶段中的学生状态，同时给出原因,
    "state": 所处状态的编号
}}
"""

        user_input = f"""
历史对话记录:
{self.get_full_formatted_history()}

当前学生输入: {student_input}
"""

        try:
            response = self.consultant_client.chat.completions.create(
                model=self.consultant_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}
            )

            # 获取原始响应内容
            raw_content = response.choices[0].message.content

            # 处理可能包含markdown代码块的响应
            if raw_content.startswith("```json") and raw_content.endswith("```"):
                # 移除markdown代码块标记
                raw_content = raw_content.replace("```json", "", 1)
                raw_content = raw_content.replace("```", "", 1)
                raw_content = raw_content.strip()

            try:
                # 尝试解析JSON
                result = json.loads(raw_content)
                return result
            except json.JSONDecodeError as json_err:
                # JSON解析错误，打印原始内容
                print(f"JSON解析错误: {json_err}")
                print(f"原始响应内容: {raw_content}")
                # 返回默认值
                return {
                    "evaluation": "无法评估当前状态，JSON解析错误",
                    "state": self.current_state  # 保持当前状态不变
                }

        except Exception as e:
            print(f"苏格拉底教学顾问调用失败: {e}")
            print("无法获取原始响应内容，API调用失败")
            # 返回默认值
            return {
                "evaluation": "无法评估当前状态，API调用失败",
                "state": self.current_state  # 保持当前状态不变
            }

    def get_action_for_state(self, state: str) -> str:
        """根据状态获取对应的提问操作"""
        return self.state_to_action.get(state, "继续提问")

    def socrates_teacher(self, student_input: str, evaluation: str, action: str) -> str:
        """苏格拉底教师 - 苏格拉底教学法的教师"""

        # 构建系统提示词
        system_prompt = """
你是一位使用苏格拉底教学法的小学科学教师，擅长启发式教学。
接下来你会收到历史对话记录、当前学生输入和苏格拉底教学顾问对当前教学对话的评估及建议操作；
你的任务是遵循建议的操作并参考评估结果对学生提问以完成苏格拉底式教学。
以下是你需要遵守的规则：
- 每次只能提出一个问题（输出时请检查问题数量，如超出请删去多余问题）
- 提出的问题必须与解题直接相关（输出时请检查问题是否偏离解题，如偏题请重新输出与解题直接相关的问题）
- 请确保提问符合小学阶段学生的知识水平，避免过于困难
- 语气应该非常亲切并具有鼓励性
- 除非苏格拉底教学顾问建议的操作要求，否则不能给出过于明显的提示
- 如果接收到的建议操作为：对题目进行总结，则总结题目且不再提出问题
        """

        # 准备用户输入
        user_input = f"""
历史对话记录:
{self.get_formatted_history()}

当前学生输入: {student_input}

苏格拉底教学顾问评估结果: {evaluation}
苏格拉底教学顾问建议的操作: {action}
"""

        # 调用API - 使用教师专用客户端
        try:
            response = self.teacher_client.chat.completions.create(
                model=self.teacher_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )

            # 获取返回结果
            return response.choices[0].message.content

        except Exception as e:
            print(f"苏格拉底教师调用失败: {e}")
            return "我需要思考一下如何回答你的问题。请稍等片刻，让我组织一下思路。"

    def process_student_input(self, student_input: str) -> str:
        """处理学生输入并返回苏格拉底教师的回复"""

        # 添加学生输入到对话历史
        self.add_to_history("student", student_input)

        # 调用苏格拉底教学顾问
        consultant_result = self.socratic_teaching_consultant(student_input)

        # 获取状态和对应的提问操作
        previous_state = self.current_state
        state = consultant_result.get("state", self.current_state)
        action = self.get_action_for_state(state)
        evaluation = consultant_result.get("evaluation", "无法确定当前状态")

        # 防止阶段回退（确保严格递进：a → b → c → d → e）
        if previous_state and state:
            prev_phase = previous_state[0]  # 获取阶段前缀（a, b, c, d, e）
            curr_phase = state[0]

            # 如果当前阶段比前一阶段更早，则强制保持在当前阶段
            if (prev_phase == 'b' and curr_phase == 'a') or \
                    (prev_phase == 'c' and curr_phase in ['a', 'b']) or \
                    (prev_phase == 'd' and curr_phase in ['a', 'b', 'c']) or \
                    (prev_phase == 'e' and curr_phase in ['a', 'b', 'c', 'd']):
                state = previous_state
                action = self.get_action_for_state(state)
                evaluation = f"防止阶段回退：保持在{state}状态而不是回退到{consultant_result['state']}"

        # 更新教学轮数计数器
        # 如果进入教学阶段（a0 -> a1或更高阶段）或已在教学阶段，则计数
        if previous_state == "a0" and state != "a0":
            # 刚进入教学阶段，第一轮
            self.teaching_rounds = 1
        elif previous_state != "a0" and state != "a0":
            # 已在教学阶段，增加轮数
            self.teaching_rounds += 1

        # 将顾问分析添加到历史记录中
        self.add_to_consultant_history(evaluation, state, action)

        # 处理超过教学轮数限制的情况
        if self.teaching_rounds > self.max_teaching_rounds:
            # 检查是否学生提供了正确答案（顾问判断可以进入e34阶段）
            if state == "e34":
                # 允许转入总结阶段
                pass
            elif previous_state == "d33" and state != "d33":
                # 如果之前在d33，但现在不是d33，也不是e34，则强制回到d33
                state = "d33"
                action = self.get_action_for_state("d33")
                evaluation = f"已达到教学阶段最大轮数限制({self.max_teaching_rounds}轮)，强制维持在d33阶段等待正确答案"
            elif state != "d33":
                # 如果不在d33或e34，则强制进入d33
                state = "d33"
                action = self.get_action_for_state("d33")
                evaluation = f"已达到教学阶段最大轮数限制({self.max_teaching_rounds}轮)，强制转入规则建构阶段"
        elif self.teaching_rounds == self.max_teaching_rounds and state not in ["d33", "e34"]:
            # 刚好达到最大轮数且还未进入d33或e34，强制进入d33
            state = "d33"
            action = self.get_action_for_state("d33")
            evaluation = f"已达到教学阶段最大轮数限制({self.max_teaching_rounds}轮)，强制转入规则建构阶段"

        # 如果开启调试模式，则打印智能体1的输出
        if self.debug_mode:
            print("\n=== 苏格拉底教学顾问分析 ===")
            if state != "a0":
                print(f"教学阶段对话轮数: {self.teaching_rounds}/{self.max_teaching_rounds}")
            print(f"评估: {evaluation}")
            print(f"状态: {state}")
            print(f"行动: {action}")
            print("=============================\n")

        # 更新当前状态
        self.current_state = state

        # 调用苏格拉底教师
        socrates_response = self.socrates_teacher(
            student_input,
            evaluation,
            action
        )

        # 添加老师回复到对话历史
        self.add_to_history("teacher", socrates_response)

        return socrates_response

    def start_conversation(self) -> None:
        """开始对话"""
        print("苏格拉底教学系统已启动。")
        print("请输入你的问题，与苏格拉底教师开始对话。")
        print("(输入'exit'退出对话)")

        while True:
            student_input = input("\n你: ")

            if student_input.lower() == 'exit':
                print("\n感谢使用苏格拉底教学系统，再见！")
                break

            teacher_response = self.process_student_input(student_input)
            print(f"\n苏格拉底: {teacher_response}")

            # 如果到达了e34状态，对话结束，询问是否继续新对话
            if self.current_state == "e34":
                print("\n对话已完成！苏格拉底教师已总结了本次学习。")

                # 询问用户是否继续新的教学
                while True:
                    continue_choice = input("\n是否开始新的教学对话？(是/否): ")
                    if continue_choice.lower() in ["是", "y", "yes"]:
                        # 重置会话状态
                        self.reset_session()
                        print("\n新的苏格拉底教学对话已开始。")
                        print("请输入你的问题，与苏格拉底教师开始对话。")
                        break
                    elif continue_choice.lower() in ["否", "n", "no"]:
                        print("\n感谢使用苏格拉底教学系统，再见！")
                        return  # 结束整个对话
                    else:
                        print("无效输入，请输入'是'或'否'。")

            # 如果教学轮数达到上限且处于d33状态，提示用户
            elif self.teaching_rounds >= self.max_teaching_rounds and self.current_state == "d33":
                print(f"\n【教学阶段已达到最大轮数({self.max_teaching_rounds}轮)，请给出最终答案以进入总结阶段】")



if __name__ == "__main__":

    # 顾问智能体API配置
    CONSULTANT_API_KEY = "Please input your consultant API key"  # 顾问智能体的API密钥
    CONSULTANT_BASE_URL = "Please input your consultant API base URL" # 顾问智能体的API基础URL地址
    CONSULTANT_MODEL_NAME = "Please input your consultant model name" # 顾问智能体使用的模型名称


    # 教师智能体API配置
    TEACHER_API_KEY = "Please input your teacher API key" # 教师智能体的API密钥
    TEACHER_BASE_URL = "Please input your teacher API base URL" # 教师智能体的API基础URL地址
    TEACHER_MODEL_NAME = "Please input your teacher model name" # 教师智能体使用的模型名称


    DEBUG_MODE = True  # 设置是否显示苏格拉底教学顾问的输出
    MAX_TEACHING_ROUNDS = 8  # 设置最大教学轮数

    # 创建教学系统实例
    teaching_system = SocraticTeachingSystem(
        consultant_api_key=CONSULTANT_API_KEY,
        consultant_base_url=CONSULTANT_BASE_URL,
        consultant_model_name=CONSULTANT_MODEL_NAME,
        teacher_api_key=TEACHER_API_KEY,
        teacher_base_url=TEACHER_BASE_URL,
        teacher_model_name=TEACHER_MODEL_NAME,
        debug_mode=DEBUG_MODE,
        max_teaching_rounds=MAX_TEACHING_ROUNDS
    )

    # 启动对话
    teaching_system.start_conversation()
