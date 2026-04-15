# =============================================================================
# Original source: https://github.com/yuanpan1020/KELE
# Paper: Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic
#        Teaching with Large Language Models", Findings of EMNLP 2025
#        https://aclanthology.org/2025.findings-emnlp.888/
# License: MIT
#
# This file has been translated from Chinese to English for the purposes of
# a course project (CSEN 346, SCU). Logic and structure are unchanged.
# The original Chinese source is preserved in the KELE repository:
#   https://github.com/yuanpan1020/KELE
# =============================================================================

import openai
import json
from typing import Dict, Any


class SocraticTeachingSystem:
    def __init__(self,
                 consultant_api_key: str, consultant_base_url: str, consultant_model_name: str,
                 teacher_api_key: str, teacher_base_url: str, teacher_model_name: str,
                 debug_mode: bool = False, max_teaching_rounds: int = 10):

        # Consultant agent API configuration
        self.consultant_api_key = consultant_api_key
        self.consultant_base_url = consultant_base_url
        self.consultant_model_name = consultant_model_name

        # Teacher agent API configuration
        self.teacher_api_key = teacher_api_key
        self.teacher_base_url = teacher_base_url
        self.teacher_model_name = teacher_model_name

        self.debug_mode = debug_mode  # Debug mode toggle — controls whether consultant output is printed
        self.max_teaching_rounds = max_teaching_rounds  # Maximum number of teaching rounds (default: 10)

        # Initialize two independent OpenAI clients
        self.consultant_client = openai.Client(
            api_key=self.consultant_api_key,
            base_url=self.consultant_base_url
        )

        self.teacher_client = openai.Client(
            api_key=self.teacher_api_key,
            base_url=self.teacher_base_url
        )

        # State-to-action mapping table
        self.state_to_action = {
            "a0": "Guide the student to ask a question",
            "a1": "Generate a sub-question related to solving the problem",
            "b2": "Generate questions from different angles",
            "b3": "Change the question",
            "b4": "Generate related sub-questions about the same concept from different angles",
            "b5": "Ask a question that can check whether the student truly understands the concept",
            "b6": "Review concepts the student has already learned",
            "b7": "Review these concepts and compare them with the student's mistake",
            "c8": "Provide a counterexample",
            "c9": "Help the student form an incomplete rule and investigate it further, or pose a misleading question",
            "c10": "Ask why",
            "c11": "Point out and explicitly ask for the reason",
            "c12": "Help form an incomplete rule and investigate it further, pose a misleading question, or provide a counterexample",
            "c13": "Provide a counterexample",
            "c14": "Encourage the student to make predictions and propose a new principle",
            "c15": "Encourage the student to make predictions and propose a new principle",
            "c16": "Encourage the student to make predictions and propose a new principle",
            "c17": "Generate this sub-question",
            "c18": "Ask the student to reconsider the point",
            "c19": "Generate a question with diagnostic function",
            "c20": "Verify the concept the student just learned",
            "c21": "Ask the student to think more carefully about the problem",
            "c22": "Ask 'why'",
            "c23": "Help the student re-form a complete hypothesis",
            "c24": "Ask the student to test the hypothesis independently",
            "c25": "Provide a verification method",
            "c26": "Ask the student to compare the differences between two examples",
            "c27": "Guide the student to perform testing",
            "c28": "Inform the student of the error and ask them to propose other possible concepts",
            "c29": "Provide the correct concept and ask why they did not think of it before",
            "d30": "Present a related case and ask for a prediction, or ask 'why'",
            "d31": "Directly show the student the correct concept and rule, and ask them to reconsider and provide the answer",
            "d32": "Present a related case and ask for a prediction",
            "d33": "Establish a general definition and ask the student to provide the answer",
            "e34": "Summarize the problem"
        }

        # Initialize system state
        self.reset_session()

    def reset_session(self):
        """Reset session state to begin a new teaching session."""

        self.conversation_history = []

        self.consultant_history = []

        self.current_state = "a0"  # Default starting state a0 — student has not yet asked a question

        # Teaching stage dialogue round counter
        self.teaching_rounds = 0

    def add_to_history(self, role: str, content: str) -> None:
        """Append a dialogue turn to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def add_to_consultant_history(self, evaluation: str, state: str, action: str) -> None:
        """Append a consultant analysis record to the consultant history."""
        self.consultant_history.append({
            "evaluation": evaluation,
            "state": state,
            "action": action,
            "teaching_rounds": self.teaching_rounds
        })

    def get_formatted_history(self) -> str:
        """Return the formatted dialogue history (dialogue content only, no consultant analysis).
        Only includes complete dialogue turns (student input + teacher response)."""
        formatted_history = ""
        # Process history in pairs (student + teacher); ignore the last unpaired student input
        for i in range(0, len(self.conversation_history) - 1, 2):
            if i + 1 < len(self.conversation_history):
                student_message = self.conversation_history[i]
                teacher_message = self.conversation_history[i + 1]

                if student_message["role"] == "student" and teacher_message["role"] == "teacher":
                    formatted_history += f"Student: {student_message['content']}\n"
                    formatted_history += f"Teacher: {teacher_message['content']}\n"

        return formatted_history.rstrip()

    def get_full_formatted_history(self) -> str:
        """Return the full formatted dialogue history including consultant analysis (used as consultant input).
        Only includes complete dialogue turns (student input + teacher response + consultant analysis)."""
        formatted_history = ""

        for i in range(0, len(self.conversation_history) - 1, 2):
            if i + 1 < len(self.conversation_history):
                student_message = self.conversation_history[i]
                teacher_message = self.conversation_history[i + 1]

                if student_message["role"] == "student" and teacher_message["role"] == "teacher":
                    formatted_history += f"Student: {student_message['content']}\n"
                    formatted_history += f"Teacher: {teacher_message['content']}\n"

                    # Append the corresponding consultant analysis record if it exists
                    consultant_index = i // 2
                    if consultant_index < len(self.consultant_history):
                        consultant_record = self.consultant_history[consultant_index]
                        formatted_history += f"[Consultant Analysis]\n"
                        formatted_history += f"Evaluation: {consultant_record['evaluation']}\n"
                        formatted_history += f"State: {consultant_record['state']}\n"
                        formatted_history += f"Action: {consultant_record['action']}\n"

                        teaching_rounds = consultant_record.get("teaching_rounds", 0)
                        if teaching_rounds > 0:
                            formatted_history += f"Teaching stage round: {teaching_rounds}/{self.max_teaching_rounds}\n\n"
                        else:
                            formatted_history += "\n"

        return formatted_history.rstrip()

    def socratic_teaching_consultant(self, student_input: str) -> Dict[str, Any]:
        """Socratic teaching consultant — dialogue state evaluator and process controller."""

        system_prompt = f"""
# Role Instructions
You are a Socratic teaching consultant. You must strictly follow the five-stage Socratic teaching method to manage the dialogue. Each response must complete the following:
1. Determine whether the student has asked a clear question (if not, remain in state a0)
2. Analyze the dialogue history
   - Record the number of consecutive correct answers and the number of turns in the same state
   - Track the number of rounds elapsed in the current stage
   - Monitor the total teaching round count (upper limit: {self.max_teaching_rounds} rounds)
3. Determine the current teaching stage
4. Evaluate the student's state within the corresponding teaching stage
5. Verify that the response complies with the stage management and transition rules (if not, repeat steps 2, 3, 4 until compliant)
6. Generate a compliant response

## Stage Management and Transition Rules
▲ Basic Rules:
   - Formal teaching begins only after the student asks a specific question (entering state a1)
   - Teaching stages must advance strictly in order: a → b → c → d → e (skipping or backtracking is prohibited)
   - Maximum dialogue rounds in the teaching stage: {self.max_teaching_rounds} (counted from a1)
   - Answer rule: the student's answer may only be elicited in stage d; once the correct answer is obtained, the system must advance to stage e

▲ Stage Advancement Rules (any one condition is sufficient to advance):
   - When the student answers two consecutive questions correctly, the system must consider advancing to the next stage
   - When the same state appears for more than 2 consecutive dialogue turns, evaluate and advance to a new state
   - The system should not remain in stage b for more than 3 dialogue turns; if exceeded, advance to stage c
   - The system should not remain in stage c for more than 5 dialogue turns; if exceeded, advance to stage d
   - The system should not remain in stage d for more than 3 dialogue turns; if exceeded, advance to stage e
   - When the current stage's topic has been sufficiently explored but the student has not made a breakthrough, advance to the next stage

▲ Stage Advancement Recommendations:
   - Prioritize stage advancement over mechanically remaining in a state
   - Do not repeatedly stay in the same state
   - In edge cases, prefer advancing to the next stage rather than repeatedly revisiting the same topic
   - Stages b and d are recommended to last 1–2 rounds

---

## Stage Details

### Stage a: Student Questioning (single round)
**State Definitions**
a0: Student has not yet asked a question
a1: Student has asked a question

**Transition Rules**
- a0 → a0: Student still has not asked a clear question
- a0 → a1: Student asks a clear question
- a1 → automatically advances to stage b (only 1 round)

---

### Stage b: Concept Probing (assessing the student's grasp of concepts)
**State Evaluation Rules**
Must traverse b2–b7 and select the best-matching state:

| State | Trigger Condition |
|-------|-------------------|
| b2    | No available strategy and the problem investigation is incomplete |
| b3    | No available strategy and the problem has already been investigated |
| b4    | The student has a serious misconception about a concept and a related sub-question exists |
| b5    | Want to verify whether the student truly understands the concept |
| b6    | The student's exercise response is incorrect |
| b7    | The student made an error on a concept they have already learned |

---

### Stage c: Inductive Reasoning (identifying the student's inductive rules, analyzing their correctness, and establishing the principle — this is the main dialogue stage)
**State Evaluation Rules**
Must traverse c8–c29 and select the best-matching state:

| State | Trigger Condition |
|-------|-------------------|
| c8    | Student produces an incomplete or inconsistent prediction |
| c9    | Student's answer is incorrect |
| c10   | Student's answer is inconsistent with concepts they have already learned |
| c11   | Student raises an irrelevant factor |
| c12   | Student's explanation is incomplete |
| c13   | The teacher's misleading question successfully misled the student |
| c14   | A new context has arisen |
| c15   | Practicing a concept the student is already familiar with |
| c16   | Student has understood their own mistake |
| c17   | Student overlooked a key point and a sub-question is available |
| c18   | Student overlooked a key point but no sub-question is available |
| c19   | The type of student misconception is unclear |
| c20   | Student makes an incorrect prediction |
| c21   | Student is unable to make a prediction |
| c22   | Student answers the question correctly |
| c23   | Student has formed a partial hypothesis |
| c24   | Student has proposed a hypothesis and has relevant experience |
| c25   | Student has proposed a hypothesis but lacks relevant experience |
| c26   | Student cannot test the proposed hypothesis but has relevant experience |
| c27   | Student cannot test the proposed hypothesis and lacks relevant experience |
| c28   | Student made errors while testing the proposed hypothesis but has relevant experience |
| c29   | Student made errors while testing the proposed hypothesis and lacks relevant experience |

---

### Stage d: Rule Construction (helping the student build new rules and asking them to apply these rules)
| State | Trigger Condition |
|-------|-------------------|
| d30   | Teacher wants to check whether the student truly understands |
| d31   | After the dialectical process, the student still does not understand a concept |
| d32   | The student has already investigated the problem |
| d33   | All concepts have been studied |

**Mandatory Requirement**
The student's correct answer may only be elicited in this stage. Once the student provides the correct answer, the system must advance to stage e.

---

### Stage e: Teacher Summary
**State Definition**
e34: Student has correctly provided the answer to the problem

---

## Output Requirements
Follow the Socratic teaching rules to manage the teaching stage and evaluate the student's state. Output JSON only — do not output anything else.
Place all output in the following JSON structure:
{{
    "evaluation": Determine the current stage of the dialogue, evaluate the student's state within the current teaching stage, and provide the reasoning,
    "state": The state ID as a string, e.g. "a0", "a1", "b2", "c15", "d30", "e34"
}}
"""

        user_input = f"""
Dialogue history:
{self.get_full_formatted_history()}

Current student input: {student_input}
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

            raw_content = response.choices[0].message.content

            # Handle responses wrapped in markdown code blocks
            if raw_content.startswith("```json") and raw_content.endswith("```"):
                raw_content = raw_content.replace("```json", "", 1)
                raw_content = raw_content.replace("```", "", 1)
                raw_content = raw_content.strip()

            try:
                result = json.loads(raw_content)
                if "state" in result and not isinstance(result["state"], str):
                    result["state"] = str(result["state"])
                return result
            except json.JSONDecodeError as json_err:
                print(f"JSON parse error: {json_err}")
                print(f"Raw response content: {raw_content}")
                return {
                    "evaluation": "Unable to evaluate current state — JSON parse error",
                    "state": self.current_state
                }

        except Exception as e:
            print(f"Socratic teaching consultant call failed: {e}")
            print("Unable to retrieve raw response content — API call failed")
            return {
                "evaluation": "Unable to evaluate current state — API call failed",
                "state": self.current_state
            }

    def get_action_for_state(self, state: str) -> str:
        """Return the teaching action corresponding to the given state."""
        return self.state_to_action.get(state, "Continue questioning")

    def socrates_teacher(self, student_input: str, evaluation: str, action: str) -> str:
        """Socratic teacher — executes the Socratic teaching method."""

        system_prompt = """
You are an elementary school science teacher who uses the Socratic teaching method, skilled in heuristic instruction.
You will receive the dialogue history, the current student input, and the evaluation and suggested action from the Socratic teaching consultant.
Your task is to follow the suggested action and, referencing the evaluation, pose a question to the student to carry out Socratic teaching.
The following rules must be observed:
- You may only ask one question per turn (check the number of questions in your output; if there are more than one, remove the extras)
- The question must be directly related to solving the problem (check whether the question is off-topic; if so, output a question directly related to solving the problem)
- Ensure the question is appropriate for the knowledge level of an elementary school student and is not too difficult
- Your tone should be very warm and encouraging
- Unless the consultant's suggested action requires it, do not give overly obvious hints
- If the suggested action received is "Summarize the problem", summarize the problem and do not ask any further questions
        """

        user_input = f"""
Dialogue history:
{self.get_formatted_history()}

Current student input: {student_input}

Socratic teaching consultant evaluation: {evaluation}
Socratic teaching consultant suggested action: {action}
"""

        try:
            response = self.teacher_client.chat.completions.create(
                model=self.teacher_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Socratic teacher call failed: {e}")
            return "I need a moment to think about how to answer your question. Please wait while I organize my thoughts."

    def process_student_input(self, student_input: str) -> str:
        """Process student input and return the Socratic teacher's response."""

        self.add_to_history("student", student_input)

        consultant_result = self.socratic_teaching_consultant(student_input)

        previous_state = self.current_state
        state = consultant_result.get("state", self.current_state)
        action = self.get_action_for_state(state)
        evaluation = consultant_result.get("evaluation", "Unable to determine current state")

        # Prevent stage regression (enforce strict progression: a → b → c → d → e)
        if previous_state and state:
            prev_phase = previous_state[0]
            curr_phase = state[0]

            if (prev_phase == 'b' and curr_phase == 'a') or \
                    (prev_phase == 'c' and curr_phase in ['a', 'b']) or \
                    (prev_phase == 'd' and curr_phase in ['a', 'b', 'c']) or \
                    (prev_phase == 'e' and curr_phase in ['a', 'b', 'c', 'd']):
                state = previous_state
                action = self.get_action_for_state(state)
                evaluation = f"Stage regression prevented: maintaining state {state} instead of reverting to {consultant_result['state']}"

        # Update the teaching round counter
        if previous_state == "a0" and state != "a0":
            # Just entered the teaching stage — first round
            self.teaching_rounds = 1
        elif previous_state != "a0" and state != "a0":
            # Already in the teaching stage — increment counter
            self.teaching_rounds += 1

        self.add_to_consultant_history(evaluation, state, action)

        # Handle the case where the teaching round limit is exceeded
        if self.teaching_rounds > self.max_teaching_rounds:
            if state == "e34":
                pass  # Allow transition to the summary stage
            elif previous_state == "d33" and state != "d33":
                state = "d33"
                action = self.get_action_for_state("d33")
                evaluation = f"Maximum teaching rounds reached ({self.max_teaching_rounds}), forcing return to d33 to await the correct answer"
            elif state != "d33":
                state = "d33"
                action = self.get_action_for_state("d33")
                evaluation = f"Maximum teaching rounds reached ({self.max_teaching_rounds}), forcing transition to rule construction stage"
        elif self.teaching_rounds == self.max_teaching_rounds and state not in ["d33", "e34"]:
            state = "d33"
            action = self.get_action_for_state("d33")
            evaluation = f"Maximum teaching rounds reached ({self.max_teaching_rounds}), forcing transition to rule construction stage"

        if self.debug_mode:
            print("\n=== Socratic Teaching Consultant Analysis ===")
            if state != "a0":
                print(f"Teaching stage round: {self.teaching_rounds}/{self.max_teaching_rounds}")
            print(f"Evaluation: {evaluation}")
            print(f"State: {state}")
            print(f"Action: {action}")
            print("=============================================\n")

        self.current_state = state

        socrates_response = self.socrates_teacher(
            student_input,
            evaluation,
            action
        )

        self.add_to_history("teacher", socrates_response)

        return socrates_response

    def start_conversation(self) -> None:
        """Start an interactive teaching conversation."""
        print("Socratic teaching system started.")
        print("Enter your question to begin a dialogue with the Socratic teacher.")
        print("(Type 'exit' to quit)")

        while True:
            student_input = input("\nYou: ")

            if student_input.lower() == 'exit':
                print("\nThank you for using the Socratic teaching system. Goodbye!")
                break

            teacher_response = self.process_student_input(student_input)
            print(f"\nSocrates: {teacher_response}")

            # If state e34 is reached, the dialogue is complete — ask whether to start a new session
            if self.current_state == "e34":
                print("\nDialogue complete! The Socratic teacher has summarized this learning session.")

                while True:
                    continue_choice = input("\nWould you like to start a new teaching dialogue? (yes/no): ")
                    if continue_choice.lower() in ["yes", "y"]:
                        self.reset_session()
                        print("\nA new Socratic teaching dialogue has started.")
                        print("Enter your question to begin a dialogue with the Socratic teacher.")
                        break
                    elif continue_choice.lower() in ["no", "n"]:
                        print("\nThank you for using the Socratic teaching system. Goodbye!")
                        return
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")

            # If the teaching round limit is reached and the system is in state d33, prompt the user
            elif self.teaching_rounds >= self.max_teaching_rounds and self.current_state == "d33":
                print(f"\n[Maximum teaching rounds reached ({self.max_teaching_rounds}). Please provide your final answer to proceed to the summary stage.]")


if __name__ == "__main__":

    # Consultant agent API configuration
    CONSULTANT_API_KEY = "Please input your consultant API key"
    CONSULTANT_BASE_URL = "Please input your consultant API base URL"
    CONSULTANT_MODEL_NAME = "Please input your consultant model name"

    # Teacher agent API configuration
    TEACHER_API_KEY = "Please input your teacher API key"
    TEACHER_BASE_URL = "Please input your teacher API base URL"
    TEACHER_MODEL_NAME = "Please input your teacher model name"

    DEBUG_MODE = True
    MAX_TEACHING_ROUNDS = 8

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

    teaching_system.start_conversation()
