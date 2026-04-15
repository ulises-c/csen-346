# [KELE:](https://music.163.com/#/song?id=29759733) A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models (EMNLP 2025)

> 📄 [Paper](https://aclanthology.org/2025.findings-emnlp.888/) | 🤗 [Model](https://huggingface.co/yuanpan/SocratTeachLLM)

## Overview

[KELE](https://music.163.com/#/song?id=29759733) is a multi-agent framework designed specifically for Socratic teaching. Through a "consultant–teacher" collaborative mechanism, it realizes a controllable, progressive, and interpretable Socratic heuristic teaching process — addressing the challenge that traditional Socratic teaching relies heavily on teacher expertise and is difficult to scale.

The repository provides the complete code for the Socratic teaching system (consultant–teacher core mechanism), the Socratic dialogue dataset, and the Socratic teacher large language model.

![](https://cdn.nlark.com/yuque/0/2025/png/50896216/1762088527961-466e4118-84c3-413f-9594-d95592892658.png)
> Traditional knowledge-imparting teaching vs. heuristic Socratic teaching

## Framework and Core Contributions

The overall architecture and multi-agent mechanism of KELE are shown below:

![](https://cdn.nlark.com/yuque/0/2025/png/50896216/1762088731939-20978fc7-ac30-4055-bca0-83b3565ccad4.png)

> _Left: KELE framework implementation details. Right: "Consultant–Teacher" collaborative flow — the consultant analyzes student state and plans teaching stages; the teacher generates specific dialogue content. Together they implement structured Socratic teaching._

| **Core Contribution** | **Description** |
| --- | --- |
| **SocRule** | **Structured Socratic teaching rules.** Divides Socratic teaching into 5 progressive stages (student questioning → concept probing → inductive reasoning → rule construction → teacher summary), with 34 scenario-based teaching strategies covering the full "question–explore–summarize" workflow. |
| **Consultant–Teacher Mechanism** | Dual-agent collaborative mechanism:<br>- **Consultant agent**: responsible for teaching progress planning (stage/state evaluation, strategy selection);<br>- **Teacher agent**: responsible for teaching execution (generating heuristic questions and feedback). |
| **SocratDataset** | **Structured Socratic teaching dataset.** Contains 6,803 multi-turn dialogues (42,000+ interaction turns), covering all 34 SocRule strategies, built on elementary school science knowledge. |
| **SocratTeachLLM** | **Specialized Socratic teaching model.** Fine-tuned on GLM4-9B; outperforms GPT-4o on all Socratic teaching metrics. |
| **Socratic Teaching Quality Evaluation System** | A **Socratic teaching evaluation framework** with systematicity and generalizability, comprehensively covering both single-turn dialogue and multi-turn teaching processes. |

## Repository Contents

| File | Description |
| --- | --- |
| `consultant_teacher_socratic_teaching_system.py` | Socratic multi-agent teaching system |
| `SocratDataset.json` | Structured Socratic Teaching Dataset and the Guiding Problem-Solving Dataset used to generate it |

## Quick Start

### 1. Install dependencies
```bash
pip install openai
```

### 2. Configure API keys
Fill in the API information for both agents in `consultant_teacher_socratic_teaching_system.py`:

```python
CONSULTANT_API_KEY = "Your consultant agent API key"
CONSULTANT_BASE_URL = "Consultant agent API URL"
CONSULTANT_MODEL_NAME = "Consultant agent model name"

TEACHER_API_KEY = "Your teacher agent API key"
TEACHER_BASE_URL = "Teacher agent API URL"
TEACHER_MODEL_NAME = "Teacher agent model name"
```

> 💡 It is recommended to use [SocratTeachLLM](https://huggingface.co/yuanpan/SocratTeachLLM) as the teacher agent model.

### 3. Run the teaching system
```bash
python consultant_teacher_socratic_teaching_system.py
```

After starting, you will see an interactive prompt and can have a multi-turn heuristic teaching dialogue with the "Socratic teacher". Example interaction:

```
Socratic teaching system started.
Enter your question to begin a dialogue with the Socratic teacher.
(Type 'exit' to quit)

You: Help me answer this question: Compared to a regular magnet, what is different about an electromagnet? (Options: "has magnetic properties" / "has two poles" / "both poles and magnetic force can be changed")

=== Socratic Teaching Consultant Analysis ===
Teaching stage round: 1/8
Evaluation: Student has asked a question, entering stage a, state a1.
State: a1
Action: Generate a sub-question related to solving the problem
=============================================

Socrates: Can you tell me what an electromagnet and a regular magnet have in common?

You: They both have magnetic properties.

=== Socratic Teaching Consultant Analysis ===
Teaching stage round: 2/8
Evaluation: Student answered correctly, entering stage b, state b5 — verifying whether the student truly understands the concept.
State: b5
Action: Ask a question that can check whether the student truly understands the concept
=============================================

Socrates: Great! So do you know how the magnetic properties of an electromagnet are produced?
```

## SocratDataset

- Covers 34 teaching strategies, contains 6,803 teaching tasks, and over 42,000 rounds of simulated teacher–student dialogues.
- Example record structure:

```json
{
  "student_input": "Help me answer this question: Compared to a regular magnet, what is different about an electromagnet? (Options: ...)",
  "teacher_response": "Can you tell me what an electromagnet and a regular magnet have in common?",
  "evaluation": "Student has asked a question, entering stage a, state a1.",
  "state": "a1",
  "action": "Generate a sub-question related to solving the problem"
}
```

## SocratTeachLLM

- Base model: GLM4-9B-Chat
- Fine-tuning method: LoRA
- Training data: SocratDataset (train: 90%, validation: 10%)
- Results:

![SocratTeachLLM outperforms GPT-4o on all metrics](https://cdn.nlark.com/yuque/0/2025/png/50896216/1762090224324-947360cb-4f19-4733-a746-b6a2bc7c0ddf.png)

## License

This project is open-sourced under the **MIT License**.
