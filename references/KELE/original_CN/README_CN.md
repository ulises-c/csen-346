# [KELE:](https://music.163.com/#/song?id=29759733) A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models（EMNLP 2025）

> 📄 [论文链接](https://aclanthology.org/2025.findings-emnlp.888/) ｜ 🤗 [模型地址](https://huggingface.co/yuanpan/SocratTeachLLM)



## 🔍 项目介绍

[KELE](https://music.163.com/#/song?id=29759733) 是一款专为苏格拉底教学设计的多智能体框架。  该框架通过“顾问–教师”协同机制，实现了可控、递进、可解释的苏格拉底启发式教学过程，解决了传统苏格拉底教学依赖教师专业能力、难以规模化的难题。
目前已提供包含“顾问-教师”核心机制的苏格拉底教学系统完整代码、  苏格拉底对话数据集及苏格拉底教师大模型文件。

![](https://cdn.nlark.com/yuque/0/2025/png/50896216/1762088527961-466e4118-84c3-413f-9594-d95592892658.png)
> 传统知识灌输性教学 vs 启发性的苏格拉底教学


## 🛠️ 整体框架与核心贡献

KELE 的整体架构及多智能体机制如下图所示 👇  

![](https://cdn.nlark.com/yuque/0/2025/png/50896216/1762088731939-20978fc7-ac30-4055-bca0-83b3565ccad4.png)

> _左侧为 KELE 框架实现详情；右侧为“顾问-教师”协作流程——顾问负责分析学生状态与阶段规划，教师负责生成具体对话内容，两者配合实现结构化苏格拉底教学。_

| **核心贡献** | **描述** |
| --- | --- |
| **SocRule** | **结构化苏格拉底教学规则。** 将苏格拉底教学分为 5 个递进阶段（学生提问 → 概念探查 → 归纳推理 → 规则建构 → 教师总结），配套 34 种场景化教学策略，覆盖“提问-探索-总结”全流程。 |
| **Consultant–Teacher Mechanism** | 双智能体协作机制：<br> - **顾问智能体**：负责教学进度规划（阶段/状态判断、策略选择）；<br> - **教师智能体**：负责教学执行（生成启发式问题、反馈）。 |
| **SocratDataset** | **结构化苏格拉底教学数据集。** 包含 6803 组多轮对话（共 4.2 万+交互轮次），覆盖全部 34 种 SocRule 策略，基于小学科学知识构建。 |
| **SocratTeachLLM** | **专用苏格拉底教学模型。** 基于 GLM4-9B 微调，在所有苏格拉底教学指标上均优于 GPT-4o。 |
| **Socratic Teaching Quality Evaluation System** | 具备系统性和通用性的 **苏格拉底教学评估框架**，全面覆盖单轮对话与多轮教学过程。 |



## 🧩 仓库内容

| 文件 | 说明 |
| --- | --- |
| `consultant_teacher_socratic_teaching_system.py` | 苏格拉底多智能体教学系统 |
| `SocratDataset.json` | Structured Socratic Teaching Dataset 以及用于生成该数据集的 Guiding Problem-Solving Dataset |



## 🚀 快速开始

### 1️⃣ 安装依赖
```bash
pip install openai
```

### 2️⃣配置 API 密钥
在 `consultant_teacher_socratic_teaching_system.py` 中填写两个智能体的 API 信息：

```python
CONSULTANT_API_KEY = "你的顾问智能体API Key"
CONSULTANT_BASE_URL = "顾问智能体API URL"
CONSULTANT_MODEL_NAME = "顾问智能体模型名称"

TEACHER_API_KEY = "你的教师智能体API Key"
TEACHER_BASE_URL = "教师智能体API URL"
TEACHER_MODEL_NAME = "教师智能体模型名称"
```

> 💡推荐使用 [SocratTeachLLM](https://huggingface.co/yuanpan/SocratTeachLLM) 作为教师智能体模型。
>

### 3️⃣运行教学系统
```bash
python consultant_teacher_socratic_teaching_system.py
```

程序启动后，你将看到交互提示并可与“苏格拉底教师”进行多轮启发式教学对话，示例交互：

```plain
苏格拉底教学系统已启动。
请输入你的问题，与苏格拉底教师开始对话。
(输入 'exit' 退出对话)


你: 帮我解答一下这道题：电磁铁和磁铁相比，其不同点是？（选项：“有磁性” “有两极” “两极和磁力大小都可以改变”）

=== 苏格拉底教学顾问分析 ===
教学阶段对话轮数: 1/8
评估: 学生提出问题，进入阶段a，状态为a1。
状态: a1
行动: 生成一个问题
=============================

苏格拉底: 你能告诉我电磁铁和普通磁铁有什么共同点吗？

你: 它们都有磁性。

=== 苏格拉底教学顾问分析 ===
教学阶段对话轮数: 2/8
评估: 学生回答正确，进入阶段b，状态为b5，验证学生是否真正理解该概念。
状态: b5
行动: 提出可以检查学生概念的问题
=============================

苏格拉底: 很好！那么你知道电磁铁的磁性是如何产生的吗？
```


## 📚 SocratDataset
+ 覆盖 34 种教学策略，包含 6,803 个教学任务，超过 42,000 轮模拟的师生对话；
+ 每条记录示例结构：

```json
{
  "student_input": "帮我解答一下这道题：电磁铁和磁铁相比，其不同点是？（选项：“有磁性” “有两极” “两极和磁力大小都可以改变”）",
  "teacher_response": "你能告诉我电磁铁和普通磁铁有什么共同点吗？",
  "evaluation": "学生提出问题，进入阶段a，状态为a1。",
  "state": "a1",
  "action": "生成一个问题"
}
```



## 🧠 **SocratTeachLLM**
+ 基座模型：GLM4-9B-Chat
+ 微调方式：LoRA
+ 训练数据：SocratDataset（训练集：90%、验证集：10%）
+ 实验结果：

![SocratTeachLLM 在全部指标上均超越 GPT-4o](https://cdn.nlark.com/yuque/0/2025/png/50896216/1762090224324-947360cb-4f19-4733-a746-b6a2bc7c0ddf.png)



## 🧾 许可证
本项目基于 **MIT License** 开源。


