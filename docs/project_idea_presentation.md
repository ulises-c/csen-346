# Slide 1 - Quick Intro

KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models

Peng, Yuan, Li, Cheng, Fang, Liu - Central China Normal University 

Note: Just present which research paper we're using as our foundation

# Slide 2 - Team Members

Note: Photos and team member names

# Slide 3 - Inspiration + Problem & Motivation

Human tutors use guided questioning, but that kind of Socratic teaching is hard to scale.

KELE stood out because it uses role-specific agents, which connects naturally to Mixture of Experts ideas where specialized models can handle different teaching roles or subjects.

This is still an active research area, making it a strong foundation for our project.

Socratic teaching is powerful, but does not scale. 
Heuristic questioning prompts deep thinking and cognitive development, but demands high teacher expertise and real-time adaptability (which is infeasible in large educational settings).
Existing LLM approaches fail in two ways: 
Prompt engineering (e..g. SPL/GPT-4): mimics Socratic style but has no structured pedagogy, which leads to arbitrary guidance, abrupt topic shifts, disorganized feedback.  
Fine-tuning (e.g. EduChat, SocraticLM): learns to withhold answers but lacks coherent teaching progression or summarization/reflection mechanisms.  
The identified gap: No prior work combines structured teaching rules with LLM-based execution. Style mimicry without pedagogical structure is counterproductive - EduChat scores 90.73% on not giving answers, but only 2.93/5 on actually guiding the student. 
KELE's thesis: Structure + specialization beats raw scale. A 9B model with the right framework outperforms GPT-4o.

Note: Why this topic?

# Slide 4 - The KELE Framework: SocRule + Consultant-Teach Architecture 

Consultant (Planner): Analyzes student cognitive state and dialogue history, which then outputs an evaluation to lookup a recommended teaching action 
Teacher (Executor): Receives history and the consultant plan, which then generates a heuristic response
Per-turn flow: Given a student input that flows into the consultant classification system, the teacher generates a response given a SocRule lookup result. 


# Slide 5 - Results and Limitations

Result:
KELE's 9B model beats GPT-4o on all 13 evaluated teaching metrics, suggesting that structured pedagogy and role specialization can outperform a larger general-purpose model.

Limitations:
1. Domain specificity: Socrat Dataset covers only elementary science. No cross-domain or cross-lingual evaluation. Performance on math, humanities, or programming is unknown.
2. Imperfect rule compliance: PRR of 75.13 means ~25% of turns still have relevance issues. The consultant-teacher mechanism mitigates but doesn't solve this.

# Slide 6 - Future Improvements and Project Roadmap
1
Mixture of Experts for Multi-Subject Teaching (gating)
2
RL-Based Stage Transitions (replace rigid rules)
3
Learned Consultant (via discrete classifier replacing LLM)
4
Stronger Evaluation (via ablation study and cross-domain)

Step 1
Read and break down the KELE paper, identify the core consultant-teacher pipeline, and decide what parts are feasible to reproduce in our course timeline.

Step 2
Build a simplified baseline implementation that captures the main SocRule workflow and multi-agent interaction.

Step 3
Evaluate the reproduced baseline on a small but meaningful set of tutoring scenarios to understand where it succeeds and fails.

Step 4
Choose one focused improvement, such as learned stage transitions or a lighter consultant module, and implement it as our extension.

Step 5
Compare the extension against the baseline, analyze tradeoffs, and prepare the final report and presentation.

Goal
Deliver a working reproduction-plus-improvement project that tests whether structured multi-agent tutoring can be made more adaptive, scalable, and educationally effective.
