# Attribution

This directory contains the original, unmodified source code and dataset from the **KELE** project.

## Original Source

- **Repository:** https://github.com/yuanpan1020/KELE
- **Paper:** Peng, Yuan, Li, Cheng, Fang, Liu — "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", _Findings of EMNLP 2025_
- **Paper link:** https://aclanthology.org/2025.findings-emnlp.888/
- **Model (SocratTeachLLM):** https://huggingface.co/yuanpan/SocratTeachLLM
- **License:** MIT

## Contents

| File                                             | Description                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `consultant_teacher_socratic_teaching_system.py` | Original KELE multi-agent Socratic teaching system                                            |
| `SocratDataset.json`                             | SocratDataset — 6,803 multi-turn dialogues, 42,000+ interaction turns, 34 teaching strategies |
| `README.md`                                      | Original README from the KELE repository                                                      |

## Translation

The following files have been translated from Chinese to English for readability. Logic and structure are unchanged from the original source.

| File                                             | Translation status                                               |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| `consultant_teacher_socratic_teaching_system.py` | Translated — comments, docstrings, prompts, and UI strings       |
| `README.md`                                      | Translated — see `README_CN.md` for the original Chinese version |
| `README_CN.md`                                   | Original Chinese README, preserved unmodified                    |
| `SocratDataset.json`                             | Untranslated — dataset content remains in Chinese                |

## Note

These files are included here as the **baseline** for a course project (CSEN 346, SCU) that reproduces and extends the KELE framework. All improvements and extensions are maintained separately in `src/`.
