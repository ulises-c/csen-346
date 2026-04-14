# csen-346
Natural Language Processing - CSEN 346 SCU

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs.

- **Paper:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **SocratTeachLLM model:** https://huggingface.co/yuanpan/SocratTeachLLM

## Repository Structure

| Directory | Description |
| --- | --- |
| `KELE_original/` | Original KELE source code and dataset, translated to English (see [`KELE_original/ATTRIBUTION.md`](KELE_original/ATTRIBUTION.md)) |
| `src/` | Our reproduction and extensions |
| `resources/` | Research paper and reference materials |
| `deliverables/` | Course deliverables |
| `scripts/` | Utility scripts |

## Syncing to the org repo

This repo is mirrored from [CSEN-346](https://github.com/ulises-c/csen-346/tree/main) to [SCU-CSEN346/KELE](https://github.com/SCU-CSEN346/KELE). To sync changes:

```bash
./scripts/sync_kele.sh                        # default commit message
./scripts/sync_kele.sh "your commit message"  # custom commit message
```

## Dependencies

[Poetry](https://python-poetry.org/) - Python package manager
