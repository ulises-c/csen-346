# csen-346
Natural Language Processing - CSEN 346 SCU

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs.

- **Paper:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **SocratTeachLLM model:** https://huggingface.co/yuanpan/SocratTeachLLM

## Repository Structure

| Directory | Description |
| --- | --- |
| `src/` | Our reproduction and extensions |
| `resources/KELE/` | KELE baseline — translated source, dataset, and attribution (see [`resources/KELE/ATTRIBUTION.md`](resources/KELE/ATTRIBUTION.md)) |
| `resources/requirements/` | Course guidelines |
| `resources/ideation/` | Topic research and project idea notes |
| `deliverables/` | Course deliverables |
| `PLAN.md` | Project plan with phases and deadlines |
## Mirroring to the org repo

[ulises-c/csen-346](https://github.com/ulises-c/csen-346) is the primary development repo. It is mirrored to [SCU-CSEN346/KELE](https://github.com/SCU-CSEN346/KELE) via a dual-push remote — every `git push` publishes to both simultaneously, preserving full git history.

### How it works

`origin` is configured with two push URLs:

```
origin  git@github.com:ulises-c/csen-346.git  (fetch)
origin  git@github.com:ulises-c/csen-346.git  (push)
origin  git@github.com:SCU-CSEN346/KELE.git   (push)
```

A normal `git push` hits both. Fetch and pull still only come from `ulises-c/csen-346`.

### Setup (first time, per machine)

If you cloned from `SCU-CSEN346/KELE`, re-point your fetch remote and add the second push URL:

```bash
git remote set-url origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
```

If you cloned from `ulises-c/csen-346`, only the two push URLs are needed:

```bash
git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
```

Verify with `git remote -v` — you should see one fetch URL and two push URLs.

## Dependencies

[Poetry](https://python-poetry.org/) - Python package manager
