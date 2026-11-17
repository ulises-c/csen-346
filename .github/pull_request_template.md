## Summary

<!-- What does this PR do and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature / experiment
- [ ] Refactor / cleanup
- [ ] Docs / config only
- [ ] Dependency bump

## Checklist

- [ ] All checks pass — run the pre-commit hook (`make install-hooks`) or manually:
  - `uv run ruff format --check .` — formatting
  - `uv run ruff check .` — linting
  - `uv run pyright` — type checking
  - `uv run codespell .` — spell checking
  - `uv run pytest -rs --cov --cov-report=term-missing` — tests + coverage
- [ ] New code has test coverage (or explain why not)
- [ ] Docs updated if behaviour changed (README, setup guides, etc.)

## Experiment results (if applicable)

<!-- Attach or link metrics_summary.json / eval output if this PR affects model behaviour -->
