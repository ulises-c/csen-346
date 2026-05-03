"""Console script wrappers so `poetry run test` and `poetry run lint` work."""

import subprocess
import sys


def test() -> None:
    """Run pytest, forwarding any extra CLI args."""
    import pytest

    sys.exit(pytest.main(sys.argv[1:]))


def lint() -> None:
    """Run ruff format + ruff check on src/ and tests/. Pass --fix to auto-fix both."""
    fix = "--fix" in sys.argv[1:]

    fmt_cmd = ["ruff", "format"] + ([] if fix else ["--check"]) + ["src", "tests"]
    check_cmd = ["ruff", "check"] + (["--fix"] if fix else []) + ["src", "tests"]

    r = subprocess.run(fmt_cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

    sys.exit(subprocess.run(check_cmd).returncode)
