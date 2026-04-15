"""Console script wrappers so `poetry run test` and `poetry run lint` work."""

import subprocess
import sys


def test() -> None:
    """Run pytest, forwarding any extra CLI args."""
    import pytest
    sys.exit(pytest.main(sys.argv[1:]))


def lint() -> None:
    """Run ruff check on src/ and tests/, forwarding any extra CLI args."""
    result = subprocess.run(["ruff", "check", "src", "tests"] + sys.argv[1:])
    sys.exit(result.returncode)
