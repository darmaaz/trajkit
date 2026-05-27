"""Smoke tests: package imports and exposes its version."""

import re

import trajkit


def test_version_exists() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[+\-.][\w.]+)?", trajkit.__version__)


def test_modules_import() -> None:
    """Each public module imports without error."""
    import trajkit.clean  # noqa: F401
    import trajkit.compare  # noqa: F401
    import trajkit.embed  # noqa: F401
    import trajkit.episode  # noqa: F401
    import trajkit.segment  # noqa: F401
    import trajkit.testing  # noqa: F401
    import trajkit.types  # noqa: F401
