"""Smoke tests: package imports and exposes its version."""

import trajkit


def test_version_exists() -> None:
    assert trajkit.__version__ == "0.1.0"


def test_modules_import() -> None:
    """Each public module imports without error."""
    import trajkit.clean  # noqa: F401
    import trajkit.compare  # noqa: F401
    import trajkit.embed  # noqa: F401
    import trajkit.episode  # noqa: F401
    import trajkit.segment  # noqa: F401
    import trajkit.testing  # noqa: F401
    import trajkit.types  # noqa: F401
