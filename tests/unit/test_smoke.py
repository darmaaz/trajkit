"""Smoke tests for the v0.0.1 scaffold.

These tests assert the package imports cleanly and exposes its version.
They exist primarily so that ``pytest`` exits 0 in CI before any module
implementation lands.
"""

import trajkit


def test_version_exists() -> None:
    assert trajkit.__version__ == "0.1.0"


def test_module_stubs_import() -> None:
    """Every planned module imports without error, even though empty."""
    import trajkit.clean  # noqa: F401
    import trajkit.compare  # noqa: F401
    import trajkit.embed  # noqa: F401
    import trajkit.episode  # noqa: F401
    import trajkit.io  # noqa: F401
    import trajkit.presets  # noqa: F401
    import trajkit.runner  # noqa: F401
    import trajkit.segment  # noqa: F401
    import trajkit.testing  # noqa: F401
    import trajkit.types  # noqa: F401
