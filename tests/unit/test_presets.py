"""Tests for ``trajkit.presets``."""

from __future__ import annotations

import pytest

from trajkit.presets import SCALE_PRESETS, get_preset
from trajkit.runner import RunParams


def test_scale_presets_contains_v1_domains() -> None:
    assert "logistics_vehicle" in SCALE_PRESETS
    assert "pedestrian" in SCALE_PRESETS


def test_scale_presets_values_are_runparams() -> None:
    for name, params in SCALE_PRESETS.items():
        assert isinstance(params, RunParams), f"{name} is not RunParams"


def test_logistics_vehicle_uses_defaults() -> None:
    p = SCALE_PRESETS["logistics_vehicle"]
    assert p == RunParams()


def test_pedestrian_overrides_episode_params() -> None:
    p = SCALE_PRESETS["pedestrian"]
    assert p.episode.R_m == 30.0
    assert p.episode.T_s == 120.0
    assert p.episode.min_stay_s == 120.0


def test_get_preset_returns_named_bundle() -> None:
    p = get_preset("pedestrian")
    assert p.episode.R_m == 30.0


def test_get_preset_raises_for_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("martian_rover")


def test_runparams_from_preset_classmethod() -> None:
    p = RunParams.from_preset("pedestrian")
    assert isinstance(p, RunParams)
    assert p.episode.R_m == 30.0


def test_presets_are_immutable_via_runparams_frozen() -> None:
    """Bundles are frozen so callers can't mutate the cached values."""
    from pydantic import ValidationError

    p = SCALE_PRESETS["pedestrian"]
    with pytest.raises(ValidationError):
        p.run_stale_merge = True  # type: ignore[misc]
