"""Per-domain default ``RunParams`` bundles.

v0.1.0 ships two presets per LIBRARY.md §17:

* ``"logistics_vehicle"`` — built from fleet-tuned defaults (the
  ``RunParams()`` defaults are already vehicle-shaped).
* ``"pedestrian"`` — calibrated for walking-scale data (Geolife-shape).
  Differs from logistics_vehicle in BOTH segment AND episode parameters:

  - Segment thresholds tightened for walking pace (~1.0-1.5 m/s typical):
    ``stop_speed_kmh=1.0`` (was 2.0), ``resume_speed_kmh=3.0`` (was 5.0),
    ``max_stop_displacement_m=50.0`` (was 500.0). The vehicle defaults
    classified slow walking as stopped and tolerated up to 500 m of
    "stop" displacement, producing visually wide STOP_DWELL polylines
    on real Geolife data — see ``examples/geolife/explore.ipynb``.
  - Episode envelope tightened: ``R_m=30.0`` (was 200.0),
    ``T_s=120.0`` (was 300.0), ``min_stay_s=120.0`` (was 180.0).

Other domains (maritime, wildlife, mobile_phone) are documented in the
design but deferred to v1.1+.

Use ``RunParams.from_preset(name)`` for the ergonomic form.
"""

from __future__ import annotations

from trajkit.episode import EpisodeParams
from trajkit.runner import RunParams
from trajkit.segment import SegmentParams

SCALE_PRESETS: dict[str, RunParams] = {
    "logistics_vehicle": RunParams(),
    "pedestrian": RunParams(
        segment=SegmentParams(
            stop_speed_kmh=1.0,
            resume_speed_kmh=3.0,
            max_stop_displacement_m=50.0,
        ),
        episode=EpisodeParams(R_m=30.0, T_s=120.0, min_stay_s=120.0),
    ),
}


def get_preset(name: str) -> RunParams:
    """Return a deep copy of the named preset.

    Raises ``KeyError`` with the available preset names when ``name``
    is unknown.
    """
    if name not in SCALE_PRESETS:
        msg = (
            f"unknown preset {name!r}; available: "
            f"{sorted(SCALE_PRESETS.keys())}"
        )
        raise KeyError(msg)
    # RunParams is frozen so returning the cached instance is safe.
    return SCALE_PRESETS[name]


__all__ = ["SCALE_PRESETS", "get_preset"]
