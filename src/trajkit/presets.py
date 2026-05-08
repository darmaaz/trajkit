"""Per-domain default ``RunParams`` bundles.

v0.1.0 ships two presets per LIBRARY.md §17:

* ``"logistics_vehicle"`` — built from fleet-tuned defaults (the
  ``RunParams()`` defaults are already vehicle-shaped).
* ``"pedestrian"`` — calibrated for walking-scale data (Geolife-shape).
  Differs from logistics_vehicle in episode parameters: tighter envelope
  radius, shorter departure persistence, shorter minimum stay.

Other domains (maritime, wildlife, mobile_phone) are documented in the
design but deferred to v1.1+.

Use ``RunParams.from_preset(name)`` for the ergonomic form.
"""

from __future__ import annotations

from trajkit.episode import EpisodeParams
from trajkit.runner import RunParams

SCALE_PRESETS: dict[str, RunParams] = {
    "logistics_vehicle": RunParams(),
    "pedestrian": RunParams(
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
