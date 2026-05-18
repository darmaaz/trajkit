"""Frozen parameter model for ``trajkit.clean``.

User-facing thresholds are in human-friendly units (km/h, meters,
minutes). Conversion to SI happens once at the boundary of each
function that uses them; nothing else in the module mixes units.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt


class CleanParams(BaseModel):
    """Quality-flag thresholds for ``trajkit.clean.clean``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_speed_kmh: PositiveFloat = Field(
        default=150.0,
        description="Pings with derived speed above this are SPEED_OUTLIER.",
    )
    drift_radius_m: PositiveFloat = Field(
        default=50.0,
        description="Tiny movement within this radius + near-zero speed is DRIFT.",
    )
    drift_speed_kmh: NonNegativeFloat = Field(
        default=1.0,
        description="Speed below this counts as 'near-zero' for DRIFT detection.",
    )
    gap_threshold_min: PositiveFloat = Field(
        default=5.0,
        description="Inter-ping gap above this triggers GAP_FOLLOWS.",
    )

    device_fault_min_pings: PositiveInt = Field(
        default=20,
        description="Minimum entity pings required to evaluate DEVICE_FAULT.",
    )
    device_fault_max_unique_positions: PositiveInt = Field(
        default=2,
        description="Stuck position threshold (unique lat/lon count).",
    )
    device_fault_max_speed_std_kmh: NonNegativeFloat = Field(
        default=2.0,
        description=(
            "Max stddev of reported speed (input ``speed_ms``, converted to km/h) "
            "for a stuck-position entity to be classified DEVICE_FAULT."
        ),
    )
