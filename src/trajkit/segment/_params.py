"""Frozen parameters for ``trajkit.segment``.

User-facing speed thresholds are in km/h for readability. Conversion to
SI (m/s) happens once at the boundary of each function. Time thresholds
are in seconds or minutes per the convention that matches their natural
unit.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class SegmentParams(BaseModel):
    """Hysteresis, bearing, and classification thresholds for ``segment``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stop_speed_kmh: NonNegativeFloat = Field(
        default=2.0,
        description="Speed below this enters stopped state (lower hysteresis edge).",
    )
    resume_speed_kmh: PositiveFloat = Field(
        default=5.0,
        description="Speed above this re-enters moving state (upper hysteresis edge).",
    )
    stop_min_duration_s: NonNegativeFloat = Field(
        default=30.0,
        description=(
            "Stop runs shorter than this are noise; reclassified as moving. "
            "Time-based filter handles merged data and mixed ping rates correctly."
        ),
    )

    bearing_change_deg: PositiveFloat = Field(
        default=45.0,
        description="Sustained rolling-mean bearing delta to split a MOVE.",
    )
    bearing_window_min: PositiveFloat = Field(
        default=2.0,
        description="Rolling-mean window for bearing-delta detection (minutes).",
    )
    bearing_sustain_s: PositiveFloat = Field(
        default=180.0,
        description="Bearing delta must exceed threshold continuously for this long.",
    )

    dwell_threshold_min: PositiveFloat = Field(
        default=5.0,
        description="STOP_BRIEF vs STOP_DWELL boundary (minutes).",
    )
    move_brief_min_pings: PositiveInt = Field(
        default=5,
        description="MOVE segments below this raw-ping count → MOVE_BRIEF.",
    )
    move_brief_max_duration_s: PositiveFloat = Field(
        default=180.0,
        description="MOVE segments below this duration → MOVE_BRIEF.",
    )
    max_stop_displacement_m: PositiveFloat = Field(
        default=500.0,
        description=(
            "Stops with crow-fly displacement above this are reclassified as MOVE. "
            "Catches the false-stop case where hysteresis classified a slow drive "
            "as stopped."
        ),
    )

    h3_resolution: int = Field(
        default=9,
        ge=0,
        le=15,
        description="H3 resolution for start_h3 / end_h3 in aggregated segments.",
    )

    @model_validator(mode="after")
    def _resume_above_stop(self) -> SegmentParams:
        if self.resume_speed_kmh <= self.stop_speed_kmh:
            raise ValueError(
                "resume_speed_kmh must exceed stop_speed_kmh for hysteresis to "
                f"function (got resume={self.resume_speed_kmh}, "
                f"stop={self.stop_speed_kmh})."
            )
        return self
