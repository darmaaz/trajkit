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

    # ── Bearing change detection (circular-R over distance windows) ──
    #
    # Replaces a previous rolling-mean-of-deltas detector that mixed
    # noise + signal at the per-sample level. Mean resultant length R is
    # the correct circular-statistics primitive; distance-based windows
    # are ping-rate invariant; multi-scale windows catch both street
    # corners (~75 m) and arterial / sustained turns (~200 m); a Schmitt-
    # trigger hysteresis on R prevents flicker around the threshold.
    bearing_window_short_m: PositiveFloat = Field(
        default=75.0,
        description="Short distance window for circular-R, in metres. Catches "
        "street-corner-scale direction changes.",
    )
    bearing_window_long_m: PositiveFloat = Field(
        default=200.0,
        description="Long distance window for circular-R, in metres. Catches "
        "arterial / sustained-turn-scale direction changes.",
    )
    bearing_window_min_pings: PositiveInt = Field(
        default=5,
        description="Minimum valid moving pings inside the distance window for "
        "R to be considered trustworthy. Sparse windows produce NaN R, which "
        "in turn fires no boundary.",
    )
    bearing_r_enter: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description="R threshold for entering the 'direction-changing' state. "
        "R below this in EITHER window indicates a direction change.",
    )
    bearing_r_exit: float = Field(
        default=0.85,
        gt=0.0,
        lt=1.0,
        description="R threshold for exiting the 'direction-changing' state. "
        "R above this in BOTH windows indicates direction has stabilised.",
    )
    bearing_sustain_m: PositiveFloat = Field(
        default=30.0,
        description="Distance (metres) over which the entry/exit R signal "
        "must persist for the hysteresis state to flip.",
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
    def _validate_hysteresis_order(self) -> SegmentParams:
        if self.resume_speed_kmh <= self.stop_speed_kmh:
            raise ValueError(
                "resume_speed_kmh must exceed stop_speed_kmh for hysteresis to "
                f"function (got resume={self.resume_speed_kmh}, "
                f"stop={self.stop_speed_kmh})."
            )
        if self.bearing_r_exit <= self.bearing_r_enter:
            raise ValueError(
                "bearing_r_exit must exceed bearing_r_enter for hysteresis to "
                f"function (got exit={self.bearing_r_exit}, "
                f"enter={self.bearing_r_enter})."
            )
        if self.bearing_window_long_m <= self.bearing_window_short_m:
            raise ValueError(
                "bearing_window_long_m must exceed bearing_window_short_m "
                f"(got long={self.bearing_window_long_m}, "
                f"short={self.bearing_window_short_m})."
            )
        return self
