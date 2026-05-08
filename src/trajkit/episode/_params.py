"""Frozen parameters for ``trajkit.episode``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class EpisodeParams(BaseModel):
    """Parameters for ``trajkit.episode.detect_episodes``.

    Three knobs control the spatial-envelope closure rule:

    * ``R_m`` — envelope radius (meters). Defines "here." Smaller than R is
      same place; larger and persistent is departure.
    * ``T_s`` — departure persistence (seconds). Once outside the envelope,
      the entity has this long to come back before the stay closes.
    * ``min_stay_s`` — minimum stay duration. Below this, the candidate
      stay is rejected and its segments fall into the surrounding transit.

    See ``docs/design/episode.md`` for parameter rationale and per-domain
    presets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    R_m: PositiveFloat = Field(
        default=200.0,
        description="Envelope radius — segments whose centroids stay within "
        "this radius of the running anchor count as the same place.",
    )
    T_s: PositiveFloat = Field(
        default=300.0,
        description="Departure persistence — outside time before a stay "
        "closes, and the inter-segment gap that closes any episode.",
    )
    min_stay_s: PositiveFloat = Field(
        default=180.0,
        description="Minimum stay duration. Candidates below this are folded "
        "into the surrounding transit.",
    )
    h3_resolution: int = Field(
        default=9,
        ge=0,
        le=15,
        description="H3 resolution for the anchor_h3 column on STAY episodes.",
    )
