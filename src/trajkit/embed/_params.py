"""Frozen parameters and the plugin protocol for ``trajkit.embed``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

# Number of features in each base block. Documented constants so
# ``EmbedParams.expected_dim`` is computed from one source of truth.
#
# The kinematic block carries the four conceptually independent scale +
# intensity features of a segment. ``mean_speed_ms`` is path/duration
# (redundant), ``n_pings`` is duration × ping_rate (redundant),
# ``straightness`` and ``bearing_variance`` are shape signals — those
# live in the shape block.
KINEMATIC_FEATURES: tuple[str, ...] = (
    "duration_s",
    "path_length_m",
    "displacement_m",
    "max_speed_ms",
)
KINEMATIC_DIM = len(KINEMATIC_FEATURES)

CYCLIC_FEATURES: tuple[str, ...] = ("hour_of_day", "day_of_week")
CYCLIC_FEATURE_DIM = 2  # sin + cos per harmonic per feature

SEGMENT_TYPE_ORDER: tuple[str, ...] = ("MOVE", "MOVE_BRIEF", "STOP_BRIEF", "STOP_DWELL")
SEGMENT_TYPE_DIM = len(SEGMENT_TYPE_ORDER)

SPATIAL_DIM = 4  # start_lat, start_lon, end_lat, end_lon normalised to [0, 1]

# Distance-resampled bearing-shape features. Each feature is mapped
# into ``[0, 1]`` (or ``[-1, 1]`` for signed) by ``_shape_block`` and
# scaled by ``EmbedParams.shape_weight`` so shape and kinematics
# carry comparable energy in the final L2-normalised vector.
SHAPE_FEATURES: tuple[str, ...] = (
    "shape_R",
    "shape_R2",
    "shape_signed_net_revs",
    "shape_int_curv_deg_per_step",
    "shape_abs_delta_p95_deg",
)
SHAPE_DIM = len(SHAPE_FEATURES)

# Default shape weight: 3× brings shape from ~4% to ~27% of the
# cohort-mean unit-vector energy on the Geolife week, comparable to
# kinematic's ~58%. Empirically validated.
DEFAULT_SHAPE_WEIGHT: float = 3.0


class EmbedParams(BaseModel):
    """Hyperparameters for the trajkit base embedding recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cyclic_harmonics: PositiveInt = Field(
        default=4,
        le=10,
        description="Number of sinusoidal harmonics per cyclic feature.",
    )
    spatial_bounds: tuple[float, float, float, float] = Field(
        default=(-90.0, 90.0, -180.0, 180.0),
        description=(
            "(lat_min, lat_max, lon_min, lon_max) for the spatial-normalisation block. "
            "Defaults to the world; tighten to the cohort bounding box for a "
            "denser mapping."
        ),
    )
    l2_normalize: bool = Field(
        default=True,
        description="Whether to L2-normalise output vectors (FAISS cosine pre-req).",
    )
    shape_weight: PositiveFloat = Field(
        default=DEFAULT_SHAPE_WEIGHT,
        description=(
            "Multiplier applied to the shape block before L2-normalisation. "
            "Default 3.0 makes shape ~27% of the cohort-mean unit-vector "
            "energy. Set 1.0 to make shape a tiebreaker; raise to push the "
            "metric to match shape over scale."
        ),
    )
    epsilon: PositiveFloat = Field(
        default=1e-8,
        description="Numerical-stability epsilon for L2 norm and similar ops.",
    )

    def base_dim(self) -> int:
        """Dimension of the base block before plugin contributions."""
        cyclic = len(CYCLIC_FEATURES) * CYCLIC_FEATURE_DIM * self.cyclic_harmonics
        return KINEMATIC_DIM + cyclic + SEGMENT_TYPE_DIM + SPATIAL_DIM + SHAPE_DIM

    def expected_dim(self, features: tuple[FeaturePlugin, ...] = ()) -> int:
        """Total embedded dimension given a tuple of plugins."""
        return self.base_dim() + sum(f.dim for f in features)


@runtime_checkable
class FeaturePlugin(Protocol):
    """Contributes a fixed-width block to the segment embedding.

    Implementations supply a ``name`` (for diagnostics), a ``dim`` (declared
    output width), and a ``compute`` callable returning ``shape == (len(df),
    dim)`` of dtype ``float32``. Output shape is asserted on every call.
    """

    name: str
    dim: int

    def compute(self, segments_df: pd.DataFrame) -> np.ndarray: ...
