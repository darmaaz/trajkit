"""Canonical Pandera + Arrow schemas for trajkit's public DataFrames.

Single source of truth for the data contract. See
``docs/design/schemas.md`` for the column-level reference.

Each schema is declared twice:

* As a Pandera ``DataFrameModel`` for runtime validation at the L2 iterator
  boundary and at pass-2 read boundaries.
* As a ``pyarrow.Schema`` for parquet I/O round-trip.

A test in ``tests/unit/test_schemas.py`` asserts the two declarations agree
on column names, dtypes, and nullability.

Schema versioning: a single ``SCHEMA_VERSION`` integer applies to all v0.x
outputs. The version is stored in each parquet's key-value metadata under
``trajkit.schema_version``. Bumped only on breaking changes — column rename,
dtype change, or semantic shift. Adding a column with a default does not
require a bump; readers tolerate missing-with-default for older files.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

import numpy as np
import pandas as pd
import pandera.pandas as pdr
import pyarrow as pa
from pandera.typing import Series

# ── Versioning ──────────────────────────────────────────────────────

SCHEMA_VERSION = 1

ARROW_VERSION_META_KEY = b"trajkit.schema_version"


def _with_version(schema: pa.Schema) -> pa.Schema:
    """Attach the trajkit schema version to an Arrow schema's metadata."""
    return schema.with_metadata({ARROW_VERSION_META_KEY: str(SCHEMA_VERSION).encode()})


def _extend(base: pa.Schema, new_fields: list[pa.Field]) -> pa.Schema:
    """Return ``base`` with ``new_fields`` appended; reapplies version metadata."""
    fields = [base.field(i) for i in range(len(base))]
    return _with_version(pa.schema(fields + new_fields))


# ── Enum vocabularies ───────────────────────────────────────────────

QUALITY_FLAGS = frozenset({"VALID", "DRIFT", "SPEED_OUTLIER", "GAP_FOLLOWS", "DEVICE_FAULT"})
SEGMENT_TYPES = frozenset({"MOVE", "MOVE_BRIEF", "STOP_BRIEF", "STOP_DWELL"})
EPISODE_TYPES = frozenset({"STAY", "TRANSIT"})


# ── PingsSchema (L2 boundary input) ─────────────────────────────────


class PingsSchema(pdr.DataFrameModel):
    """L1 input contract — one entity per frame, sorted by ``ts``.

    iter_entities yields frames matching this schema. L1 functions trust
    these invariants and assert single-entity / sorted on entry.
    """

    entity_id: Series[str] = pdr.Field(nullable=False)
    ts: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pdr.Field(nullable=False)
    lat: Series[float] = pdr.Field(nullable=False, ge=-90.0, le=90.0)
    lon: Series[float] = pdr.Field(nullable=False, ge=-180.0, le=180.0)
    speed_ms: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    bearing_deg: Series[np.float32] = pdr.Field(nullable=True, ge=0.0, lt=360.0)

    class Config:
        strict = True
        coerce = False

    @pdr.dataframe_check
    def _single_entity(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]
        if len(df) == 0:
            return True
        first = df["entity_id"].iloc[0]
        return bool(df["entity_id"].eq(first).all())

    @pdr.dataframe_check
    def _ts_monotonic(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]
        return bool(df["ts"].is_monotonic_increasing)


PINGS_ARROW: pa.Schema = _with_version(
    pa.schema(
        [
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("lat", pa.float64(), nullable=False),
            pa.field("lon", pa.float64(), nullable=False),
            pa.field("speed_ms", pa.float32(), nullable=True),
            pa.field("bearing_deg", pa.float32(), nullable=True),
        ]
    )
)


# ── CleanedPingsSchema (output of trajkit.clean) ────────────────────


class CleanedPingsSchema(PingsSchema):
    """PingsSchema + quality flags and derived kinematics."""

    dt_seconds: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    displacement_m: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    is_duplicate: Series[bool] = pdr.Field(nullable=False)
    quality_flag: Series[str] = pdr.Field(nullable=False, isin=QUALITY_FLAGS)
    merge_count: Series[pd.Int32Dtype] = pdr.Field(nullable=True, ge=1)
    run_duration_s: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)


CLEANED_PINGS_ARROW: pa.Schema = _extend(
    PINGS_ARROW,
    [
        pa.field("dt_seconds", pa.float32(), nullable=True),
        pa.field("displacement_m", pa.float32(), nullable=True),
        pa.field("is_duplicate", pa.bool_(), nullable=False),
        pa.field("quality_flag", pa.string(), nullable=False),
        pa.field("merge_count", pa.int32(), nullable=True),
        pa.field("run_duration_s", pa.float32(), nullable=True),
    ],
)


# ── SegmentedPingsSchema (output of trajkit.segment per-ping pass) ──


class SegmentedPingsSchema(CleanedPingsSchema):
    """CleanedPingsSchema + segment_id and segment_type per ping."""

    segment_id: Series[str] = pdr.Field(nullable=False)
    segment_type: Series[str] = pdr.Field(nullable=False, isin=SEGMENT_TYPES)


SEGMENTED_PINGS_ARROW: pa.Schema = _extend(
    CLEANED_PINGS_ARROW,
    [
        pa.field("segment_id", pa.string(), nullable=False),
        pa.field("segment_type", pa.string(), nullable=False),
    ],
)


# ── SegmentsSchema (output of aggregate_segments) ───────────────────


class SegmentsSchema(pdr.DataFrameModel):
    """One row per segment — input to ``episode`` and ``embed``."""

    segment_id: Series[str] = pdr.Field(nullable=False, unique=True)
    entity_id: Series[str] = pdr.Field(nullable=False)
    segment_type: Series[str] = pdr.Field(nullable=False, isin=SEGMENT_TYPES)
    start_ts: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pdr.Field(nullable=False)
    end_ts: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pdr.Field(nullable=False)
    duration_s: Series[np.float32] = pdr.Field(nullable=False, gt=0.0)
    start_lat: Series[float] = pdr.Field(nullable=False, ge=-90.0, le=90.0)
    start_lon: Series[float] = pdr.Field(nullable=False, ge=-180.0, le=180.0)
    end_lat: Series[float] = pdr.Field(nullable=False, ge=-90.0, le=90.0)
    end_lon: Series[float] = pdr.Field(nullable=False, ge=-180.0, le=180.0)
    start_h3: Series[str] = pdr.Field(nullable=False)
    end_h3: Series[str] = pdr.Field(nullable=False)
    path_length_m: Series[np.float32] = pdr.Field(nullable=False, ge=0.0)
    displacement_m: Series[np.float32] = pdr.Field(nullable=False, ge=0.0)
    straightness: Series[np.float32] = pdr.Field(nullable=False, ge=0.0, le=1.0)
    mean_speed_ms: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    max_speed_ms: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    bearing_variance: Series[np.float32] = pdr.Field(nullable=True, ge=0.0, le=1.0)
    n_pings: Series[np.int32] = pdr.Field(nullable=False, ge=1)

    class Config:
        strict = True
        coerce = False


SEGMENTS_ARROW: pa.Schema = _with_version(
    pa.schema(
        [
            pa.field("segment_id", pa.string(), nullable=False),
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("segment_type", pa.string(), nullable=False),
            pa.field("start_ts", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("end_ts", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("duration_s", pa.float32(), nullable=False),
            pa.field("start_lat", pa.float64(), nullable=False),
            pa.field("start_lon", pa.float64(), nullable=False),
            pa.field("end_lat", pa.float64(), nullable=False),
            pa.field("end_lon", pa.float64(), nullable=False),
            pa.field("start_h3", pa.string(), nullable=False),
            pa.field("end_h3", pa.string(), nullable=False),
            pa.field("path_length_m", pa.float32(), nullable=False),
            pa.field("displacement_m", pa.float32(), nullable=False),
            pa.field("straightness", pa.float32(), nullable=False),
            pa.field("mean_speed_ms", pa.float32(), nullable=True),
            pa.field("max_speed_ms", pa.float32(), nullable=True),
            pa.field("bearing_variance", pa.float32(), nullable=True),
            pa.field("n_pings", pa.int32(), nullable=False),
        ]
    )
)


# ── EpisodesSchema (output of detect_episodes) ──────────────────────


class EpisodesSchema(pdr.DataFrameModel):
    """One row per episode. STAY-only and TRANSIT-only fields are nullable."""

    episode_id: Series[str] = pdr.Field(nullable=False, unique=True)
    entity_id: Series[str] = pdr.Field(nullable=False)
    episode_type: Series[str] = pdr.Field(nullable=False, isin=EPISODE_TYPES)
    start_ts: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pdr.Field(nullable=False)
    end_ts: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pdr.Field(nullable=False)
    duration_s: Series[np.float32] = pdr.Field(nullable=False, gt=0.0)
    segment_ids: Series[object] = pdr.Field(nullable=False)
    n_segments: Series[np.int32] = pdr.Field(nullable=False, ge=1)
    # STAY-only
    anchor_lat: Series[float] = pdr.Field(nullable=True, ge=-90.0, le=90.0)
    anchor_lon: Series[float] = pdr.Field(nullable=True, ge=-180.0, le=180.0)
    anchor_h3: Series[str] = pdr.Field(nullable=True)
    envelope_radius_m: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    # TRANSIT-only
    start_lat: Series[float] = pdr.Field(nullable=True, ge=-90.0, le=90.0)
    start_lon: Series[float] = pdr.Field(nullable=True, ge=-180.0, le=180.0)
    end_lat: Series[float] = pdr.Field(nullable=True, ge=-90.0, le=90.0)
    end_lon: Series[float] = pdr.Field(nullable=True, ge=-180.0, le=180.0)
    displacement_m: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    path_length_m: Series[np.float32] = pdr.Field(nullable=True, ge=0.0)
    straightness: Series[np.float32] = pdr.Field(nullable=True, ge=0.0, le=1.0)

    class Config:
        strict = True
        coerce = False

    @pdr.check("segment_ids")
    def _segment_ids_are_lists(cls, s: Series[object]) -> Series[bool]:
        """Each entry must be a list of strings."""
        result = s.apply(
            lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)
        )
        return cast("Series[bool]", result)


EPISODES_ARROW: pa.Schema = _with_version(
    pa.schema(
        [
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("episode_type", pa.string(), nullable=False),
            pa.field("start_ts", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("end_ts", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("duration_s", pa.float32(), nullable=False),
            pa.field("segment_ids", pa.list_(pa.string()), nullable=False),
            pa.field("n_segments", pa.int32(), nullable=False),
            pa.field("anchor_lat", pa.float64(), nullable=True),
            pa.field("anchor_lon", pa.float64(), nullable=True),
            pa.field("anchor_h3", pa.string(), nullable=True),
            pa.field("envelope_radius_m", pa.float32(), nullable=True),
            pa.field("start_lat", pa.float64(), nullable=True),
            pa.field("start_lon", pa.float64(), nullable=True),
            pa.field("end_lat", pa.float64(), nullable=True),
            pa.field("end_lon", pa.float64(), nullable=True),
            pa.field("displacement_m", pa.float32(), nullable=True),
            pa.field("path_length_m", pa.float32(), nullable=True),
            pa.field("straightness", pa.float32(), nullable=True),
        ]
    )
)


# ── VectorsSchema (output of embed_segments / embed_episodes) ───────


class VectorsSchema(pdr.DataFrameModel):
    """One row per id with a fixed-size float32 vector.

    The exact vector dimension is determined by ``EmbedParams`` and
    plugin set; the schema asserts only that vectors are 1-D float32
    ndarrays of consistent length within a frame.
    """

    id: Series[str] = pdr.Field(nullable=False, unique=True)
    entity_id: Series[str] = pdr.Field(nullable=False)
    vector: Series[object] = pdr.Field(nullable=False)

    class Config:
        strict = True
        coerce = False

    @pdr.check("vector")
    def _vector_is_float32_array(cls, s: Series[object]) -> Series[bool]:
        result = s.apply(
            lambda v: isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim == 1
        )
        return cast("Series[bool]", result)

    @pdr.dataframe_check
    def _consistent_dimension(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]
        if len(df) == 0:
            return True
        lengths = df["vector"].apply(len)
        return bool((lengths == lengths.iloc[0]).all())


VECTORS_ARROW: pa.Schema = _with_version(
    pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("entity_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32()), nullable=False),
        ]
    )
)


# ── BaselinesSchema (output of pass-2 fit_baselines) ────────────────


def make_baselines_schema(cohort_keys: list[str]) -> type[pdr.DataFrameModel]:
    """Build a Baselines schema parameterized by cohort key columns.

    cohort_keys vary per call (e.g. ``["entity_id"]``, ``["road_class",
    "segment_type"]``), so a static schema can't enumerate them. Each
    cohort key is added as a non-nullable string column alongside the
    fixed core columns.
    """

    # Each entry is (annotation type, pandera Field). Pandera/Field types are
    # opaque to mypy, so values are typed as ``tuple[Any, Any]``.
    fields: dict[str, tuple[Any, Any]] = {
        key: (Series[str], pdr.Field(nullable=False)) for key in cohort_keys
    }
    fields["metric"] = (Series[str], pdr.Field(nullable=False))
    fields["mean"] = (Series[np.float32], pdr.Field(nullable=False))
    fields["std"] = (Series[np.float32], pdr.Field(nullable=False, ge=0.0))
    fields["n_samples"] = (Series[np.int32], pdr.Field(nullable=False, ge=1))
    fields["is_fallback"] = (Series[bool], pdr.Field(nullable=False))

    annotations: dict[str, Any] = {name: tp for name, (tp, _field) in fields.items()}
    body: dict[str, Any] = {"__annotations__": annotations}
    for name, (_tp, field) in fields.items():
        body[name] = field

    class _Config:
        strict = True
        coerce = False

    body["Config"] = _Config

    return type("BaselinesSchema", (pdr.DataFrameModel,), body)


def make_baselines_arrow(cohort_keys: list[str]) -> pa.Schema:
    """Arrow counterpart of ``make_baselines_schema``."""
    fields = [pa.field(key, pa.string(), nullable=False) for key in cohort_keys]
    fields.extend(
        [
            pa.field("metric", pa.string(), nullable=False),
            pa.field("mean", pa.float32(), nullable=False),
            pa.field("std", pa.float32(), nullable=False),
            pa.field("n_samples", pa.int32(), nullable=False),
            pa.field("is_fallback", pa.bool_(), nullable=False),
        ]
    )
    return _with_version(pa.schema(fields))


__all__ = [
    "ARROW_VERSION_META_KEY",
    "CLEANED_PINGS_ARROW",
    "CleanedPingsSchema",
    "EPISODE_TYPES",
    "EPISODES_ARROW",
    "EpisodesSchema",
    "PINGS_ARROW",
    "PingsSchema",
    "QUALITY_FLAGS",
    "SCHEMA_VERSION",
    "SEGMENT_TYPES",
    "SEGMENTED_PINGS_ARROW",
    "SEGMENTS_ARROW",
    "SegmentedPingsSchema",
    "SegmentsSchema",
    "VECTORS_ARROW",
    "VectorsSchema",
    "make_baselines_arrow",
    "make_baselines_schema",
]
