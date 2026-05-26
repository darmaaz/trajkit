"""Schema validation tests for ``trajkit.types``.

Two layers of coverage:

1. Each schema accepts a known-good fixture frame and rejects each
   single-axis perturbation we care about (wrong dtype, out-of-range,
   missing column, duplicate id, etc.).
2. Each pair of (Pandera DataFrameModel, pyarrow Schema) is consistent
   on column names, field nullability, and dtype family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pdr
import pyarrow as pa
import pytest

from trajkit import types as tt

# Pandera raises SchemaError on the first failure and SchemaErrors when
# strict-mode aggregates multiple. Tests catch the union so callers don't
# have to know which path tripped.
SCHEMA_ERROR = (pdr.errors.SchemaError, pdr.errors.SchemaErrors)


# ── Fixture builders ────────────────────────────────────────────────


def _ts_range(n: int) -> pd.DatetimeIndex:
    # ``pd.date_range`` defaults to microsecond precision in pandas 2.x;
    # the canonical trajkit dtype is ``datetime64[ns, UTC]``.
    return pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC").astype(
        "datetime64[ns, UTC]"
    )


def _str_col(value: str, n: int) -> pd.Series:
    """Helper: build a string-dtype column at the canonical type."""
    return pd.Series([value] * n, dtype="string")


def _pings(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": _str_col("v1", n),
            "ts": _ts_range(n),
            "lat": np.linspace(19.4, 19.5, n, dtype=np.float64),
            "lon": np.linspace(-99.2, -99.1, n, dtype=np.float64),
            "speed_ms": np.full(n, 10.0, dtype=np.float32),
            "bearing_deg": np.full(n, 45.0, dtype=np.float32),
        }
    )


def _cleaned(n: int = 5) -> pd.DataFrame:
    df = _pings(n)
    df["dt_seconds"] = np.full(n, 1.0, dtype=np.float32)
    df["displacement_m"] = np.full(n, 10.0, dtype=np.float32)
    df["is_duplicate"] = np.zeros(n, dtype=bool)
    df["quality_flag"] = _str_col("VALID", n)
    df["merge_count"] = pd.array([1] * n, dtype="Int32")
    df["run_duration_s"] = np.full(n, 1.0, dtype=np.float32)
    return df


def _segmented(n: int = 5) -> pd.DataFrame:
    df = _cleaned(n)
    df["segment_id"] = _str_col("v1_seg_00001", n)
    df["segment_type"] = _str_col("MOVE", n)
    return df


def _segments(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": pd.Series([f"v1_seg_{i:05d}" for i in range(1, n + 1)], dtype="string"),
            "entity_id": _str_col("v1", n),
            "segment_type": _str_col("MOVE", n),
            "start_ts": _ts_range(n),
            "end_ts": _ts_range(n) + pd.Timedelta("60s"),
            "duration_s": np.full(n, 60.0, dtype=np.float32),
            "start_lat": np.full(n, 19.4, dtype=np.float64),
            "start_lon": np.full(n, -99.2, dtype=np.float64),
            "end_lat": np.full(n, 19.5, dtype=np.float64),
            "end_lon": np.full(n, -99.1, dtype=np.float64),
            "start_h3": _str_col("8a2a1072b59ffff", n),
            "end_h3": _str_col("8a2a1072b5bffff", n),
            "path_length_m": np.full(n, 600.0, dtype=np.float32),
            "displacement_m": np.full(n, 600.0, dtype=np.float32),
            "straightness": np.full(n, 1.0, dtype=np.float32),
            "mean_speed_ms": np.full(n, 10.0, dtype=np.float32),
            "max_speed_ms": np.full(n, 12.0, dtype=np.float32),
            "bearing_variance": np.full(n, 0.05, dtype=np.float32),
            "n_pings": np.full(n, 60, dtype=np.int32),
            "shape_R": np.full(n, 0.95, dtype=np.float32),
            "shape_R2": np.full(n, 0.90, dtype=np.float32),
            "shape_signed_net_revs": np.full(n, 0.0, dtype=np.float32),
            "shape_int_curv_deg_per_step": np.full(n, 3.0, dtype=np.float32),
            "shape_abs_delta_p95_deg": np.full(n, 10.0, dtype=np.float32),
        }
    )


def _episodes(n: int = 2) -> pd.DataFrame:
    types = (["STAY", "TRANSIT"] + ["STAY"] * max(0, n - 2))[:n]
    return pd.DataFrame(
        {
            "episode_id": pd.Series(
                [f"ep_v1_{i:05d}" for i in range(1, n + 1)], dtype="string"
            ),
            "entity_id": _str_col("v1", n),
            "episode_type": pd.Series(types, dtype="string"),
            "start_ts": _ts_range(n),
            "end_ts": _ts_range(n) + pd.Timedelta("300s"),
            "duration_s": np.full(n, 300.0, dtype=np.float32),
            "segment_ids": [["v1_seg_00001", "v1_seg_00002"] for _ in range(n)],
            "n_segments": np.full(n, 2, dtype=np.int32),
            "anchor_lat": [19.4] + [None] * (n - 1),
            "anchor_lon": [-99.2] + [None] * (n - 1),
            "anchor_h3": pd.Series(
                ["8a2a1072b59ffff"] + [None] * (n - 1), dtype="string"
            ),
            "envelope_radius_m": np.array([50.0] + [np.nan] * (n - 1), dtype=np.float32),
            "start_lat": [None] + [19.4] * (n - 1),
            "start_lon": [None] + [-99.2] * (n - 1),
            "end_lat": [None] + [19.5] * (n - 1),
            "end_lon": [None] + [-99.1] * (n - 1),
            "displacement_m": np.array([np.nan] + [600.0] * (n - 1), dtype=np.float32),
            "path_length_m": np.array([np.nan] + [600.0] * (n - 1), dtype=np.float32),
            "straightness": np.array([np.nan] + [1.0] * (n - 1), dtype=np.float32),
        }
    )


def _vectors(n: int = 3, dim: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.Series(
                [f"v1_seg_{i:05d}" for i in range(1, n + 1)], dtype="string"
            ),
            "entity_id": _str_col("v1", n),
            "vector": [np.zeros(dim, dtype=np.float32) for _ in range(n)],
        }
    )


# ── Happy-path validation ────────────────────────────────────────────


def test_pings_accepts_valid_frame() -> None:
    tt.PingsSchema.validate(_pings())


def test_cleaned_pings_accepts_valid_frame() -> None:
    tt.CleanedPingsSchema.validate(_cleaned())


def test_segmented_pings_accepts_valid_frame() -> None:
    tt.SegmentedPingsSchema.validate(_segmented())


def test_segments_accepts_valid_frame() -> None:
    tt.SegmentsSchema.validate(_segments())


def test_episodes_accepts_valid_frame() -> None:
    tt.EpisodesSchema.validate(_episodes())


def test_vectors_accepts_valid_frame() -> None:
    tt.VectorsSchema.validate(_vectors())


# ── Pings rejections ────────────────────────────────────────────────


def test_pings_rejects_naive_timestamps() -> None:
    df = _pings()
    df["ts"] = pd.date_range("2026-01-01", periods=len(df), freq="1s")  # tz-naive
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_lat_out_of_range() -> None:
    df = _pings()
    df.loc[0, "lat"] = 91.0
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_lon_out_of_range() -> None:
    df = _pings()
    df.loc[0, "lon"] = 181.0
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_negative_speed() -> None:
    df = _pings()
    df["speed_ms"] = df["speed_ms"].astype(np.float32)
    df.loc[0, "speed_ms"] = -1.0
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_bearing_out_of_range() -> None:
    df = _pings()
    df.loc[0, "bearing_deg"] = 360.0  # exclusive upper
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_multi_entity() -> None:
    df = _pings(4)
    df.loc[2:, "entity_id"] = "v2"
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_unsorted_ts() -> None:
    df = _pings(4)
    df = df.iloc[[0, 2, 1, 3]].reset_index(drop=True)
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


def test_pings_rejects_extra_column() -> None:
    df = _pings()
    df["junk"] = 0
    with pytest.raises(SCHEMA_ERROR):
        tt.PingsSchema.validate(df)


# ── CleanedPings rejection ──────────────────────────────────────────


def test_cleaned_pings_rejects_unknown_quality_flag() -> None:
    df = _cleaned()
    df.loc[0, "quality_flag"] = "UNKNOWN"
    with pytest.raises(SCHEMA_ERROR):
        tt.CleanedPingsSchema.validate(df)


# ── Segments rejections ─────────────────────────────────────────────


def test_segments_rejects_duplicate_id() -> None:
    df = _segments(3)
    df.loc[1, "segment_id"] = df.loc[0, "segment_id"]
    with pytest.raises(SCHEMA_ERROR):
        tt.SegmentsSchema.validate(df)


def test_segments_accepts_zero_duration() -> None:
    # A single-ping segment without run_duration_s legitimately has duration 0.
    df = _segments()
    df.loc[0, "duration_s"] = 0.0
    tt.SegmentsSchema.validate(df)


def test_segments_rejects_negative_duration() -> None:
    df = _segments()
    df.loc[0, "duration_s"] = -1.0
    with pytest.raises(SCHEMA_ERROR):
        tt.SegmentsSchema.validate(df)


def test_segments_rejects_straightness_above_one() -> None:
    df = _segments()
    df.loc[0, "straightness"] = 1.5
    with pytest.raises(SCHEMA_ERROR):
        tt.SegmentsSchema.validate(df)


def test_segments_rejects_unknown_segment_type() -> None:
    df = _segments()
    df.loc[0, "segment_type"] = "FLY"
    with pytest.raises(SCHEMA_ERROR):
        tt.SegmentsSchema.validate(df)


# ── Episodes rejections ─────────────────────────────────────────────


def test_episodes_rejects_unknown_episode_type() -> None:
    df = _episodes()
    df.loc[0, "episode_type"] = "WANDERING"
    with pytest.raises(SCHEMA_ERROR):
        tt.EpisodesSchema.validate(df)


def test_episodes_rejects_non_list_segment_ids() -> None:
    df = _episodes()
    df.loc[0, "segment_ids"] = "not-a-list"
    with pytest.raises(SCHEMA_ERROR):
        tt.EpisodesSchema.validate(df)


# ── Vectors rejections ──────────────────────────────────────────────


def test_vectors_rejects_inconsistent_dimension() -> None:
    df = _vectors(n=3, dim=8)
    df.at[0, "vector"] = np.zeros(7, dtype=np.float32)  # noqa: PD008
    with pytest.raises(SCHEMA_ERROR):
        tt.VectorsSchema.validate(df)


def test_vectors_rejects_float64_vector() -> None:
    df = _vectors()
    df.at[0, "vector"] = np.zeros(8, dtype=np.float64)  # noqa: PD008
    with pytest.raises(SCHEMA_ERROR):
        tt.VectorsSchema.validate(df)


# ── Pandera ↔ Arrow consistency ─────────────────────────────────────


# Mapping from pandera dtype repr substring to expected Arrow type.
# Used by the consistency test below; we compare via families because
# pandera's representation of e.g. ``pd.DatetimeTZDtype`` is not a
# direct equality with Arrow's ``timestamp("ns", tz="UTC")``.
_FAMILY: dict[str, pa.DataType] = {
    "string": pa.string(),
    "object": pa.string(),  # pandera Series[str] reports as object dtype
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "Int32": pa.int32(),
    "bool": pa.bool_(),
}


def _family(arrow_type: pa.DataType) -> str:
    """Reduce an Arrow type to a string family for comparison."""
    if pa.types.is_string(arrow_type):
        return "string"
    if pa.types.is_floating(arrow_type):
        return "float32" if arrow_type.bit_width == 32 else "float64"
    if pa.types.is_integer(arrow_type):
        return "int32" if arrow_type.bit_width == 32 else "int64"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    if pa.types.is_list(arrow_type):
        return f"list<{_family(arrow_type.value_type)}>"
    return str(arrow_type)


def _pandera_family(dtype_obj: object) -> str:
    """Reduce a pandera column dtype to the same family vocabulary."""
    s = str(dtype_obj)
    if "DatetimeTZDtype" in s or "datetime64" in s:
        return "timestamp"
    if "Int32" in s or "int32" in s:
        return "int32"
    if "int64" in s or "Int64" in s:
        return "int64"
    if "float32" in s:
        return "float32"
    if "float64" in s or s in {"float", "double"}:
        return "float64"
    if "bool" in s:
        return "bool"
    if "object" in s or s == "str" or "string" in s:
        return "string"
    return s


SCHEMA_PAIRS: list[tuple[type[pdr.DataFrameModel], pa.Schema]] = [
    (tt.PingsSchema, tt.PINGS_ARROW),
    (tt.CleanedPingsSchema, tt.CLEANED_PINGS_ARROW),
    (tt.SegmentedPingsSchema, tt.SEGMENTED_PINGS_ARROW),
    (tt.SegmentsSchema, tt.SEGMENTS_ARROW),
    (tt.EpisodesSchema, tt.EPISODES_ARROW),
]


@pytest.mark.parametrize(("model", "arrow"), SCHEMA_PAIRS)
def test_pandera_arrow_columns_agree(
    model: type[pdr.DataFrameModel], arrow: pa.Schema
) -> None:
    pandera_schema = model.to_schema()
    pandera_cols = set(pandera_schema.columns.keys())
    arrow_cols = set(arrow.names)
    assert pandera_cols == arrow_cols, (
        f"{model.__name__} columns differ: only-pandera={pandera_cols - arrow_cols}, "
        f"only-arrow={arrow_cols - pandera_cols}"
    )


@pytest.mark.parametrize(("model", "arrow"), SCHEMA_PAIRS)
def test_pandera_arrow_nullability_agrees(
    model: type[pdr.DataFrameModel], arrow: pa.Schema
) -> None:
    pandera_schema = model.to_schema()
    for name, col in pandera_schema.columns.items():
        afield = arrow.field(name)
        assert col.nullable == afield.nullable, (
            f"{model.__name__}.{name}: pandera nullable={col.nullable}, "
            f"arrow nullable={afield.nullable}"
        )


# Columns where pandera intentionally uses ``object`` (with a custom check)
# while arrow uses a richer list/struct type. Documented exception, not drift.
_DTYPE_FAMILY_EXCEPTIONS: dict[str, set[str]] = {
    "EpisodesSchema": {"segment_ids"},
}


@pytest.mark.parametrize(("model", "arrow"), SCHEMA_PAIRS)
def test_pandera_arrow_dtype_families_agree(
    model: type[pdr.DataFrameModel], arrow: pa.Schema
) -> None:
    pandera_schema = model.to_schema()
    skip = _DTYPE_FAMILY_EXCEPTIONS.get(model.__name__, set())
    for name, col in pandera_schema.columns.items():
        if name in skip:
            continue
        pf = _pandera_family(col.dtype)
        af = _family(arrow.field(name).type)
        assert pf == af, f"{model.__name__}.{name}: pandera={pf}, arrow={af}"


# ── Schema versioning ───────────────────────────────────────────────


@pytest.mark.parametrize("arrow", [s for _, s in SCHEMA_PAIRS] + [tt.VECTORS_ARROW])
def test_arrow_schema_carries_version(arrow: pa.Schema) -> None:
    assert arrow.metadata is not None
    assert tt.ARROW_VERSION_META_KEY in arrow.metadata
    version = int(arrow.metadata[tt.ARROW_VERSION_META_KEY].decode())
    assert version == tt.SCHEMA_VERSION
