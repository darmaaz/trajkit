"""Unit tests for ``trajkit.io.iter_entities``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pdr
import pyarrow as pa
import pytest

from trajkit.io import iter_entities

SCHEMA_ERROR = (pdr.errors.SchemaError, pdr.errors.SchemaErrors)


# ── Fixture builders ────────────────────────────────────────────────


def _two_entity_df(n_per: int = 5) -> pd.DataFrame:
    n = 2 * n_per
    return pd.DataFrame(
        {
            "entity_id": pd.Series(["v1"] * n_per + ["v2"] * n_per, dtype="string"),
            "ts": pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC").astype(
                "datetime64[ns, UTC]"
            ),
            "lat": np.linspace(19.4, 19.5, n, dtype=np.float64),
            "lon": np.linspace(-99.2, -99.1, n, dtype=np.float64),
            "speed_ms": np.full(n, 10.0, dtype=np.float32),
            "bearing_deg": np.full(n, 45.0, dtype=np.float32),
        }
    )


# ── DataFrame source ────────────────────────────────────────────────


def test_iter_entities_yields_one_frame_per_entity_id() -> None:
    df = _two_entity_df()
    yielded = list(iter_entities(df))
    assert len(yielded) == 2
    eids = [eid for eid, _ in yielded]
    assert sorted(eids) == ["v1", "v2"]


def test_iter_entities_yields_single_entity_per_frame() -> None:
    df = _two_entity_df()
    for _, group in iter_entities(df):
        first = group["entity_id"].iloc[0]
        assert (group["entity_id"] == first).all()


def test_iter_entities_sorts_each_frame_by_ts() -> None:
    df = _two_entity_df()
    # Shuffle the rows; iter_entities must restore order
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    for _, group in iter_entities(shuffled):
        assert group["ts"].is_monotonic_increasing


def test_iter_entities_does_not_mutate_input() -> None:
    df = _two_entity_df()
    snapshot = df.copy(deep=True)
    list(iter_entities(df))
    pd.testing.assert_frame_equal(df, snapshot)


# ── Boundary coercion ───────────────────────────────────────────────


def test_iter_entities_coerces_object_entity_id_to_string() -> None:
    df = _two_entity_df()
    df["entity_id"] = df["entity_id"].astype(object)
    for _, group in iter_entities(df):
        assert pd.api.types.is_string_dtype(group["entity_id"].dtype)


def test_iter_entities_coerces_naive_ts_to_utc() -> None:
    df = _two_entity_df()
    df["ts"] = pd.date_range("2026-01-01", periods=len(df), freq="1s")  # naive
    for _, group in iter_entities(df):
        assert pd.api.types.is_datetime64_any_dtype(group["ts"])
        # Should be tz-aware now
        assert group["ts"].dt.tz is not None


def test_iter_entities_adds_missing_optional_columns() -> None:
    df = _two_entity_df().drop(columns=["speed_ms", "bearing_deg"])
    for _, group in iter_entities(df):
        assert "speed_ms" in group.columns
        assert "bearing_deg" in group.columns
        # Added as all-NaN
        assert group["speed_ms"].isna().all()
        assert group["bearing_deg"].isna().all()


def test_iter_entities_rejects_missing_required_columns() -> None:
    df = _two_entity_df().drop(columns=["lat"])
    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_entities(df))


# ── Schema validation at boundary ───────────────────────────────────


def test_iter_entities_raises_on_lat_out_of_range() -> None:
    df = _two_entity_df()
    df.loc[0, "lat"] = 91.0  # out of range
    with pytest.raises(SCHEMA_ERROR):
        list(iter_entities(df))


def test_iter_entities_raises_on_lon_reversed_with_lat() -> None:
    df = _two_entity_df()
    # Swap lat/lon — lon range > 90 catches the swap
    df["lat"], df["lon"] = df["lon"].copy(), df["lat"].copy()
    with pytest.raises(SCHEMA_ERROR):
        list(iter_entities(df))


# ── Arrow Table source ──────────────────────────────────────────────


def test_iter_entities_accepts_pyarrow_table() -> None:
    df = _two_entity_df()
    table = pa.Table.from_pandas(df)
    yielded = list(iter_entities(table))
    assert len(yielded) == 2


# ── Parquet path source ─────────────────────────────────────────────


def test_iter_entities_reads_parquet_file(tmp_path: Path) -> None:
    df = _two_entity_df()
    path = tmp_path / "pings.parquet"
    df.to_parquet(path)
    yielded = list(iter_entities(path))
    assert len(yielded) == 2


def test_iter_entities_reads_hive_partitioned_parquet(tmp_path: Path) -> None:
    df = _two_entity_df()
    path = tmp_path / "pings"
    df.to_parquet(path, partition_cols=["entity_id"])
    yielded = list(iter_entities(path))
    assert len(yielded) == 2
    eids = sorted(eid for eid, _ in yielded)
    assert eids == ["v1", "v2"]


def test_iter_entities_raises_on_missing_parquet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_entities(tmp_path / "nope.parquet"))


# ── CSV source ──────────────────────────────────────────────────────


def test_iter_entities_reads_csv(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    df = _two_entity_df()
    path = tmp_path / "pings.csv"
    df.to_csv(path, index=False)
    # Reset the module-level "first CSV warning" so it fires for this test
    import trajkit.io._iter as io_mod
    io_mod._csv_warned = False
    with caplog.at_level("WARNING"):
        yielded = list(iter_entities(path, format="csv"))
    assert len(yielded) == 2
    assert any("CSV" in rec.message for rec in caplog.records)


def test_iter_entities_auto_format_picks_csv_for_csv_extension(
    tmp_path: Path,
) -> None:
    df = _two_entity_df()
    path = tmp_path / "pings.csv"
    df.to_csv(path, index=False)
    yielded = list(iter_entities(path))  # format="auto"
    assert len(yielded) == 2


# ── Source-type rejection ───────────────────────────────────────────


def test_iter_entities_rejects_unsupported_source_type() -> None:
    with pytest.raises(TypeError, match="unsupported source type"):
        list(iter_entities(42))  # type: ignore[arg-type]


def test_iter_entities_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown format"):
        # Bypass the Literal type at runtime
        list(iter_entities(Path("/tmp/x"), format="json"))  # type: ignore[arg-type]
