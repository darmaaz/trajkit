"""Unit tests for ``trajkit.baselines``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit.baselines import (
    DEFAULT_METRICS,
    BaselineParams,
    fit_baselines,
)
from trajkit.embed import baseline_zscores

# ── Fixture builders ────────────────────────────────────────────────


def _segments_for(
    n_per_entity: int = 50,
    entities: tuple[str, ...] = ("v1", "v2"),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a SegmentsSchema-shaped frame with synthetic kinematic columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for ent_idx, eid in enumerate(entities):
        for k in range(n_per_entity):
            rows.append(
                {
                    "segment_id": f"{eid}_seg_{k:05d}",
                    "entity_id": eid,
                    "segment_type": "MOVE",
                    "duration_s": float(60.0 + ent_idx * 60.0 + rng.normal(0, 5)),
                    "path_length_m": float(
                        500.0 + ent_idx * 500.0 + rng.normal(0, 50)
                    ),
                    "displacement_m": 100.0,
                    "straightness": 0.5,
                    "mean_speed_ms": float(10.0 + ent_idx * 5.0 + rng.normal(0, 1)),
                    "max_speed_ms": 15.0,
                    "bearing_variance": 0.1,
                    "n_pings": 60,
                }
            )
    df = pd.DataFrame(rows)
    df["segment_id"] = df["segment_id"].astype("string")
    df["entity_id"] = df["entity_id"].astype("string")
    df["segment_type"] = df["segment_type"].astype("string")
    for col in (
        "duration_s",
        "path_length_m",
        "displacement_m",
        "straightness",
        "mean_speed_ms",
        "max_speed_ms",
        "bearing_variance",
    ):
        df[col] = df[col].astype(np.float32)
    df["n_pings"] = df["n_pings"].astype(np.int32)
    return df


# ── Happy path ──────────────────────────────────────────────────────


def test_fit_baselines_emits_one_row_per_cohort_metric() -> None:
    df = _segments_for(n_per_entity=50, entities=("v1", "v2"))
    out = fit_baselines(df, cohort_keys=["entity_id"], metrics=["duration_s"])
    assert len(out) == 2  # 2 entities × 1 metric
    assert set(out.columns) == {
        "entity_id",
        "metric",
        "mean",
        "std",
        "n_samples",
        "is_fallback",
    }


def test_fit_baselines_default_metrics_subset_of_available() -> None:
    df = _segments_for()
    out = fit_baselines(df, cohort_keys=["entity_id"])
    used = set(out["metric"].unique())
    assert used.issubset(set(DEFAULT_METRICS))
    # Should include duration_s at minimum
    assert "duration_s" in used


def test_fit_baselines_per_entity_means_differ() -> None:
    df = _segments_for(n_per_entity=100)
    out = fit_baselines(df, cohort_keys=["entity_id"], metrics=["duration_s"])
    means = {row.entity_id: row.mean for row in out.itertuples()}
    # Fixture gives v1 ~60s, v2 ~120s
    assert abs(means["v1"] - 60.0) < 5.0
    assert abs(means["v2"] - 120.0) < 5.0


def test_fit_baselines_n_samples_counts_per_cohort() -> None:
    df = _segments_for(n_per_entity=50)
    out = fit_baselines(df, cohort_keys=["entity_id"], metrics=["duration_s"])
    for row in out.itertuples():
        assert row.n_samples == 50


# ── Sample-count fallback ───────────────────────────────────────────


def test_fit_baselines_marks_undersized_cohort_as_fallback() -> None:
    """Cohort below min_cohort_n adopts global stats and is_fallback=True."""
    # v1 has 50 (above min_cohort_n=30); v2 has 5 (below)
    big = _segments_for(n_per_entity=50, entities=("v1",))
    small = _segments_for(n_per_entity=5, entities=("v2",), seed=1)
    df = pd.concat([big, small], ignore_index=True)
    out = fit_baselines(
        df, cohort_keys=["entity_id"], metrics=["duration_s"],
    )
    v1_row = out[out["entity_id"] == "v1"].iloc[0]
    v2_row = out[out["entity_id"] == "v2"].iloc[0]
    assert not v1_row["is_fallback"]
    assert v2_row["is_fallback"]
    # v2 inherits global stats (which include both v1 and v2 data)
    global_mean = float(df["duration_s"].mean())
    assert abs(float(v2_row["mean"]) - global_mean) < 1e-3


def test_fit_baselines_drops_metric_below_min_global_n() -> None:
    df = _segments_for(n_per_entity=2, entities=("v1",))
    out = fit_baselines(
        df,
        cohort_keys=["entity_id"],
        metrics=["duration_s"],
        params=BaselineParams(min_cohort_n=1, min_global_n=10),
    )
    assert len(out) == 0


def test_fit_baselines_skips_unknown_metrics_silently() -> None:
    df = _segments_for()
    out = fit_baselines(
        df, cohort_keys=["entity_id"], metrics=["duration_s", "fictional_col"]
    )
    used = set(out["metric"].unique())
    assert "duration_s" in used
    assert "fictional_col" not in used


# ── Multi-key cohort ────────────────────────────────────────────────


def test_fit_baselines_supports_compound_cohort_keys() -> None:
    df = _segments_for(n_per_entity=50, entities=("v1",))
    df.loc[:24, "segment_type"] = "MOVE"
    df.loc[25:, "segment_type"] = "STOP_DWELL"
    df["segment_type"] = df["segment_type"].astype("string")
    out = fit_baselines(
        df,
        cohort_keys=["entity_id", "segment_type"],
        metrics=["duration_s"],
        params=BaselineParams(min_cohort_n=10),
    )
    assert {*out.columns} == {
        "entity_id",
        "segment_type",
        "metric",
        "mean",
        "std",
        "n_samples",
        "is_fallback",
    }
    types_present = set(out["segment_type"].unique())
    assert types_present == {"MOVE", "STOP_DWELL"}


# ── Schema conformance ──────────────────────────────────────────────


def test_fit_baselines_output_validates_against_factory_schema() -> None:
    from trajkit.types import make_baselines_schema

    df = _segments_for()
    out = fit_baselines(df, cohort_keys=["entity_id"])
    make_baselines_schema(["entity_id"]).validate(out)


# ── Persistence ─────────────────────────────────────────────────────


def test_fit_baselines_writes_parquet_when_out_path_set(tmp_path: Path) -> None:
    df = _segments_for()
    out_file = tmp_path / "baselines.parquet"
    fit_baselines(df, cohort_keys=["entity_id"], out_path=out_file)
    assert out_file.exists()
    reread = pd.read_parquet(out_file)
    assert len(reread) > 0
    assert "metric" in reread.columns


def test_fit_baselines_reads_segments_from_parquet_directory(tmp_path: Path) -> None:
    df = _segments_for()
    src = tmp_path / "segments"
    df.to_parquet(src, partition_cols=["entity_id"])
    out = fit_baselines(src, cohort_keys=["entity_id"])
    assert len(out) > 0


def test_fit_baselines_raises_on_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fit_baselines(tmp_path / "nope", cohort_keys=["entity_id"])


# ── Argument validation ─────────────────────────────────────────────


def test_fit_baselines_rejects_missing_cohort_key() -> None:
    df = _segments_for()
    with pytest.raises(ValueError, match="cohort_keys not in source"):
        fit_baselines(df, cohort_keys=["nonexistent_col"])


def test_fit_baselines_rejects_empty_cohort_keys() -> None:
    df = _segments_for()
    with pytest.raises(ValueError, match="at least one column"):
        fit_baselines(df, cohort_keys=[])


def test_fit_baselines_rejects_no_available_metrics() -> None:
    df = _segments_for()
    with pytest.raises(ValueError, match="none of the requested metrics"):
        fit_baselines(df, cohort_keys=["entity_id"], metrics=["fictional_col"])


# ── Round-trip with baseline_zscores ────────────────────────────────


def test_fit_baselines_output_consumable_by_baseline_zscores() -> None:
    df = _segments_for()
    baselines = fit_baselines(
        df, cohort_keys=["entity_id"], metrics=["duration_s"]
    )
    out = baseline_zscores(df, baselines, cohort_keys=["entity_id"])
    assert "duration_s_z" in out.columns
    # The z-scores should be roughly mean-0 unit-variance per entity
    for _eid, group in out.groupby("entity_id"):
        z = group["duration_s_z"].astype(np.float64)
        assert abs(z.mean()) < 0.5
        assert abs(z.std() - 1.0) < 0.5


# ── Params plumbing ─────────────────────────────────────────────────


def test_baseline_params_are_frozen() -> None:
    p = BaselineParams()
    with pytest.raises(ValidationError):
        p.min_cohort_n = 999  # type: ignore[misc]


def test_baseline_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        BaselineParams(junk="oops")  # type: ignore[call-arg]
