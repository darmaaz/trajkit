"""Sanity tests for ``trajkit.testing`` builders.

The builders are themselves test fixtures, so the tests here are
deliberately small: validate schema conformance, basic shape, and the
motion options each produce something distinguishable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trajkit import types as tt
from trajkit.clean import clean
from trajkit.episode import detect_episodes
from trajkit.runner import process
from trajkit.segment import aggregate_segments, segment
from trajkit.testing import make_pings, make_segments

# ── make_pings ──────────────────────────────────────────────────────


def test_make_pings_validates_against_pings_schema() -> None:
    df = make_pings(60)
    tt.PingsSchema.validate(df)


def test_make_pings_canonical_dtypes() -> None:
    df = make_pings(60)
    assert df["entity_id"].dtype == "string"
    assert str(df["ts"].dtype) == "datetime64[ns, UTC]"
    assert df["lat"].dtype == np.float64
    assert df["lon"].dtype == np.float64
    assert df["speed_ms"].dtype == np.float32


def test_make_pings_default_motion_advances_position() -> None:
    df = make_pings(60, motion="linear")
    assert df["lat"].iloc[-1] > df["lat"].iloc[0]


def test_make_pings_stationary_does_not_advance() -> None:
    df = make_pings(60, motion="stationary")
    assert (df["lat"] == df["lat"].iloc[0]).all()


def test_make_pings_stop_then_move_has_two_phases() -> None:
    df = make_pings(60, motion="stop_then_move")
    half = len(df) // 2
    # First half stationary
    assert (df["lat"].iloc[:half] == df["lat"].iloc[0]).all()
    # Second half advancing
    assert df["lat"].iloc[-1] > df["lat"].iloc[half]


def test_make_pings_zero_n_returns_empty_frame() -> None:
    df = make_pings(0)
    assert len(df) == 0


def test_make_pings_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_pings(-1)


def test_make_pings_rejects_unknown_motion() -> None:
    with pytest.raises(ValueError, match="unknown motion"):
        make_pings(10, motion="warp")  # type: ignore[arg-type]


def test_make_pings_runs_through_full_pipeline() -> None:
    """End-to-end sanity: pings → clean → segment → aggregate → episode."""
    df = make_pings(300, motion="linear")
    cleaned = clean(df)
    seg = segment(cleaned)
    agg = aggregate_segments(seg)
    eps = detect_episodes(agg)
    # End-to-end produced something validatable
    tt.SegmentsSchema.validate(agg)
    tt.EpisodesSchema.validate(eps)


# ── make_segments ───────────────────────────────────────────────────


def test_make_segments_validates_against_segments_schema() -> None:
    df = make_segments(5)
    tt.SegmentsSchema.validate(df)


def test_make_segments_canonical_dtypes() -> None:
    df = make_segments(5)
    assert df["segment_id"].dtype == "string"
    assert df["entity_id"].dtype == "string"
    assert str(df["start_ts"].dtype) == "datetime64[ns, UTC]"
    assert df["duration_s"].dtype == np.float32
    assert df["n_pings"].dtype == np.int32


def test_make_segments_segment_ids_are_unique_and_zero_padded() -> None:
    df = make_segments(5)
    sids = df["segment_id"].astype(str).tolist()
    assert len(set(sids)) == 5
    for sid in sids:
        suffix = sid.split("_")[-1]
        assert len(suffix) == 5


def test_make_segments_all_stationary_emits_only_stop_dwell() -> None:
    df = make_segments(5, motion="all_stationary")
    assert (df["segment_type"] == "STOP_DWELL").all()


def test_make_segments_all_moving_emits_only_move() -> None:
    df = make_segments(5, motion="all_moving")
    assert (df["segment_type"] == "MOVE").all()


def test_make_segments_alternating_alternates_types() -> None:
    df = make_segments(6, motion="alternating")
    types = df["segment_type"].tolist()
    assert types == ["MOVE", "STOP_DWELL", "MOVE", "STOP_DWELL", "MOVE", "STOP_DWELL"]


def test_make_segments_zero_n_returns_empty_frame() -> None:
    df = make_segments(0)
    assert len(df) == 0


def test_make_segments_rejects_unknown_motion() -> None:
    with pytest.raises(ValueError, match="unknown segments motion"):
        make_segments(3, motion="warp")  # type: ignore[arg-type]


def test_make_segments_feeds_episode_detection() -> None:
    """End-to-end: segments → episodes round-trip works."""
    seg = make_segments(10, motion="all_stationary", duration_s=60.0)
    eps = detect_episodes(seg)
    tt.EpisodesSchema.validate(eps)
    assert (eps["episode_type"] == "STAY").any()


# ── End-to-end through the runner ───────────────────────────────────


def test_make_pings_runs_through_runner(tmp_path: object) -> None:
    """Quickstart-shape sanity: synthetic pings flow through the full runner."""
    df = make_pings(300, motion="linear")
    rep = process(df, tmp_path)  # type: ignore[arg-type]
    assert rep.succeeded
    assert rep.n_completed == 1


# ── Multi-entity composition ────────────────────────────────────────


def test_make_pings_concatenated_for_multi_entity() -> None:
    """Users compose multiple builder calls for multi-entity fixtures."""
    a = make_pings(20, entity_id="v1", motion="linear")
    b = make_pings(20, entity_id="v2", motion="stationary",
                  start_ts=pd.Timestamp("2026-01-02", tz="UTC"))
    combined = pd.concat([a, b], ignore_index=True)
    # Both entities present and validatable per-entity
    assert sorted(combined["entity_id"].astype(str).unique()) == ["v1", "v2"]
