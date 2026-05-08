"""Unit tests for ``trajkit.episode``.

The episode layer is greenfield (no fleet reference). Tests cover the
edge cases enumerated in ``docs/design/episode.md`` plus end-to-end
schema conformance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit import types as tt
from trajkit.episode import EpisodeParams, detect_episodes

# ── Fixture builders ────────────────────────────────────────────────


def _ts(start: str, n: int, freq: str = "60s") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC").astype(
        "datetime64[ns, UTC]"
    )


def _segment_row(
    *,
    seg_idx: int,
    start_ts: pd.Timestamp,
    duration_s: float,
    lat: float,
    lon: float,
    segment_type: str = "STOP_DWELL",
    entity_id: str = "v1",
) -> dict[str, object]:
    """Build one SegmentsSchema row at a single (lat, lon) point."""
    end_ts = start_ts + pd.Timedelta(seconds=duration_s)
    return {
        "segment_id": f"{entity_id}_seg_{seg_idx:05d}",
        "entity_id": entity_id,
        "segment_type": segment_type,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": np.float32(duration_s),
        "start_lat": np.float64(lat),
        "start_lon": np.float64(lon),
        "end_lat": np.float64(lat),
        "end_lon": np.float64(lon),
        "start_h3": "8a2a1072b59ffff",
        "end_h3": "8a2a1072b59ffff",
        "path_length_m": np.float32(0.0),
        "displacement_m": np.float32(0.0),
        "straightness": np.float32(0.0),
        "mean_speed_ms": np.float32(0.0),
        "max_speed_ms": np.float32(0.0),
        "bearing_variance": np.float32(0.0),
        "n_pings": np.int32(60),
    }


def _segments_from_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a SegmentsSchema-compatible frame from prepared rows."""
    df = pd.DataFrame(rows)
    df["segment_id"] = df["segment_id"].astype("string")
    df["entity_id"] = df["entity_id"].astype("string")
    df["segment_type"] = df["segment_type"].astype("string")
    df["start_h3"] = df["start_h3"].astype("string")
    df["end_h3"] = df["end_h3"].astype("string")
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True).astype("datetime64[ns, UTC]")
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _stationary_run(
    n: int, *, lat: float = 19.4, lon: float = -99.2, start_idx: int = 1,
    start_ts: pd.Timestamp | None = None, duration_s: float = 60.0,
) -> list[dict[str, object]]:
    """N consecutive segments stationary at (lat, lon), each ``duration_s`` long."""
    if start_ts is None:
        start_ts = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for k in range(n):
        rows.append(_segment_row(
            seg_idx=start_idx + k,
            start_ts=start_ts + pd.Timedelta(seconds=k * duration_s),
            duration_s=duration_s,
            lat=lat, lon=lon,
        ))
    return rows


# ── Smoke + contract ────────────────────────────────────────────────


def test_detect_episodes_returns_episodesschema_compatible_frame() -> None:
    df = _segments_from_rows(_stationary_run(10))
    out = detect_episodes(df)
    tt.EpisodesSchema.validate(out)


def test_detect_episodes_handles_empty_input() -> None:
    df = pd.DataFrame(
        {
            "segment_id": pd.Series([], dtype="string"),
            "entity_id": pd.Series([], dtype="string"),
            "segment_type": pd.Series([], dtype="string"),
            "start_ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "end_ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "duration_s": pd.Series([], dtype=np.float32),
            "start_lat": pd.Series([], dtype=np.float64),
            "start_lon": pd.Series([], dtype=np.float64),
            "end_lat": pd.Series([], dtype=np.float64),
            "end_lon": pd.Series([], dtype=np.float64),
            "start_h3": pd.Series([], dtype="string"),
            "end_h3": pd.Series([], dtype="string"),
            "path_length_m": pd.Series([], dtype=np.float32),
            "displacement_m": pd.Series([], dtype=np.float32),
            "straightness": pd.Series([], dtype=np.float32),
            "mean_speed_ms": pd.Series([], dtype=np.float32),
            "max_speed_ms": pd.Series([], dtype=np.float32),
            "bearing_variance": pd.Series([], dtype=np.float32),
            "n_pings": pd.Series([], dtype=np.int32),
        }
    )
    out = detect_episodes(df)
    assert len(out) == 0
    tt.EpisodesSchema.validate(out)


def test_detect_episodes_does_not_mutate_input() -> None:
    df = _segments_from_rows(_stationary_run(10))
    snapshot = df.copy(deep=True)
    _ = detect_episodes(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_detect_episodes_default_params_used_when_none() -> None:
    df = _segments_from_rows(_stationary_run(10))
    a = detect_episodes(df)
    b = detect_episodes(df, EpisodeParams())
    pd.testing.assert_frame_equal(a, b)


# ── Single-stay case ────────────────────────────────────────────────


def test_detect_episodes_single_stay_at_same_location() -> None:
    df = _segments_from_rows(_stationary_run(10, duration_s=60.0))
    out = detect_episodes(df)
    assert len(out) == 1
    assert out["episode_type"].iloc[0] == "STAY"
    assert pd.notna(out["anchor_lat"].iloc[0])
    assert pd.isna(out["start_lat"].iloc[0])  # transit-only column null


def test_detect_episodes_stay_anchor_near_centroids() -> None:
    df = _segments_from_rows(_stationary_run(10, lat=19.4, lon=-99.2))
    out = detect_episodes(df)
    assert abs(float(out["anchor_lat"].iloc[0]) - 19.4) < 1e-6
    assert abs(float(out["anchor_lon"].iloc[0]) - (-99.2)) < 1e-6


# ── Single-transit case ─────────────────────────────────────────────


def test_detect_episodes_single_transit_when_no_stay_qualifies() -> None:
    """A trace shorter than min_stay_s overall yields one TRANSIT (gotcha #1)."""
    # 2 segments, each 30s — total 60s < default min_stay_s=180s
    rows = _stationary_run(2, duration_s=30.0)
    df = _segments_from_rows(rows)
    out = detect_episodes(df)
    assert len(out) == 1
    assert out["episode_type"].iloc[0] == "TRANSIT"
    assert pd.notna(out["start_lat"].iloc[0])
    assert pd.isna(out["anchor_lat"].iloc[0])


# ── Stay → transit → stay ───────────────────────────────────────────


def test_detect_episodes_stay_transit_stay_sequence() -> None:
    """Three blocks: long stay at A, fast traverse, long stay at B (~5km away)."""
    block_a = _stationary_run(
        4, lat=19.4, lon=-99.2, start_idx=1, duration_s=60.0,
        start_ts=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
    )
    transit_seg = [
        _segment_row(
            seg_idx=10,
            start_ts=pd.Timestamp("2026-01-01 00:04:00", tz="UTC"),
            duration_s=120.0,
            lat=19.42, lon=-99.2,  # 2.2 km away — outside R
            segment_type="MOVE",
        )
    ]
    block_b = _stationary_run(
        4, lat=19.45, lon=-99.2, start_idx=20, duration_s=60.0,
        start_ts=pd.Timestamp("2026-01-01 00:06:00", tz="UTC"),
    )
    df = _segments_from_rows(block_a + transit_seg + block_b)
    # Use a small T_s so the transit segment doesn't get absorbed into block_a's stay
    out = detect_episodes(df, EpisodeParams(R_m=200.0, T_s=60.0, min_stay_s=180.0))
    types = out["episode_type"].tolist()
    assert types == ["STAY", "TRANSIT", "STAY"]


# ── Sub-min-stay rejection ──────────────────────────────────────────


def test_detect_episodes_rejects_sub_min_stay_candidates() -> None:
    """Segments form a tight cluster but total duration < min_stay_s."""
    rows = _stationary_run(2, duration_s=30.0)  # 60s total, min_stay_s=180
    df = _segments_from_rows(rows)
    out = detect_episodes(df)
    assert (out["episode_type"] != "STAY").all()


# ── Trace gap closes a stay ─────────────────────────────────────────


def test_detect_episodes_trace_gap_closes_stay() -> None:
    """A multi-hour gap between segments at the same place forces two stays."""
    block_a = _stationary_run(
        4, lat=19.4, lon=-99.2, start_idx=1, duration_s=60.0,
        start_ts=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
    )
    block_b = _stationary_run(
        4, lat=19.4, lon=-99.2, start_idx=10, duration_s=60.0,
        # 2-hour gap after block_a ends
        start_ts=pd.Timestamp("2026-01-01 02:04:00", tz="UTC"),
    )
    df = _segments_from_rows(block_a + block_b)
    # Default T_s=300 → 2-hour gap > T_s → split into two stays
    out = detect_episodes(df)
    assert (out["episode_type"] == "STAY").sum() == 2


# ── Trace gap splits a transit ──────────────────────────────────────


def test_detect_episodes_trace_gap_splits_transit() -> None:
    """A long inter-segment gap inside a non-stay run splits the transit."""
    s1 = _segment_row(
        seg_idx=1,
        start_ts=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        duration_s=30.0, lat=19.4, lon=-99.2, segment_type="MOVE",
    )
    s2 = _segment_row(
        seg_idx=2,
        # 1-hour gap after s1 ends
        start_ts=pd.Timestamp("2026-01-01 01:00:30", tz="UTC"),
        duration_s=30.0, lat=19.41, lon=-99.21, segment_type="MOVE",
    )
    df = _segments_from_rows([s1, s2])
    out = detect_episodes(df)
    # 2 transits, no stays (durations too short to qualify as stays)
    assert (out["episode_type"] == "TRANSIT").sum() == 2


# ── Boundary oscillation absorbed by grace window ───────────────────


def test_detect_episodes_absorbs_brief_excursion_into_stay() -> None:
    """Drove around the block within T_s grace window → still one stay."""
    # Long stay at depot, single brief excursion 1 km away, return.
    excursion_segs = [
        _segment_row(
            seg_idx=5,
            start_ts=pd.Timestamp("2026-01-01 00:04:00", tz="UTC"),
            duration_s=60.0,
            lat=19.41, lon=-99.2,  # 1.1 km away — outside R=200m
            segment_type="MOVE",
        )
    ]
    block_pre = _stationary_run(
        4, lat=19.4, lon=-99.2, start_idx=1, duration_s=60.0,
        start_ts=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
    )
    block_post = _stationary_run(
        5, lat=19.4, lon=-99.2, start_idx=10, duration_s=60.0,
        start_ts=pd.Timestamp("2026-01-01 00:05:00", tz="UTC"),
    )
    df = _segments_from_rows(block_pre + excursion_segs + block_post)
    # Excursion duration (60s) < default T_s=300, so absorbed into stay.
    out = detect_episodes(df)
    stays = out[out["episode_type"] == "STAY"]
    assert len(stays) == 1


# ── Running anchor: parking-then-drift ──────────────────────────────


def test_detect_episodes_running_anchor_handles_drift() -> None:
    """Centroid drifts within R; single stay covers the drift."""
    # 10 segments inching forward by ~10m per segment but staying within R=200m.
    base_ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    rows = [
        _segment_row(
            seg_idx=k + 1,
            start_ts=base_ts + pd.Timedelta(seconds=k * 60),
            duration_s=60.0,
            lat=19.4 + k * 0.0001,  # ~11m per segment
            lon=-99.2,
        )
        for k in range(10)
    ]
    df = _segments_from_rows(rows)
    out = detect_episodes(df)
    stays = out[out["episode_type"] == "STAY"]
    assert len(stays) == 1
    # envelope_radius_m should reflect the drift extent
    assert float(stays["envelope_radius_m"].iloc[0]) > 30.0


# ── Two stays close but distinct ────────────────────────────────────


def test_detect_episodes_two_distinct_stays_when_separated_beyond_radius() -> None:
    """Two stationary clusters > R apart yield two separate stays."""
    block_a = _stationary_run(
        4, lat=19.400, lon=-99.200, start_idx=1, duration_s=60.0,
        start_ts=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
    )
    block_b = _stationary_run(
        4, lat=19.405, lon=-99.200, start_idx=10, duration_s=60.0,
        # ~550m away — beyond R=200m
        start_ts=pd.Timestamp("2026-01-01 00:04:00", tz="UTC"),
    )
    df = _segments_from_rows(block_a + block_b)
    # Use small T_s so the gap between blocks doesn't bridge them
    out = detect_episodes(
        df, EpisodeParams(R_m=200.0, T_s=60.0, min_stay_s=180.0)
    )
    assert (out["episode_type"] == "STAY").sum() == 2


# ── Episode IDs and ordering ────────────────────────────────────────


def test_detect_episodes_ids_are_temporally_ordered() -> None:
    df = _segments_from_rows(_stationary_run(10))
    out = detect_episodes(df)
    assert (out["start_ts"].diff().dropna() >= pd.Timedelta(0)).all()
    # IDs are zero-padded suffix
    for eid in out["episode_id"]:
        suffix = eid.split("_")[-1]
        assert len(suffix) == 5


def test_detect_episodes_segment_ids_are_lists_of_strings() -> None:
    df = _segments_from_rows(_stationary_run(10))
    out = detect_episodes(df)
    for sids in out["segment_ids"]:
        assert isinstance(sids, list)
        assert all(isinstance(x, str) for x in sids)


def test_detect_episodes_n_segments_matches_segment_ids_length() -> None:
    df = _segments_from_rows(_stationary_run(10))
    out = detect_episodes(df)
    for _, row in out.iterrows():
        assert int(row["n_segments"]) == len(row["segment_ids"])


# ── Params ──────────────────────────────────────────────────────────


def test_episode_params_are_frozen() -> None:
    p = EpisodeParams()
    with pytest.raises(ValidationError):
        p.R_m = 999.0  # type: ignore[misc]


def test_episode_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        EpisodeParams(junk="oops")  # type: ignore[call-arg]


def test_episode_params_reject_zero_or_negative_radius() -> None:
    with pytest.raises(ValidationError):
        EpisodeParams(R_m=0.0)
    with pytest.raises(ValidationError):
        EpisodeParams(R_m=-1.0)
