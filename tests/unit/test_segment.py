"""Unit tests for ``trajkit.segment``.

Builds synthetic pings → ``clean`` → ``segment`` → ``aggregate_segments``
and exercises each behaviour end-to-end. The state machine, bearing
detector, and classification logic are covered through the public API
rather than through private helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit import types as tt
from trajkit.clean import clean
from trajkit.segment import SegmentParams, aggregate_segments, segment

# ── Fixture helpers ─────────────────────────────────────────────────


def _ts(start: str, n: int, freq: str = "1s") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC").astype(
        "datetime64[ns, UTC]"
    )


def _pings(
    n: int,
    *,
    lat_path: np.ndarray | None = None,
    lon_path: np.ndarray | None = None,
    speed_ms: np.ndarray | None = None,
    bearing_deg: np.ndarray | None = None,
    freq: str = "1s",
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Build a ``PingsSchema``-shaped frame for one entity."""
    if lat_path is None:
        lat_path = np.full(n, 19.4, dtype=np.float64)
    if lon_path is None:
        lon_path = np.full(n, -99.2, dtype=np.float64)
    if speed_ms is None:
        speed_ms = np.full(n, np.nan, dtype=np.float32)
    if bearing_deg is None:
        bearing_deg = np.full(n, np.nan, dtype=np.float32)
    return pd.DataFrame(
        {
            "entity_id": pd.Series(["v1"] * n, dtype="string"),
            "ts": _ts(start, n, freq),
            "lat": lat_path.astype(np.float64),
            "lon": lon_path.astype(np.float64),
            "speed_ms": speed_ms.astype(np.float32),
            "bearing_deg": bearing_deg.astype(np.float32),
        }
    )


def _moving_pings(n: int, lat_step: float = 0.0001) -> pd.DataFrame:
    """Steady straight-line motion at ~36 km/h (well below outlier threshold)."""
    lat_path = 19.4 + np.arange(n) * lat_step
    return _pings(n, lat_path=lat_path)


def _stationary_pings(n: int) -> pd.DataFrame:
    """All pings at the same position; no motion."""
    return _pings(n)


# ── segment(): smoke + contract ─────────────────────────────────────


def test_segment_returns_segmentedpingsschema_compatible_frame() -> None:
    cleaned = clean(_moving_pings(20))
    out = segment(cleaned)
    tt.SegmentedPingsSchema.validate(out)


def test_segment_empty_input_returns_empty_frame() -> None:
    cleaned = clean(_pings(0))
    out = segment(cleaned)
    assert len(out) == 0
    tt.SegmentedPingsSchema.validate(out)


def test_segment_does_not_mutate_input() -> None:
    cleaned = clean(_moving_pings(20))
    snapshot = cleaned.copy(deep=True)
    _ = segment(cleaned)
    pd.testing.assert_frame_equal(cleaned, snapshot)


def test_segment_default_params_used_when_none() -> None:
    cleaned = clean(_moving_pings(20))
    a = segment(cleaned)
    b = segment(cleaned, SegmentParams())
    pd.testing.assert_frame_equal(a, b)


def test_segment_id_is_zero_padded_for_lexicographic_sort() -> None:
    cleaned = clean(_moving_pings(20))
    out = segment(cleaned)
    seg_ids = out["segment_id"].unique().tolist()
    # Every segment_id matches the canonical 5-digit suffix
    for sid in seg_ids:
        assert sid.startswith("v1_seg_")
        assert len(sid.split("_")[-1]) == 5


# ── State machine: stop / move / hysteresis ─────────────────────────


def test_segment_classifies_long_stationary_run_as_stop_dwell() -> None:
    # 10 minutes of stationary pings — well above default dwell_threshold_min=5
    cleaned = clean(_stationary_pings(600))  # 600s = 10 min
    out = segment(cleaned)
    assert (out["segment_type"] == "STOP_DWELL").all()


def test_segment_classifies_short_stationary_run_as_stop_brief() -> None:
    # 60 seconds of stationary pings — below dwell_threshold_min=5min
    # but above stop_min_duration_s=30s, so it's a real stop
    pings = pd.concat(
        [
            _moving_pings(20),
            _pings(
                60, freq="1s", start="2026-01-01 00:00:20",
            ).assign(lat=19.402, lon=-99.198),
            _moving_pings(20).assign(
                ts=_ts("2026-01-01 00:01:20", 20),
                lat=19.402 + np.arange(20) * 0.0001,
                lon=-99.198,
            ),
        ],
        ignore_index=True,
    )
    cleaned = clean(pings)
    out = segment(cleaned)
    types = set(out["segment_type"].unique())
    assert "STOP_BRIEF" in types


def test_segment_classifies_continuous_movement_as_move() -> None:
    cleaned = clean(_moving_pings(20))
    out = segment(cleaned)
    # All pings are moving at ~36 km/h
    move_count = (out["segment_type"] == "MOVE").sum()
    move_brief_count = (out["segment_type"] == "MOVE_BRIEF").sum()
    # Either MOVE or MOVE_BRIEF depending on duration thresholds
    assert (move_count + move_brief_count) == len(out)


def test_segment_marks_short_movement_as_move_brief() -> None:
    # 3 pings of motion at 1 Hz — fewer than move_brief_min_pings=5
    cleaned = clean(_moving_pings(3))
    out = segment(cleaned)
    assert (out["segment_type"] == "MOVE_BRIEF").all()


def test_segment_filters_short_stop_runs_as_moving() -> None:
    # Move 20 → stop 5s (< stop_min_duration_s=30) → move 20.
    # The 5-second stop should be absorbed into the surrounding motion.
    pings = pd.concat(
        [
            _moving_pings(20),
            _pings(5, start="2026-01-01 00:00:20").assign(lat=19.402, lon=-99.198),
            _pings(
                20, start="2026-01-01 00:00:25",
                lat_path=19.402 + np.arange(20) * 0.0001,
                lon_path=np.full(20, -99.198),
            ),
        ],
        ignore_index=True,
    )
    cleaned = clean(pings)
    out = segment(cleaned)
    # No STOP_BRIEF for the 5-second pause
    assert "STOP_BRIEF" not in set(out["segment_type"].unique())


# ── State change creates segment boundary ────────────────────────────


def test_segment_creates_boundary_at_motion_state_transition() -> None:
    # Stationary then moving → at least 2 segments
    pings = pd.concat(
        [
            _stationary_pings(60),  # 60s dwell
            _pings(
                30, start="2026-01-01 00:01:00",
                lat_path=19.4 + np.arange(30) * 0.0001,
            ),
        ],
        ignore_index=True,
    )
    cleaned = clean(pings)
    out = segment(cleaned)
    n_segments = out["segment_id"].nunique()
    assert n_segments >= 2


# ── Sustained bearing change ────────────────────────────────────────


def test_segment_splits_move_on_sustained_bearing_change() -> None:
    """The circular-R detector fires on a continuously-turning trajectory.

    A spiral causes bearings to rotate uniformly; ``R`` over any
    distance window covering many spiral pings is low (bearings spread
    around the unit circle). After ``bearing_sustain_m`` of low R, the
    boundary fires.
    """
    n_straight = 120
    n_spiral = 120

    straight_lat = 19.4 + np.arange(n_straight) * 0.0001
    straight_lon = np.full(n_straight, -99.2)

    # Spiral: each ping the bearing rotates by ~10°. We construct lat/lon so
    # consecutive bearings differ by ≈10°.
    radius = 0.0005  # degrees, ~55m
    angles = np.deg2rad(np.arange(n_spiral) * 10.0)
    spiral_lat = straight_lat[-1] + radius * np.sin(angles)
    spiral_lon = straight_lon[-1] + radius * np.cos(angles)

    pings = pd.concat(
        [
            _pings(n_straight, lat_path=straight_lat, lon_path=straight_lon),
            _pings(
                n_spiral,
                lat_path=spiral_lat,
                lon_path=spiral_lon,
                start="2026-01-01 00:02:00",
            ),
        ],
        ignore_index=True,
    )
    cleaned = clean(pings)
    out = segment(cleaned)  # default bearing params suffice
    assert out["segment_id"].nunique() >= 2


def test_segment_does_not_split_on_single_sharp_turn() -> None:
    """A 90° corner with straight walking before/after should NOT fire bearing.

    The circular-R detector requires *sustained* low R (low R for at
    least ``bearing_sustain_m`` of trajectory). A single 1-ping turn
    surrounded by long straight stretches keeps R high in any window.
    Boundary should not fire on bearing alone.
    """
    n_pre = 200
    n_post = 200
    pre_lat = 19.4 + np.arange(n_pre) * 0.0001  # walking north
    pre_lon = np.full(n_pre, -99.2)
    # Sharp 90° turn: continue from end of pre, now walking east
    post_lat = np.full(n_post, pre_lat[-1])
    post_lon = pre_lon[-1] + np.arange(1, n_post + 1) * 0.0001
    pings = pd.concat(
        [
            _pings(n_pre, lat_path=pre_lat, lon_path=pre_lon),
            _pings(
                n_post,
                lat_path=post_lat,
                lon_path=post_lon,
                start="2026-01-01 00:03:20",
            ),
        ],
        ignore_index=True,
    )
    cleaned = clean(pings)
    out = segment(cleaned)
    # We expect zero bearing-induced splits: the only acceptable boundaries
    # are state-change (none here, all moving) or gap (none here). Total
    # segments should be ≤ 2 (one for the whole motion, possibly an
    # initial null-bearing first ping).
    assert out["segment_id"].nunique() <= 2


def test_segment_does_not_split_on_brief_bearing_spike() -> None:
    # Single-ping bearing spike inside an otherwise-straight leg should not
    # fire a boundary because of the time-sustainment requirement.
    n = 600  # 10 minutes
    lat_path = 19.4 + np.arange(n) * 0.0001
    lon_path = np.full(n, -99.2)
    # Inject one anomalous lat at index 300
    lat_path[300] += 0.001
    pings = _pings(n, lat_path=lat_path, lon_path=lon_path)
    cleaned = clean(pings)
    out = segment(cleaned)
    # Without sustainment, this would split. With it, ≤ 2 segments
    # (at most a short MOVE_BRIEF if the spike triggers SPEED_OUTLIER).
    assert out["segment_id"].nunique() <= 3


# ── False-stop override ─────────────────────────────────────────────


def test_segment_overrides_stop_with_significant_displacement() -> None:
    # Crawl forward at ~1 km/h for 10 minutes — below stop_speed_kmh=2,
    # so hysteresis classifies as stopped. But total displacement is
    # ~166m (10 min * 1 km/h * 1000/60). Still below max_stop_displacement_m=500.
    # To trigger override, crawl over 1 km — adjust lat step so total
    # displacement crosses the threshold.
    n = 600
    # ~1 m per ping → 600 m total over 10 min
    lat_path = 19.4 + np.arange(n) * (1e-5 * 1.5)  # ~1.5m per ping
    pings = _pings(n, lat_path=lat_path)
    cleaned = clean(pings)
    out = segment(cleaned)
    types = set(out["segment_type"].unique())
    # No STOP_DWELL because crow-fly displacement triggers the override
    assert "STOP_DWELL" not in types


# ── GAP_FOLLOWS creates segment boundary ────────────────────────────


def test_segment_creates_boundary_at_gap() -> None:
    """A GAP_FOLLOWS ping creates a new segment boundary.

    The gap fixture must place the entity far enough during the gap that
    derived speed is above the drift threshold (otherwise DRIFT claims the
    row first per the precedence) and below the outlier threshold.
    """
    n = 20
    pings = _moving_pings(n)
    # 10-minute gap before row 10, with the entity having moved ~1.1 km
    # during the gap. dt=600s, displacement≈1100m → speed≈1.8 m/s (~6.6 km/h),
    # above drift but well below outlier.
    pings.loc[10:, "ts"] = pings.loc[10:, "ts"] + pd.Timedelta("10min")
    pings.loc[10:, "lat"] = pings.loc[10:, "lat"] + 0.01
    cleaned = clean(pings)
    assert (cleaned["quality_flag"] == "GAP_FOLLOWS").any()
    out = segment(cleaned)
    assert out["segment_id"].nunique() >= 2


# ── aggregate_segments ──────────────────────────────────────────────


def test_aggregate_segments_returns_segmentsschema_compatible_frame() -> None:
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    tt.SegmentsSchema.validate(out)


def test_aggregate_segments_handles_empty_input() -> None:
    seg = segment(clean(_pings(0)))
    out = aggregate_segments(seg)
    assert len(out) == 0
    tt.SegmentsSchema.validate(out)


def test_aggregate_segments_one_row_per_segment_id() -> None:
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    assert len(out) == seg["segment_id"].nunique()


def test_aggregate_segments_straightness_in_unit_interval() -> None:
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    assert (out["straightness"] >= 0.0).all()
    assert (out["straightness"] <= 1.0).all()


def test_aggregate_segments_path_length_geq_displacement() -> None:
    """Path length must be ≥ great-circle displacement (triangle inequality)."""
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    # Allow tiny float-precision tolerance
    assert (out["path_length_m"] + 1e-3 >= out["displacement_m"]).all()


def test_aggregate_segments_h3_cells_populated() -> None:
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    assert out["start_h3"].notna().all()
    assert out["end_h3"].notna().all()


def test_aggregate_segments_n_pings_matches_row_count_without_merge() -> None:
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    expected = seg.groupby("segment_id", sort=False).size()
    actual = out.set_index("segment_id")["n_pings"]
    assert (actual == expected).all()


def test_aggregate_segments_bearing_variance_in_unit_interval() -> None:
    cleaned = clean(_moving_pings(60))
    seg = segment(cleaned)
    out = aggregate_segments(seg)
    # bearing_variance is nullable; check the non-null subset
    bv = out["bearing_variance"].dropna()
    assert (bv >= 0.0).all()
    assert (bv <= 1.0).all()


# ── Params plumbing ─────────────────────────────────────────────────


def test_segment_params_are_frozen() -> None:
    p = SegmentParams()
    with pytest.raises(ValidationError):
        p.stop_speed_kmh = 999.0  # type: ignore[misc]


def test_segment_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SegmentParams(junk="oops")  # type: ignore[call-arg]


def test_segment_params_reject_invalid_hysteresis_order() -> None:
    with pytest.raises(ValidationError):
        SegmentParams(stop_speed_kmh=10.0, resume_speed_kmh=5.0)
