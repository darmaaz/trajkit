"""Unit tests for ``trajkit.clean``.

Coverage philosophy: each public flag/behaviour has a positive case
(input that should trigger it) and a negative case (similar input that
should not). Synthetic fixtures only — no real fleet data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit import types as tt
from trajkit.clean import (
    CleanParams,
    StaleMergeParams,
    clean,
    detect_stale_pattern,
    merge_stale_positions,
)

# ── Fixture helpers ─────────────────────────────────────────────────


def _ts(start: str, n: int, freq: str = "1s") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC").astype(
        "datetime64[ns, UTC]"
    )


def _make_pings(
    n: int,
    lat_path: np.ndarray | None = None,
    lon_path: np.ndarray | None = None,
    speed_ms: np.ndarray | None = None,
    bearing_deg: np.ndarray | None = None,
    freq: str = "1s",
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Build a single-entity PingsSchema-shaped frame.

    Defaults to a stationary trace at (19.4, -99.2). Caller overrides
    lat_path/lon_path to introduce motion.
    """
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


# ── Smoke + schema conformance ──────────────────────────────────────


def test_clean_returns_cleanedpingsschema_compatible_frame() -> None:
    df = _make_pings(5)
    out = clean(df)
    tt.CleanedPingsSchema.validate(out)


def test_clean_empty_input_returns_empty_compatible_frame() -> None:
    empty = _make_pings(0)
    out = clean(empty)
    assert len(out) == 0
    tt.CleanedPingsSchema.validate(out)


def test_clean_does_not_mutate_input() -> None:
    df = _make_pings(5)
    snapshot = df.copy(deep=True)
    _ = clean(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_clean_default_params_used_when_none() -> None:
    df = _make_pings(5)
    a = clean(df)
    b = clean(df, CleanParams())
    pd.testing.assert_frame_equal(a, b)


# ── Dedup ───────────────────────────────────────────────────────────


def test_clean_drops_exact_duplicate_rows() -> None:
    df = _make_pings(3)
    dup = pd.concat([df, df.iloc[[1]]], ignore_index=True).sort_values("ts")
    dup = dup.reset_index(drop=True)
    out = clean(dup)
    assert len(out) == 3


# ── Kinematics derivation ───────────────────────────────────────────


def test_clean_first_row_has_null_kinematics() -> None:
    df = _make_pings(3, lat_path=np.array([19.4, 19.41, 19.42]))
    out = clean(df)
    assert pd.isna(out.loc[0, "dt_seconds"])
    assert pd.isna(out.loc[0, "displacement_m"])
    assert pd.isna(out.loc[0, "speed_ms"])
    assert pd.isna(out.loc[0, "bearing_deg"])


def test_clean_derives_displacement_for_moving_entity() -> None:
    # ~11m per 0.0001 deg of latitude → ~40 km/h, well below default outlier
    lats = np.array([19.4, 19.4001, 19.4002])
    df = _make_pings(3, lat_path=lats)
    out = clean(df)
    assert out.loc[1, "displacement_m"] > 9.0
    assert out.loc[1, "displacement_m"] < 13.0


def test_clean_derives_speed_consistent_with_displacement() -> None:
    lats = np.array([19.4, 19.4001, 19.4002])
    df = _make_pings(3, lat_path=lats, freq="1s")
    out = clean(df)
    expected_speed = float(out.loc[1, "displacement_m"]) / 1.0
    assert abs(float(out.loc[1, "speed_ms"]) - expected_speed) < 1e-3


# ── SPEED_OUTLIER ───────────────────────────────────────────────────


def test_clean_flags_speed_outlier_for_implausible_jump() -> None:
    # Jump 2 deg of latitude (~222 km) in 1 second → ~800,000 km/h
    df = _make_pings(3, lat_path=np.array([19.4, 21.4, 21.401]))
    out = clean(df)
    assert (out["quality_flag"] == "SPEED_OUTLIER").any()


def test_clean_nulls_outlier_and_next_row_kinematics() -> None:
    df = _make_pings(4, lat_path=np.array([19.4, 21.4, 21.401, 21.402]))
    out = clean(df)
    # Row 1 is the outlier (B), row 2 is the tainted neighbour (C).
    assert pd.isna(out.loc[1, "speed_ms"])
    assert pd.isna(out.loc[2, "speed_ms"])
    assert pd.isna(out.loc[1, "displacement_m"])
    assert pd.isna(out.loc[2, "displacement_m"])


def test_clean_does_not_flag_speed_below_threshold() -> None:
    df = _make_pings(3, lat_path=np.array([19.4, 19.4001, 19.4002]))
    out = clean(df)
    assert (out["quality_flag"] != "SPEED_OUTLIER").all()


# ── DRIFT ───────────────────────────────────────────────────────────


def test_clean_flags_drift_for_tiny_movement_with_low_speed() -> None:
    # Tiny lat increment per 100s — far below drift_speed_kmh threshold
    lats = np.array([19.4, 19.40005, 19.4001])
    df = _make_pings(3, lat_path=lats, freq="100s")
    out = clean(df)
    assert (out["quality_flag"] == "DRIFT").any()


# ── GAP_FOLLOWS ─────────────────────────────────────────────────────


def test_clean_flags_gap_follows_when_dt_exceeds_threshold() -> None:
    df = _make_pings(2, lat_path=np.array([19.4, 19.401]))
    # 10-minute gap between row 0 and row 1
    df.loc[1, "ts"] = df.loc[0, "ts"] + pd.Timedelta("10min")
    out = clean(df)
    assert out.loc[1, "quality_flag"] == "GAP_FOLLOWS"
    # Derived columns nulled on the gap-spanning row
    assert pd.isna(out.loc[1, "speed_ms"])
    assert pd.isna(out.loc[1, "displacement_m"])


def test_clean_does_not_flag_gap_below_threshold() -> None:
    df = _make_pings(3, lat_path=np.array([19.4, 19.401, 19.402]))
    out = clean(df)
    assert (out["quality_flag"] != "GAP_FOLLOWS").all()


# ── DEVICE_FAULT ────────────────────────────────────────────────────


def test_clean_flags_device_fault_for_stuck_position_with_reported_motion() -> None:
    # 25 pings at the same position, with a steady reported speed of 10 m/s
    n = 25
    df = _make_pings(n, speed_ms=np.full(n, 10.0, dtype=np.float32))
    out = clean(df)
    assert (out["quality_flag"] == "DEVICE_FAULT").all()


def test_clean_skips_device_fault_when_no_reported_speed() -> None:
    # Stuck position but no reported speed: cannot detect sensor lying
    n = 25
    df = _make_pings(n)  # speed_ms defaults to all-NaN
    out = clean(df)
    assert (out["quality_flag"] != "DEVICE_FAULT").all()


def test_clean_skips_device_fault_below_min_pings() -> None:
    # Same pattern as the positive case, but only 5 pings
    n = 5
    df = _make_pings(n, speed_ms=np.full(n, 10.0, dtype=np.float32))
    out = clean(df)
    assert (out["quality_flag"] != "DEVICE_FAULT").all()


def test_clean_does_not_flag_legitimate_dwell_as_device_fault() -> None:
    # 25 pings at the same position with reported speed = 0 (parked)
    n = 25
    df = _make_pings(n, speed_ms=np.zeros(n, dtype=np.float32))
    out = clean(df)
    assert (out["quality_flag"] != "DEVICE_FAULT").all()


# ── is_duplicate ────────────────────────────────────────────────────


def test_clean_marks_consecutive_identical_positions_as_duplicate() -> None:
    df = _make_pings(3)  # all same position by default
    out = clean(df)
    assert out.loc[0, "is_duplicate"] == np.bool_(False)
    assert out.loc[1, "is_duplicate"] == np.bool_(True)
    assert out.loc[2, "is_duplicate"] == np.bool_(True)


# ── Stale-position merge ────────────────────────────────────────────


def _stale_pattern_pings(n: int = 30) -> pd.DataFrame:
    """Pings at 1-second intervals; position updates every 5 seconds.

    Simulates a device that pings every second but only acquires a new
    GPS fix every 5 seconds — the canonical stale-position pattern.
    """
    lat_path = np.repeat(np.linspace(19.4, 19.4 + 0.001 * (n // 5), n // 5), 5)[:n]
    return _make_pings(n, lat_path=lat_path, speed_ms=np.full(n, 10.0, dtype=np.float32))


def test_detect_stale_pattern_identifies_stale_device() -> None:
    df = _stale_pattern_pings(n=30)
    assert detect_stale_pattern(df) is True


def test_detect_stale_pattern_returns_false_for_normal_device() -> None:
    n = 30
    lat_path = np.linspace(19.4, 19.43, n)
    df = _make_pings(n, lat_path=lat_path)
    assert detect_stale_pattern(df) is False


def test_detect_stale_pattern_returns_false_below_min_pings() -> None:
    df = _stale_pattern_pings(n=10)
    assert detect_stale_pattern(df) is False


def test_merge_stale_positions_collapses_runs() -> None:
    pings = _stale_pattern_pings(n=30)
    cleaned = clean(pings)
    merged = merge_stale_positions(cleaned)
    # 30 pings, position changes every 5 → 6 runs
    assert len(merged) == 6
    tt.CleanedPingsSchema.validate(merged)


def test_merge_stale_positions_populates_merge_count_and_duration() -> None:
    pings = _stale_pattern_pings(n=30)
    cleaned = clean(pings)
    merged = merge_stale_positions(cleaned)
    assert (merged["merge_count"] == 5).all()
    assert (merged["run_duration_s"].iloc[1:] > 0).all()  # first run may be 0


def test_merge_stale_positions_handles_empty_input() -> None:
    cleaned = clean(_make_pings(0))
    merged = merge_stale_positions(cleaned)
    assert len(merged) == 0


def test_merge_stale_positions_does_not_mutate_input() -> None:
    pings = _stale_pattern_pings(n=30)
    cleaned = clean(pings)
    snapshot = cleaned.copy(deep=True)
    _ = merge_stale_positions(cleaned)
    pd.testing.assert_frame_equal(cleaned, snapshot)


# ── Params plumbing ─────────────────────────────────────────────────


def test_clean_params_are_frozen() -> None:
    p = CleanParams()
    with pytest.raises(ValidationError):
        p.max_speed_kmh = 999.0  # type: ignore[misc]


def test_clean_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CleanParams(max_speed_kmh=100.0, junk="oops")  # type: ignore[call-arg]


def test_stale_merge_params_reject_invalid_ratio() -> None:
    with pytest.raises(ValidationError):
        StaleMergeParams(detection_ratio_threshold=0.5)
