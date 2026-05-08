"""Minimal synthetic-trace builders for sanity checks.

Public, but minimal. Two builders:

* ``make_pings(n, motion, ...)`` — single-entity ``PingsSchema`` frame.
* ``make_segments(n, motion, ...)`` — single-entity ``SegmentsSchema`` frame.

Each builder produces a schema-valid frame at canonical dtypes that
downstream library functions can immediately accept. They are the
fastest path to "let me see this work end-to-end" without needing
real-world data.

Designed for sanity checks and quickstart examples, not as a
comprehensive test-data generator. The 8-scenario generator described
in ``docs/design/LIBRARY.md`` §13 is deferred to v1.1+.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

PingsMotion = Literal["stationary", "linear", "stop_then_move"]
SegmentsMotion = Literal["all_stationary", "all_moving", "alternating"]


def make_pings(
    n: int = 60,
    *,
    entity_id: str = "v1",
    start_ts: pd.Timestamp | str = "2026-01-01",
    freq: str = "1s",
    motion: PingsMotion = "linear",
    lat: float = 19.4,
    lon: float = -99.2,
    lat_step: float = 0.0001,
) -> pd.DataFrame:
    """Build a single-entity ``PingsSchema`` frame.

    Parameters
    ----------
    n
        Ping count.
    entity_id
        Single entity identifier.
    start_ts
        First ping's timestamp (UTC).
    freq
        Pandas frequency string between consecutive pings.
    motion
        One of:

        * ``"stationary"`` — all pings at ``(lat, lon)``.
        * ``"linear"`` — lat increments by ``lat_step`` per ping
          (~11 m/step at 0.0001).
        * ``"stop_then_move"`` — first half stationary, second half
          linear motion.

    lat, lon
        Starting coordinate.
    lat_step
        Per-ping lat increment for moving traces.
    """
    if n < 0:
        msg = f"n must be non-negative, got {n}"
        raise ValueError(msg)

    if isinstance(start_ts, str):
        start_ts = pd.Timestamp(start_ts, tz="UTC")
    elif start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")

    ts = pd.date_range(start_ts, periods=n, freq=freq, tz="UTC").astype(
        "datetime64[ns, UTC]"
    )
    lats = _lat_path(n, motion, lat, lat_step)
    lons = np.full(n, lon, dtype=np.float64)

    return pd.DataFrame(
        {
            "entity_id": pd.Series([entity_id] * n, dtype="string"),
            "ts": ts,
            "lat": lats,
            "lon": lons,
            "speed_ms": pd.Series([np.nan] * n, dtype=np.float32),
            "bearing_deg": pd.Series([np.nan] * n, dtype=np.float32),
        }
    )


def make_segments(
    n: int = 5,
    *,
    entity_id: str = "v1",
    start_ts: pd.Timestamp | str = "2026-01-01",
    duration_s: float = 60.0,
    motion: SegmentsMotion = "all_moving",
    lat: float = 19.4,
    lon: float = -99.2,
    lat_step: float = 0.0001,
) -> pd.DataFrame:
    """Build a single-entity ``SegmentsSchema`` frame.

    Parameters
    ----------
    n
        Segment count.
    entity_id
        Single entity identifier.
    start_ts
        First segment's start timestamp (UTC).
    duration_s
        Per-segment duration.
    motion
        One of:

        * ``"all_stationary"`` — every segment is ``STOP_DWELL`` at
          ``(lat, lon)``.
        * ``"all_moving"`` — every segment is ``MOVE`` advancing by
          ``lat_step`` per segment.
        * ``"alternating"`` — alternates ``MOVE`` and ``STOP_DWELL``,
          advancing during the move segments.

    lat, lon
        Starting coordinate.
    lat_step
        Per-segment lat increment for moving segments (~11 m at 0.0001).
    """
    if n < 0:
        msg = f"n must be non-negative, got {n}"
        raise ValueError(msg)

    if isinstance(start_ts, str):
        start_ts = pd.Timestamp(start_ts, tz="UTC")
    elif start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")

    if n == 0:
        return _empty_segments_frame()

    rows: list[dict[str, object]] = []
    cur_lat = lat
    for i in range(n):
        seg_start_ts = start_ts + pd.Timedelta(seconds=i * duration_s)
        seg_end_ts = seg_start_ts + pd.Timedelta(seconds=duration_s)

        seg_type, seg_lat_end = _segment_kinematics(
            i, motion, cur_lat, lat_step
        )

        path_length_m = _haversine_m(cur_lat, lon, seg_lat_end, lon)
        displacement_m = path_length_m  # straight-line synthetic
        straightness = 1.0 if path_length_m > 0 else 0.0

        rows.append(
            {
                "segment_id": f"{entity_id}_seg_{i + 1:05d}",
                "entity_id": entity_id,
                "segment_type": seg_type,
                "start_ts": seg_start_ts,
                "end_ts": seg_end_ts,
                "duration_s": float(duration_s),
                "start_lat": float(cur_lat),
                "start_lon": float(lon),
                "end_lat": float(seg_lat_end),
                "end_lon": float(lon),
                "start_h3": "8a2a1072b59ffff",
                "end_h3": "8a2a1072b59ffff",
                "path_length_m": float(path_length_m),
                "displacement_m": float(displacement_m),
                "straightness": float(straightness),
                "mean_speed_ms": (
                    float(path_length_m / duration_s) if path_length_m > 0 else 0.0
                ),
                "max_speed_ms": (
                    float(path_length_m / duration_s) if path_length_m > 0 else 0.0
                ),
                "bearing_variance": 0.05,
                "n_pings": int(round(duration_s)),
            }
        )

        cur_lat = seg_lat_end

    return _coerce_segments_dtypes(rows)


# ── Path / motion helpers ───────────────────────────────────────────


def _lat_path(
    n: int, motion: PingsMotion, lat: float, lat_step: float
) -> np.ndarray:
    if motion == "stationary":
        return np.full(n, lat, dtype=np.float64)
    if motion == "linear":
        return lat + np.arange(n, dtype=np.float64) * lat_step
    if motion == "stop_then_move":
        half = n // 2
        front = np.full(half, lat, dtype=np.float64)
        tail = lat + np.arange(n - half, dtype=np.float64) * lat_step
        return np.concatenate([front, tail])
    msg = f"unknown motion: {motion!r}"
    raise ValueError(msg)


def _segment_kinematics(
    i: int, motion: SegmentsMotion, cur_lat: float, lat_step: float
) -> tuple[str, float]:
    if motion == "all_stationary":
        return "STOP_DWELL", cur_lat
    if motion == "all_moving":
        return "MOVE", cur_lat + lat_step
    if motion == "alternating":
        if i % 2 == 0:
            return "MOVE", cur_lat + lat_step
        return "STOP_DWELL", cur_lat
    msg = f"unknown segments motion: {motion!r}"
    raise ValueError(msg)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    )
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _empty_segments_frame() -> pd.DataFrame:
    """Schema-shaped empty frame for ``make_segments(0)``."""
    return pd.DataFrame(
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


def _coerce_segments_dtypes(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Cast a list-of-dicts to canonical SegmentsSchema dtypes."""
    df = pd.DataFrame(rows)
    string_cols = ["segment_id", "entity_id", "segment_type", "start_h3", "end_h3"]
    f32_cols = [
        "duration_s",
        "path_length_m",
        "displacement_m",
        "straightness",
        "mean_speed_ms",
        "max_speed_ms",
        "bearing_variance",
    ]
    f64_cols = ["start_lat", "start_lon", "end_lat", "end_lon"]
    for c in string_cols:
        df[c] = df[c].astype("string")
    for c in f32_cols:
        df[c] = df[c].astype(np.float32)
    for c in f64_cols:
        df[c] = df[c].astype(np.float64)
    df["n_pings"] = df["n_pings"].astype(np.int32)
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True).astype("datetime64[ns, UTC]")
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True).astype("datetime64[ns, UTC]")
    return df


__all__ = ["PingsMotion", "SegmentsMotion", "make_pings", "make_segments"]
