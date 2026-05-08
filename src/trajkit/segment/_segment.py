"""Per-entity segmentation: hysteresis state machine, boundary detection, classification.

Implements ``segment`` and its private helpers. Single-entity input
sorted by ``ts`` from ``trajkit.clean``. Output is a frame conforming to
``SegmentedPingsSchema``: input columns plus ``segment_id`` and
``segment_type``.

See ``docs/design/segment.md`` for the full specification, including the
rationale for the four-state taxonomy and the closure rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trajkit.segment._params import SegmentParams

_KMH_TO_MS = 1000.0 / 3600.0
_SEG_ID_DIGITS = 5  # zero-padded to 5 digits → up to 99,999 segments per entity


def segment(
    cleaned_pings_df: pd.DataFrame, params: SegmentParams | None = None
) -> pd.DataFrame:
    """Add ``segment_id`` and ``segment_type`` to a cleaned per-ping frame.

    Parameters
    ----------
    cleaned_pings_df
        Output of ``trajkit.clean.clean`` for one entity. Sorted by ``ts``.
    params
        Frozen ``SegmentParams``; defaults are scale-class agnostic.

    Returns
    -------
    pd.DataFrame
        New frame conforming to ``SegmentedPingsSchema``.
    """
    p = params if params is not None else SegmentParams()
    n = len(cleaned_pings_df)
    if n == 0:
        return _empty_segmented_frame()

    df = cleaned_pings_df.copy().reset_index(drop=True)
    entity_id = str(df["entity_id"].iloc[0])

    stop_speed_ms = p.stop_speed_kmh * _KMH_TO_MS
    resume_speed_ms = p.resume_speed_kmh * _KMH_TO_MS

    # ── 1. Motion state with hysteresis ─────────────────────────────
    speed = _fill_outlier_nans(df["speed_ms"], df["quality_flag"])
    is_stop = _hysteresis_state(speed, stop_speed_ms, resume_speed_ms)

    # ── 2. Filter short stop runs (time-based) ──────────────────────
    is_stop = _filter_short_stops(df, is_stop, p.stop_min_duration_s)

    # ── 3. Boundary detection ───────────────────────────────────────
    state_change = _state_change_boundaries(is_stop)
    # Explicit dtype: comparing StringDtype yields ``bool[pyarrow]`` which
    # doesn't OR cleanly with numpy arrays; force numpy bool here.
    gap_boundary = (df["quality_flag"] == "GAP_FOLLOWS").to_numpy(dtype=bool)
    bearing_boundary = _bearing_boundaries(df, is_stop, p)

    new_segment = state_change | gap_boundary | bearing_boundary
    new_segment[0] = True
    seg_num = np.cumsum(new_segment).astype(np.int64)

    # ── 4. Build segment_id ─────────────────────────────────────────
    df["segment_id"] = pd.Series(
        [f"{entity_id}_seg_{n:0{_SEG_ID_DIGITS}d}" for n in seg_num],
        dtype="string",
        index=df.index,
    )

    # ── 5. Classify segments ────────────────────────────────────────
    df["segment_type"] = _classify_segments(df, seg_num, is_stop, p)

    return df


# ── State-machine helpers ───────────────────────────────────────────


def _fill_outlier_nans(speed: pd.Series, quality_flag: pd.Series) -> np.ndarray:
    """Fill NaN speeds caused by outlier-edge nulling, but not across GAP_FOLLOWS.

    The cleaning layer nulls speed on outlier-tainted pings. Those nulls
    should inherit motion state from neighbours (a 1–2 ping outlier in
    the middle of a movement run shouldn't break the segment). But a
    GAP_FOLLOWS marks a discontinuity where state must NOT propagate,
    so we forward/backward-fill within gap-bounded runs only.

    Note: ``quality_flag`` is a pandas StringDtype column. Equality
    against a string literal returns ``bool[pyarrow]``, which doesn't
    support ``cumsum``; we materialise the mask as numpy bool first.
    """
    s = speed.copy()
    gap_arr = (quality_flag == "GAP_FOLLOWS").to_numpy(dtype=bool)
    fill_group = pd.Series(np.cumsum(gap_arr), index=speed.index)
    filled = (
        s.groupby(fill_group)
        .ffill(limit=2)
        .groupby(fill_group)
        .bfill(limit=2)
    )
    return np.asarray(filled.to_numpy(dtype=np.float64), dtype=np.float64)


def _hysteresis_state(
    speed: np.ndarray, stop_threshold_ms: float, resume_threshold_ms: float
) -> np.ndarray:
    """Classify per-ping motion state via a hysteresis state machine.

    Below ``stop_threshold_ms`` the state moves to stopped; above
    ``resume_threshold_ms`` it moves to moving. Between them (the dead
    zone) and on NaN/inf, the current state persists.

    The carry-forward state requires a Python loop. For per-entity input
    of ≤ 1M pings this is acceptable; at fleet scale, the multiprocess
    runner amortises across cores.
    """
    n = len(speed)
    is_stop = np.zeros(n, dtype=bool)

    state_stopped = (
        bool(speed[0] < stop_threshold_ms) if np.isfinite(speed[0]) else True
    )
    is_stop[0] = state_stopped

    for i in range(1, n):
        v = speed[i]
        if np.isfinite(v):
            if state_stopped and v >= resume_threshold_ms:
                state_stopped = False
            elif not state_stopped and v < stop_threshold_ms:
                state_stopped = True
        is_stop[i] = state_stopped

    return is_stop


def _filter_short_stops(
    df: pd.DataFrame, is_stop: np.ndarray, min_duration_s: float
) -> np.ndarray:
    """Reclassify too-short stop runs as moving (noise filter)."""
    if min_duration_s <= 0 or not is_stop.any():
        return is_stop

    is_stop_s = pd.Series(is_stop, index=df.index)
    run_boundary = is_stop_s != is_stop_s.shift(1, fill_value=~is_stop_s.iloc[0])
    run_id = run_boundary.cumsum()

    run_start = df["ts"].groupby(run_id).transform("min")
    run_end = df["ts"].groupby(run_id).transform("max")
    run_duration = (run_end - run_start).dt.total_seconds()
    if "run_duration_s" in df.columns:
        # Merged data: add the trailing time the last row in the run represents.
        last_run = df["run_duration_s"].groupby(run_id).transform("last")
        run_duration = run_duration + last_run.fillna(0.0)

    keep_stop = is_stop_s & (run_duration >= min_duration_s)
    return np.asarray(keep_stop.to_numpy(dtype=bool), dtype=bool)


# ── Boundary helpers ────────────────────────────────────────────────


def _state_change_boundaries(is_stop: np.ndarray) -> np.ndarray:
    """True where the hysteresis state changed from the previous ping."""
    changes = np.zeros_like(is_stop, dtype=bool)
    if len(is_stop) > 1:
        changes[1:] = is_stop[1:] != is_stop[:-1]
    return changes


def _bearing_boundaries(
    df: pd.DataFrame, is_stop: np.ndarray, p: SegmentParams
) -> np.ndarray:
    """True where a sustained bearing change should split a MOVE.

    The detector is time-based on both axes so it stays consistent across
    ping rates and across merged-row data:

    * Rolling mean of consecutive bearing deltas over a
      ``bearing_window_min``-minute window.
    * Sustainment: the rolling-mean must exceed the threshold continuously
      for at least ``bearing_sustain_s`` seconds (``min_periods=2`` so a
      single ping can't fire a boundary).

    Bearings flagged as NaN (typically very short displacements) are
    skipped by the rolling mean naturally; they don't produce boundaries.
    """
    n = len(df)
    boundary = np.zeros(n, dtype=bool)
    moving = ~is_stop
    if not moving.any():
        return boundary

    bearing = df["bearing_deg"]
    # Wraparound-safe consecutive delta in [-180, 180], absolute value
    delta = ((bearing - bearing.shift(1) + 180.0) % 360.0 - 180.0).abs()

    delta_t = delta.copy()
    delta_t.index = pd.DatetimeIndex(df["ts"])
    rolling_delta = delta_t.rolling(
        f"{p.bearing_window_min}min", min_periods=2
    ).mean()

    exceeds = pd.Series(moving, index=df.index) & pd.Series(
        rolling_delta.to_numpy() > p.bearing_change_deg, index=df.index
    )

    exceeds_t = exceeds.copy()
    exceeds_t.index = pd.DatetimeIndex(df["ts"])
    sustained = (
        exceeds_t.rolling(f"{int(p.bearing_sustain_s)}s", min_periods=2)
        .min()
        .fillna(0)
        .astype(bool)
    )
    sustained_arr = sustained.to_numpy()
    # Fire on the first ping of each sustained run
    if n > 1:
        boundary[1:] = sustained_arr[1:] & ~sustained_arr[:-1]
    return boundary


# ── Classification ──────────────────────────────────────────────────


def _classify_segments(
    df: pd.DataFrame,
    seg_num: np.ndarray,
    is_stop: np.ndarray,
    p: SegmentParams,
) -> pd.Series:
    """Assign one of {MOVE, MOVE_BRIEF, STOP_BRIEF, STOP_DWELL} per ping.

    All pings within a segment receive the same label.
    """
    seg_series = pd.Series(seg_num, index=df.index, name="_seg")

    is_stop_s = pd.Series(is_stop, index=df.index)
    seg_is_stop = is_stop_s.groupby(seg_series).transform("mean") > 0.5

    # Duration: ts span + trailing run_duration_s (for merged data)
    seg_start = df["ts"].groupby(seg_series).transform("min")
    seg_end = df["ts"].groupby(seg_series).transform("max")
    seg_duration_s = (seg_end - seg_start).dt.total_seconds()
    if "run_duration_s" in df.columns:
        last_run = df["run_duration_s"].groupby(seg_series).transform("last")
        seg_duration_s = seg_duration_s + last_run.fillna(0.0)

    seg_duration_min = seg_duration_s / 60.0

    types = pd.Series(["MOVE"] * len(df), index=df.index, dtype="string")
    types[seg_is_stop & (seg_duration_min >= p.dwell_threshold_min)] = "STOP_DWELL"
    types[seg_is_stop & (seg_duration_min < p.dwell_threshold_min)] = "STOP_BRIEF"

    # False-stop override: a segment classified STOP* but with significant
    # crow-fly displacement is actually slow-moving traffic, not a real stop.
    seg_first_lat = df["lat"].groupby(seg_series).transform("first")
    seg_first_lon = df["lon"].groupby(seg_series).transform("first")
    seg_last_lat = df["lat"].groupby(seg_series).transform("last")
    seg_last_lon = df["lon"].groupby(seg_series).transform("last")
    crow_fly = _haversine_array(
        seg_first_lat.to_numpy(),
        seg_first_lon.to_numpy(),
        seg_last_lat.to_numpy(),
        seg_last_lon.to_numpy(),
    )
    is_stop_label = types.isin(["STOP_DWELL", "STOP_BRIEF"])
    types[is_stop_label & (crow_fly > p.max_stop_displacement_m)] = "MOVE"

    # MOVE → MOVE_BRIEF if both ping count and duration are below thresholds
    if "merge_count" in df.columns:
        # Sum the per-row merge_count to recover raw ping counts per segment
        raw_pings = df["merge_count"].fillna(1).groupby(seg_series).transform("sum")
    else:
        raw_pings = seg_series.groupby(seg_series).transform("size")

    is_move = types == "MOVE"
    is_brief = (
        is_move
        & (raw_pings < p.move_brief_min_pings)
        & (seg_duration_s < p.move_brief_max_duration_s)
    )
    types[is_brief] = "MOVE_BRIEF"

    return types


def _haversine_array(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance in metres (mean Earth radius)."""
    r = 6_371_000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    result: np.ndarray = 2.0 * r * np.arcsin(np.sqrt(a))
    return result


# ── Empty-frame factory ─────────────────────────────────────────────


def _empty_segmented_frame() -> pd.DataFrame:
    """Empty frame with all SegmentedPingsSchema columns at canonical dtypes."""
    return pd.DataFrame(
        {
            "entity_id": pd.Series([], dtype="string"),
            "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "lat": pd.Series([], dtype=np.float64),
            "lon": pd.Series([], dtype=np.float64),
            "speed_ms": pd.Series([], dtype=np.float32),
            "bearing_deg": pd.Series([], dtype=np.float32),
            "dt_seconds": pd.Series([], dtype=np.float32),
            "displacement_m": pd.Series([], dtype=np.float32),
            "is_duplicate": pd.Series([], dtype=bool),
            "quality_flag": pd.Series([], dtype="string"),
            "merge_count": pd.array([], dtype="Int32"),
            "run_duration_s": pd.Series([], dtype=np.float32),
            "segment_id": pd.Series([], dtype="string"),
            "segment_type": pd.Series([], dtype="string"),
        }
    )
