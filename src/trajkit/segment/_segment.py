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
    """True where a direction change should split a MOVE.

    Detector: mean resultant length ``R`` of bearings inside a *distance-
    based* sliding window centred on each ping. Low R = bearings spread
    around the unit circle = direction is changing. High R = bearings
    cluster = direction is stable. Multi-scale (short + long windows)
    catches both street-corner turns and arterial / sustained turns.

    Why circular statistics. Bearing is angular: averaging linear deltas
    confounds "single sharp turn" (one big delta diluted by many zeros)
    with "noise around a stable direction" (deltas oscillating around
    zero). R correctly distinguishes them — the unit-circle vector mean
    has length 1 for tight clusters and < 1 for spread.

    Why distance windows. Time-based windows are ping-rate-dependent: a
    90° turn at 1-Hz pings dilutes to ≈ 1° per ping in a 2-min window
    while the same turn at 1-min pings shows as ≈ 90°. A 200 m window
    contains the same physical-behaviour magnitude regardless of ping
    rate.

    Why multi-scale. A 200 m window smears a sharp street-corner over a
    long stretch — most of the window still points in the pre-turn
    direction, so R stays high and the corner is missed. A 75 m short
    window concentrates the turn in a larger fraction of the window and
    pulls R below the entry threshold. Long windows catch arterial /
    sustained turns; short windows catch street corners. Boundary fires
    when R drops in *either* window.

    Why hysteresis. Two-threshold Schmitt trigger over distance:

    * Enter "direction-changing" when R drops below ``bearing_r_enter``
      (in either window) and stays there for ``bearing_sustain_m`` metres.
    * Exit when R rises above ``bearing_r_exit`` (in both windows) and
      stays there for ``bearing_sustain_m`` metres.

    Boundary fires on entry (the rising edge of the in-direction-change
    state). Restricted to moving pings; stopped-period bearings are
    masked out so they don't pollute R.

    Sparse-window guard: a per-window minimum count blocks R from being
    computed on starved windows. The configured
    ``bearing_window_min_pings`` is the ceiling; it adapts down to a
    floor of 2 when the trace's observed median per-ping displacement
    would otherwise make the configured value impossible to satisfy
    inside ``window_m``. See ``_effective_min_pings`` for the rule.
    """
    n = len(df)
    boundary = np.zeros(n, dtype=bool)
    moving = ~is_stop
    if n < 2 or not moving.any():
        return boundary

    bearing_arr = df["bearing_deg"].to_numpy(dtype=np.float64)
    valid = moving & ~np.isnan(bearing_arr)

    # Cumulative distance over MOTION ONLY (stops collapse to zero so a
    # pause in a journey doesn't burn the window with no-progress pings).
    disp = df["displacement_m"].fillna(0.0).to_numpy(dtype=np.float64)
    disp_motion = np.where(moving, disp, 0.0)
    cum_dist = np.cumsum(disp_motion)

    # The configured ``bearing_window_min_pings`` is the CEILING; the
    # per-window minimum adapts to the actual ping density inside each
    # window. See ``_circular_r_over_distance`` for the rule.
    r_short = _circular_r_over_distance(
        cum_dist, bearing_arr, valid,
        p.bearing_window_short_m, p.bearing_window_min_pings,
    )
    r_long = _circular_r_over_distance(
        cum_dist, bearing_arr, valid,
        p.bearing_window_long_m, p.bearing_window_min_pings,
    )

    # NaN R is ambiguous evidence — neither low nor high. For entry,
    # treat NaN as "high" (won't trigger entry); for exit, treat as "low"
    # (won't trigger exit). Either way, sparse windows are conservative.
    short_high = np.where(np.isnan(r_short), 1.0, r_short)
    long_high = np.where(np.isnan(r_long), 1.0, r_long)
    short_low = np.where(np.isnan(r_short), 0.0, r_short)
    long_low = np.where(np.isnan(r_long), 0.0, r_long)

    enter_signal = (short_high < p.bearing_r_enter) | (long_high < p.bearing_r_enter)
    exit_signal = (short_low > p.bearing_r_exit) & (long_low > p.bearing_r_exit)

    in_change = _distance_hysteresis(
        cum_dist, enter_signal, exit_signal, p.bearing_sustain_m
    )

    # Boundary fires on rising edge of in_change state, on a moving ping.
    in_change_active = in_change & moving
    boundary[1:] = in_change_active[1:] & ~in_change_active[:-1]
    return boundary


def _circular_r_over_distance(
    cum_dist: np.ndarray,
    bearing: np.ndarray,
    valid: np.ndarray,
    window_m: float,
    min_count: int,
) -> np.ndarray:
    """Mean resultant length R over a symmetric-distance window per ping.

    Vectorised via cumulative sums of cos/sin/valid-count and binary
    searches on ``cum_dist`` — O(n log n) over the trace.

    Uses ``cos`` / ``sin`` of bearings directly, which side-steps the
    angular wraparound problem entirely (no signed-delta needed; the
    unit-circle vector mean *is* circularly correct by construction).

    The ``min_count`` parameter is the **ceiling** of the per-window
    minimum-valid-bearings guard. The effective minimum per window
    adapts to the actual ping count in the window: ``max(2, floor(
    n_in_window / 2))``, clamped at ``min_count``. This keeps the
    detector from going blind on sparse-cadence data (e.g. 5 s
    vehicular pings) while preserving the high-density confidence
    bar on dense data. A hard floor of 2 prevents R from being
    computed on a single sample.
    """
    n = len(cum_dist)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    bearing_rad = np.deg2rad(bearing)
    cos_b = np.where(valid, np.cos(bearing_rad), 0.0)
    sin_b = np.where(valid, np.sin(bearing_rad), 0.0)

    cum_cos = np.concatenate([[0.0], np.cumsum(cos_b)])
    cum_sin = np.concatenate([[0.0], np.cumsum(sin_b)])
    cum_valid = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])

    half = window_m / 2.0
    lo = np.searchsorted(cum_dist, cum_dist - half, side="left")
    hi = np.searchsorted(cum_dist, cum_dist + half, side="right")

    n_in_window = hi - lo
    n_valid = cum_valid[hi] - cum_valid[lo]
    sum_cos = cum_cos[hi] - cum_cos[lo]
    sum_sin = cum_sin[hi] - cum_sin[lo]

    safe_n = np.where(n_valid > 0, n_valid, 1)
    mean_cos = sum_cos / safe_n
    mean_sin = sum_sin / safe_n
    r_raw = np.sqrt(mean_cos**2 + mean_sin**2)

    adaptive_min = np.clip(n_in_window // 2, 2, min_count)
    out: np.ndarray = np.where(n_valid >= adaptive_min, r_raw, np.nan)
    return out


def _distance_hysteresis(
    cum_dist: np.ndarray,
    enter_signal: np.ndarray,
    exit_signal: np.ndarray,
    sustain_m: float,
) -> np.ndarray:
    """Schmitt-trigger state machine over cumulative distance.

    Enters the "low-R" state when ``enter_signal`` has been True
    continuously for ``sustain_m`` metres; exits when ``exit_signal`` has
    been True continuously for ``sustain_m`` metres. Pending-distance
    counter resets whenever the relevant signal goes False, preventing
    intermittent low-R blips from accumulating into a state flip.
    """
    n = len(cum_dist)
    in_low = np.zeros(n, dtype=bool)
    if n == 0:
        return in_low
    state = False
    pending = 0.0
    for i in range(1, n):
        dx = max(float(cum_dist[i] - cum_dist[i - 1]), 0.0)
        if state:
            if exit_signal[i]:
                pending += dx
                if pending >= sustain_m:
                    state = False
                    pending = 0.0
            else:
                pending = 0.0
        else:
            if enter_signal[i]:
                pending += dx
                if pending >= sustain_m:
                    state = True
                    pending = 0.0
            else:
                pending = 0.0
        in_low[i] = state
    return in_low


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
