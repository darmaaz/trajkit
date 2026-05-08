"""Stale-position merge — collapse duplicate-position runs.

Some GPS devices ping more often than they update position, producing
runs of identical (lat, lon) interspersed with large jumps when the
position finally refreshes. The cleaning layer would mis-flag those
jumps as ``SPEED_OUTLIER``; the segmentation layer would mis-classify
the affected vehicles as stationary on a highway.

This module detects the pattern and collapses each same-position run
into a single representative row. The merged frame has fewer rows but
is more accurate per-row. ``merge_count`` and ``run_duration_s`` are
populated so downstream segmentation can use time-based thresholds
rather than ping counts.

Single-entity input. Optional — call only when ``detect_stale_pattern``
returns True for the entity, or when the user explicitly opts in
because they know their provider's behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trajkit.clean._clean import (
    _derive_kinematics,
    _detect_duplicates,
    _flag_drift,
    _flag_gaps,
    _flag_speed_outliers,
    _null_gap_edges,
    _null_outlier_edges,
)
from trajkit.clean._params import CleanParams, StaleMergeParams


def detect_stale_pattern(
    pings_df: pd.DataFrame, params: StaleMergeParams | None = None
) -> bool:
    """Return True if the entity exhibits the stale-position pattern.

    Compares ``median(time-between-position-updates)`` to
    ``median(time-between-pings)``. A ratio above
    ``params.detection_ratio_threshold`` means GPS positions update less
    often than the device pings.
    """
    p = params if params is not None else StaleMergeParams()

    if len(pings_df) < p.min_pings_for_detection:
        return False

    # Why exact float equality and not np.isclose: stale-position devices
    # repeat the same bytes for lat/lon. A tolerance check would falsely
    # match GPS drift around real stops, eroding "real stop" detection.
    same_pos = (pings_df["lat"] == pings_df["lat"].shift(1)) & (
        pings_df["lon"] == pings_df["lon"].shift(1)
    )
    pos_changed = ~same_pos

    dt_pings = pings_df["ts"].diff().dt.total_seconds().dropna()
    change_times = pings_df.loc[pos_changed, "ts"]
    dt_changes = change_times.diff().dt.total_seconds().dropna()

    if len(dt_pings) == 0 or len(dt_changes) == 0:
        return False

    ping_med = float(dt_pings.median())
    pos_med = float(dt_changes.median())

    if ping_med <= 0 or pd.isna(pos_med):
        return False

    return (pos_med / ping_med) > p.detection_ratio_threshold


def merge_stale_positions(
    cleaned_pings_df: pd.DataFrame,
    params: StaleMergeParams | None = None,
    clean_params: CleanParams | None = None,
) -> pd.DataFrame:
    """Collapse same-position runs into single rows; recompute kinematics + flags.

    Parameters
    ----------
    cleaned_pings_df
        Output of ``trajkit.clean.clean`` for one entity.
    params
        Stale-merge parameters (currently unused inside merge — detection is
        a separate call — but kept for API symmetry and future extension).
    clean_params
        Quality-flag thresholds for the post-merge re-flagging pass. If
        ``None``, ``CleanParams()`` defaults are used.

    Returns
    -------
    pd.DataFrame
        New frame, validated by ``CleanedPingsSchema``. Row count ≤ input.
        ``merge_count`` and ``run_duration_s`` are populated for every row.
    """
    _ = params if params is not None else StaleMergeParams()
    cp = clean_params if clean_params is not None else CleanParams()

    if len(cleaned_pings_df) == 0:
        return cleaned_pings_df.copy()

    df = cleaned_pings_df.copy().reset_index(drop=True)

    # Identify same-position runs. Exact-equality rationale: see
    # detect_stale_pattern.
    same_pos = (df["lat"] == df["lat"].shift(1)) & (df["lon"] == df["lon"].shift(1))
    run_id = (~same_pos).cumsum()

    runs = df.groupby(run_id, sort=False)
    first_idx = runs.head(1).index
    merged = df.loc[first_idx].reset_index(drop=True)

    merge_count = runs.size().to_numpy().astype(np.int32)
    run_first_ts = runs["ts"].first()
    run_last_ts = runs["ts"].last()
    run_duration_s = (
        (run_last_ts - run_first_ts).dt.total_seconds().to_numpy(dtype=np.float32)
    )

    # Recompute kinematics on the merged timeline.
    merged = _derive_kinematics(merged)

    # Reset and re-apply quality flags. We skip DEVICE_FAULT here because
    # the merged frame is by construction stale-position; the flag is no
    # longer informative once we've explicitly handled the pattern.
    merged["quality_flag"] = pd.Series(
        ["VALID"] * len(merged), index=merged.index, dtype="string"
    )
    _flag_speed_outliers(merged, cp)
    _null_outlier_edges(merged)
    _flag_drift(merged, cp)
    _flag_gaps(merged, cp)
    _null_gap_edges(merged)

    merged["is_duplicate"] = _detect_duplicates(merged)
    merged["merge_count"] = pd.array(merge_count, dtype="Int32")
    merged["run_duration_s"] = run_duration_s

    return merged
