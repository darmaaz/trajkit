"""Collapse a segmented per-ping frame into one row per segment.

Output conforms to ``SegmentsSchema``.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from pyproj import Geod

from trajkit.segment._params import SegmentParams

_GEOD = Geod(ellps="WGS84")


def aggregate_segments(
    segmented_pings_df: pd.DataFrame,
    params: SegmentParams | None = None,
) -> pd.DataFrame:
    """Collapse a per-ping segmented frame into one row per ``segment_id``.

    Parameters
    ----------
    segmented_pings_df
        Output of ``trajkit.segment.segment``.
    params
        Frozen ``SegmentParams``; only ``h3_resolution`` is used here.

    Returns
    -------
    pd.DataFrame
        New frame conforming to ``SegmentsSchema``.
    """
    p = params if params is not None else SegmentParams()

    if len(segmented_pings_df) == 0:
        return _empty_segments_frame()

    df = segmented_pings_df
    rows: list[dict[str, object]] = []

    for seg_id, g in df.groupby("segment_id", sort=False):
        rows.append(_aggregate_one_segment(seg_id, g, p))

    result = pd.DataFrame(rows)
    return _enforce_dtypes(result)


# ── Per-segment aggregation ─────────────────────────────────────────


def _aggregate_one_segment(
    seg_id: str, g: pd.DataFrame, p: SegmentParams
) -> dict[str, object]:
    """Build one ``SegmentsSchema`` row from a per-ping group."""
    entity_id = str(g["entity_id"].iloc[0])
    segment_type = str(g["segment_type"].iloc[0])

    start_ts = g["ts"].iloc[0]
    end_ts = g["ts"].iloc[-1]
    duration_s = float((end_ts - start_ts).total_seconds())
    if "run_duration_s" in g.columns:
        trailing = g["run_duration_s"].iloc[-1]
        if pd.notna(trailing):
            duration_s += float(trailing)
    duration_s = max(duration_s, 0.0)  # clamp microscopic negatives from float math

    start_lat = float(g["lat"].iloc[0])
    start_lon = float(g["lon"].iloc[0])
    end_lat = float(g["lat"].iloc[-1])
    end_lon = float(g["lon"].iloc[-1])

    start_h3 = h3.latlng_to_cell(start_lat, start_lon, p.h3_resolution)
    end_h3 = h3.latlng_to_cell(end_lat, end_lon, p.h3_resolution)

    # Path length: sum of valid per-ping displacements. Outlier-edge nulls
    # were already applied by clean(), so dropna() yields trustworthy edges.
    disp = g["displacement_m"].dropna()
    path_length_m = float(disp.sum())

    # Crow-fly displacement: great-circle from segment start to end
    _, _, crow_fly = _GEOD.inv(start_lon, start_lat, end_lon, end_lat)
    displacement_m = float(abs(crow_fly))

    # Straightness ∈ [0, 1]. Undefined (and unimportant) for zero-path stops.
    if path_length_m > 0.0:
        raw_straightness = displacement_m / path_length_m
        straightness = float(min(max(raw_straightness, 0.0), 1.0))
    else:
        straightness = 0.0

    speed = g["speed_ms"].dropna()
    if len(speed) > 0:
        mean_speed_ms: float | None = float(speed.mean())
        max_speed_ms: float | None = float(speed.max())
    else:
        mean_speed_ms = None
        max_speed_ms = None

    bearing_variance = _circular_variance(g["bearing_deg"].dropna().to_numpy())

    if "merge_count" in g.columns and g["merge_count"].notna().any():
        n_pings = int(g["merge_count"].fillna(1).sum())
    else:
        n_pings = int(len(g))

    return {
        "segment_id": str(seg_id),
        "entity_id": entity_id,
        "segment_type": segment_type,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": duration_s,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "start_h3": start_h3,
        "end_h3": end_h3,
        "path_length_m": path_length_m,
        "displacement_m": displacement_m,
        "straightness": straightness,
        "mean_speed_ms": mean_speed_ms,
        "max_speed_ms": max_speed_ms,
        "bearing_variance": bearing_variance,
        "n_pings": n_pings,
    }


def _circular_variance(bearings_deg: np.ndarray) -> float | None:
    """Return circular variance ∈ [0, 1] of bearings, or None if undefined.

    1 − |R̂| where R̂ is the mean resultant vector of the bearings on
    the unit circle. 0 means all bearings identical; 1 means uniformly
    distributed.
    """
    if len(bearings_deg) == 0:
        return None
    rad = np.deg2rad(bearings_deg)
    mean_cos = float(np.cos(rad).mean())
    mean_sin = float(np.sin(rad).mean())
    r_bar = float(np.sqrt(mean_cos**2 + mean_sin**2))
    return float(min(max(1.0 - r_bar, 0.0), 1.0))


# ── Schema dtype enforcement ────────────────────────────────────────


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a freshly built segments frame to canonical SegmentsSchema dtypes.

    pd.DataFrame from a list-of-dicts uses object/float64 for everything;
    SegmentsSchema expects float32 for floating columns and int32/string
    for the rest.
    """
    df = df.copy()
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


def _empty_segments_frame() -> pd.DataFrame:
    """Empty SegmentsSchema-shaped frame at canonical dtypes."""
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
