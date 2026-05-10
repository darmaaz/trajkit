"""Per-entity cleaning: dedup, kinematics, quality flags.

Implements the ``clean`` function and its private helpers.
Single-entity input sorted by ``ts``; trusts the contract.
See ``docs/design/clean.md`` for the full spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Geod

from trajkit.clean._params import CleanParams

_GEOD = Geod(ellps="WGS84")
_KMH_TO_MS = 1000.0 / 3600.0
_F32_NAN = np.float32("nan")


def clean(pings_df: pd.DataFrame, params: CleanParams | None = None) -> pd.DataFrame:
    """Clean raw pings: dedup, derive kinematics, flag outliers and faults.

    Parameters
    ----------
    pings_df
        A frame validated against ``PingsSchema``: single ``entity_id``,
        sorted by ``ts``, WGS84 lat/lon. ``speed_ms`` and ``bearing_deg``
        may be present as user-reported values; they are used for
        ``DEVICE_FAULT`` detection (reported speed) and otherwise
        recomputed from positions.

    params
        Frozen ``CleanParams`` instance. Defaults are scale-class agnostic;
        users typically override via ``CleanParams.from_preset(...)`` (added
        in v0.1.0+).

    Returns
    -------
    pd.DataFrame
        A new frame validated by ``CleanedPingsSchema``: original PingsSchema
        columns plus ``dt_seconds``, ``displacement_m``, ``is_duplicate``,
        ``quality_flag``, ``merge_count``, ``run_duration_s``. ``merge_count``
        and ``run_duration_s`` are null here; they are populated only by
        ``merge_stale_positions``.
    """
    p = params if params is not None else CleanParams()

    if len(pings_df) == 0:
        return _empty_cleaned_frame()

    df = pings_df.drop_duplicates().reset_index(drop=True)

    # Capture user-reported speed before kinematics overwrites it. Used for
    # DEVICE_FAULT detection only; the canonical output speed_ms is derived.
    reported_speed_ms = (
        df["speed_ms"].copy() if "speed_ms" in df.columns else None
    )

    df = _derive_kinematics(df)

    df["quality_flag"] = pd.Series(["VALID"] * len(df), index=df.index, dtype="string")

    # Apply flags in precedence order: a higher flag claims the row first
    # and lower stages skip it via the ``quality_flag == "VALID"`` guard.
    #
    # Order: DEVICE_FAULT > SPEED_OUTLIER > GAP_FOLLOWS > DRIFT > VALID.
    #
    # GAP_FOLLOWS outranks DRIFT because gap-spanning pings have unreliable
    # ``displacement_m`` and ``speed_ms`` — those are computed from a single
    # observation across an unobserved interval, so any "drift-shaped"
    # measurement during a gap is meaningless. Without this ordering, a
    # multi-hour gap with small inter-ping displacement gets stamped DRIFT,
    # the segmenter sees no gap boundary, and segments grow across the
    # missing interval.
    _flag_device_faults(df, reported_speed_ms, p)
    _flag_speed_outliers(df, p)
    _null_outlier_edges(df)
    _flag_gaps(df, p)
    _null_gap_edges(df)
    _flag_drift(df, p)

    df["is_duplicate"] = _detect_duplicates(df)

    # merge_count and run_duration_s are populated by merge_stale_positions
    # only. Default to null so the schema's nullability holds.
    df["merge_count"] = pd.array([pd.NA] * len(df), dtype="Int32")
    df["run_duration_s"] = np.full(len(df), np.nan, dtype=np.float32)

    return df


# ── Kinematics ──────────────────────────────────────────────────────


def _derive_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``dt_seconds``, ``displacement_m``, ``speed_ms``, ``bearing_deg``.

    Vectorised single ``Geod.inv`` call yields both distance and forward
    azimuth in one ellipsoidal pass. First row has NaN for all derived
    columns (no predecessor).
    """
    n = len(df)
    df = df.copy()

    dt_seconds = np.full(n, np.nan, dtype=np.float32)
    displacement_m = np.full(n, np.nan, dtype=np.float32)
    bearing_deg = np.full(n, np.nan, dtype=np.float32)

    if n >= 2:
        prev_lat = df["lat"].shift(1).to_numpy()
        prev_lon = df["lon"].shift(1).to_numpy()
        cur_lat = df["lat"].to_numpy()
        cur_lon = df["lon"].to_numpy()
        prev_ts = df["ts"].shift(1)

        valid = ~np.isnan(prev_lat)
        if valid.any():
            az, _, dist = _GEOD.inv(
                prev_lon[valid],
                prev_lat[valid],
                cur_lon[valid],
                cur_lat[valid],
            )
            displacement_m[valid] = np.abs(dist).astype(np.float32)
            bearing_deg[valid] = (np.asarray(az, dtype=np.float32) % 360.0)

        dt_full = (df["ts"] - prev_ts).dt.total_seconds().to_numpy(dtype=np.float64)
        dt_seconds[:] = dt_full.astype(np.float32)

    df["dt_seconds"] = dt_seconds
    df["displacement_m"] = displacement_m

    speed_ms = np.full(n, np.nan, dtype=np.float32)
    speed_valid = (dt_seconds > 0) & ~np.isnan(displacement_m)
    speed_ms[speed_valid] = displacement_m[speed_valid] / dt_seconds[speed_valid]
    df["speed_ms"] = speed_ms

    df["bearing_deg"] = bearing_deg
    return df


# ── Quality flag stages ─────────────────────────────────────────────


def _flag_device_faults(
    df: pd.DataFrame, reported_speed_ms: pd.Series | None, p: CleanParams
) -> None:
    """Mark all rows ``DEVICE_FAULT`` if the entity is stuck-position w/ reported motion.

    A device that pings repeatedly from the same coordinate while reporting
    non-zero speed has a broken sensor: trust nothing it reports.
    """
    n = len(df)
    if n < p.device_fault_min_pings:
        return

    unique_pos = df[["lat", "lon"]].drop_duplicates()
    if len(unique_pos) > p.device_fault_max_unique_positions:
        return

    if reported_speed_ms is None:
        return
    valid_reported = reported_speed_ms.dropna()
    if len(valid_reported) == 0:
        return

    mean_reported_kmh = float(valid_reported.mean()) / _KMH_TO_MS
    std_reported_kmh = float(valid_reported.std()) if len(valid_reported) > 1 else 0.0
    std_reported_kmh /= _KMH_TO_MS if std_reported_kmh > 0 else 1.0

    if mean_reported_kmh < 1.0:
        return
    if std_reported_kmh > p.device_fault_max_speed_std_kmh:
        return

    df["quality_flag"] = pd.Series(
        ["DEVICE_FAULT"] * len(df), index=df.index, dtype="string"
    )


def _flag_speed_outliers(df: pd.DataFrame, p: CleanParams) -> None:
    threshold_ms = p.max_speed_kmh * _KMH_TO_MS
    mask = (df["speed_ms"] > threshold_ms) & (df["quality_flag"] == "VALID")
    df.loc[mask, "quality_flag"] = "SPEED_OUTLIER"


def _null_outlier_edges(df: pd.DataFrame) -> None:
    """Both edges touching a SPEED_OUTLIER carry garbage; null both.

    Outlier ping B is the A→B edge; the next ping C is the B→C edge whose
    derivation came from B's untrusted position. Both lose their derived
    columns.
    """
    outlier = (df["quality_flag"] == "SPEED_OUTLIER").to_numpy()
    if not outlier.any():
        return
    after = np.zeros_like(outlier)
    after[1:] = outlier[:-1]
    tainted = outlier | after
    df.loc[tainted, ["speed_ms", "bearing_deg", "displacement_m"]] = _F32_NAN


def _flag_drift(df: pd.DataFrame, p: CleanParams) -> None:
    drift_speed_ms = p.drift_speed_kmh * _KMH_TO_MS
    mask = (
        (df["displacement_m"] > 0)
        & (df["displacement_m"] < p.drift_radius_m)
        & (df["speed_ms"] < drift_speed_ms)
        & (df["quality_flag"] == "VALID")
    )
    df.loc[mask, "quality_flag"] = "DRIFT"


def _flag_gaps(df: pd.DataFrame, p: CleanParams) -> None:
    threshold_seconds = p.gap_threshold_min * 60.0
    mask = (df["dt_seconds"] > threshold_seconds) & (df["quality_flag"] == "VALID")
    df.loc[mask, "quality_flag"] = "GAP_FOLLOWS"


def _null_gap_edges(df: pd.DataFrame) -> None:
    """Null derived columns on GAP_FOLLOWS pings (the B→C edge spanning the gap)."""
    mask = df["quality_flag"] == "GAP_FOLLOWS"
    if mask.any():
        df.loc[mask, ["speed_ms", "bearing_deg", "displacement_m"]] = _F32_NAN


# ── Misc helpers ────────────────────────────────────────────────────


def _detect_duplicates(df: pd.DataFrame) -> np.ndarray:
    """True for pings whose (lat, lon) match the previous ping exactly."""
    n = len(df)
    if n < 2:
        return np.zeros(n, dtype=bool)
    prev_lat = df["lat"].shift(1).to_numpy()
    prev_lon = df["lon"].shift(1).to_numpy()
    cur_lat = df["lat"].to_numpy()
    cur_lon = df["lon"].to_numpy()
    result: np.ndarray = (cur_lat == prev_lat) & (cur_lon == prev_lon)
    return result.astype(bool)


def _empty_cleaned_frame() -> pd.DataFrame:
    """Empty frame with all CleanedPingsSchema columns at the canonical dtypes."""
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
        }
    )
