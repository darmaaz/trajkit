"""Per-entity cleaning: dedup, kinematics, quality flags.

Single-entity input sorted by ``ts``; trusts the contract. The flag
precedence rule (DEVICE_FAULT > SPEED_OUTLIER > GAP_FOLLOWS > DRIFT) is
the design point — see ``docs/design/clean.md`` for the rationale.
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
        Frozen ``CleanParams``. Defaults are scale-class agnostic; pass
        a custom instance to retune.

    Returns
    -------
    pd.DataFrame
        A new frame validated by ``CleanedPingsSchema``: original PingsSchema
        columns plus ``dt_seconds``, ``displacement_m``, ``is_duplicate``,
        ``quality_flag``, ``merge_count``, ``run_duration_s``. The latter two
        are null by default; downstream callers may populate them when
        consolidating duplicate-position runs.
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

    # Flag precedence: DEVICE_FAULT > SPEED_OUTLIER > GAP_FOLLOWS > DRIFT > VALID.
    # See docs/design/clean.md for rationale (esp. GAP_FOLLOWS > DRIFT).
    _flag_device_faults(df, reported_speed_ms, p)
    _flag_speed_outliers(df, p)
    _null_outlier_edges(df)
    _flag_gaps(df, p)
    _null_gap_edges(df)
    _flag_drift(df, p)

    df["is_duplicate"] = _detect_duplicates(df)

    df["merge_count"] = pd.array([pd.NA] * len(df), dtype="Int32")
    df["run_duration_s"] = np.full(len(df), np.nan, dtype=np.float32)

    return df


# ── Kinematics ──────────────────────────────────────────────────────


def _derive_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``dt_seconds``, ``displacement_m``, ``speed_ms``, ``bearing_deg``."""
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
    """Mark all rows ``DEVICE_FAULT`` if the entity is stuck-position with reported motion."""
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
    """Null derived columns on SPEED_OUTLIER pings and the following row."""
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
    """Null derived columns on GAP_FOLLOWS pings."""
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
