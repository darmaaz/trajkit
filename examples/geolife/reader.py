"""Parser for Microsoft Geolife ``.plt`` trajectory files.

Geolife format (from the dataset's README):

    Lines 1-6: fixed header, ignored.
    Lines 7+:  lat,lon,0,altitude_feet,date_serial,date_str,time_str

Date+time give us a tz-naive UTC timestamp; we localise on read.

The ``read_user`` helper walks a single user's ``Trajectory/`` subdirectory,
concatenates all ``.plt`` files for that user, and returns a
``PingsSchema``-compatible DataFrame ready for ``trajkit.iter_entities``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Microsoft's PLT files have 6 fixed header lines before the data starts.
_PLT_HEADER_LINES = 6


def read_plt(path: Path) -> pd.DataFrame:
    """Parse a single Geolife ``.plt`` file into a 4-column DataFrame.

    Returns columns: ``lat``, ``lon``, ``ts``, ``altitude_m``. Empty
    DataFrame if the file has no data rows after the header.
    """
    df = pd.read_csv(
        path,
        skiprows=_PLT_HEADER_LINES,
        header=None,
        names=["lat", "lon", "_zero", "altitude_ft", "_serial", "date", "time"],
        dtype={
            "lat": np.float64,
            "lon": np.float64,
            "_zero": np.int64,
            "altitude_ft": np.float64,
            "_serial": np.float64,
            "date": str,
            "time": str,
        },
    )
    if len(df) == 0:
        return pd.DataFrame(columns=["lat", "lon", "ts", "altitude_m"])
    ts = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S", utc=True
    )
    return pd.DataFrame(
        {
            "lat": df["lat"].astype(np.float64),
            "lon": df["lon"].astype(np.float64),
            "ts": ts,
            "altitude_m": df["altitude_ft"] * 0.3048,  # feet → metres
        }
    )


def read_user(user_dir: Path, entity_id: str | None = None) -> pd.DataFrame:
    """Read all ``.plt`` files for a single Geolife user.

    Parameters
    ----------
    user_dir
        Path to a user directory (e.g. ``Geolife/Data/000/``). Contains a
        ``Trajectory/`` subdirectory of ``.plt`` files.
    entity_id
        Identifier to use for the ``entity_id`` column. Defaults to
        ``user_dir.name`` (the 3-digit user folder).

    Returns
    -------
    pd.DataFrame
        ``PingsSchema``-compatible frame with columns ``entity_id``,
        ``ts``, ``lat``, ``lon``, ``speed_ms``, ``bearing_deg``. Empty if
        the user has no trajectories.
    """
    eid = entity_id if entity_id is not None else user_dir.name
    traj_dir = user_dir / "Trajectory"
    if not traj_dir.exists():
        msg = f"no Trajectory/ subdirectory at {user_dir}"
        raise FileNotFoundError(msg)

    parts: list[pd.DataFrame] = []
    for plt_path in sorted(traj_dir.glob("*.plt")):
        sub = read_plt(plt_path)
        if not sub.empty:
            parts.append(sub)

    if not parts:
        return _empty_pings(eid)

    combined = pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True)
    return pd.DataFrame(
        {
            "entity_id": pd.Series([eid] * len(combined), dtype="string"),
            "ts": combined["ts"].astype("datetime64[ns, UTC]"),
            "lat": combined["lat"].astype(np.float64),
            "lon": combined["lon"].astype(np.float64),
            "speed_ms": pd.Series([np.nan] * len(combined), dtype=np.float32),
            "bearing_deg": pd.Series([np.nan] * len(combined), dtype=np.float32),
        }
    )


def discover_users(data_dir: Path) -> list[str]:
    """List user IDs present in a Geolife ``Data/`` directory."""
    if not data_dir.is_dir():
        msg = f"data_dir is not a directory: {data_dir}"
        raise FileNotFoundError(msg)
    return sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "Trajectory").is_dir()
    )


def _empty_pings(entity_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": pd.Series([], dtype="string"),
            "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "lat": pd.Series([], dtype=np.float64),
            "lon": pd.Series([], dtype=np.float64),
            "speed_ms": pd.Series([], dtype=np.float32),
            "bearing_deg": pd.Series([], dtype=np.float32),
        }
    )
