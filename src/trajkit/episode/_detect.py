"""Per-entity episode detection: spatial-envelope closure rule.

Implements ``detect_episodes``. Two-pass algorithm:

1. Greedy left-to-right scan finds STAY episodes — maximal segment runs
   whose centroids stay within ``R`` of the running anchor centroid,
   with grace window ``T`` on departures.
2. Maximal runs of segments not claimed by any stay become TRANSIT
   episodes, split where inter-segment time gaps exceed ``T``.

See ``docs/design/episode.md`` for the full specification including
edge-case handling.
"""

from __future__ import annotations

from typing import TypedDict

import h3
import numpy as np
import pandas as pd

from trajkit.episode._params import EpisodeParams

_EPISODE_ID_DIGITS = 5  # zero-padded suffix → up to 99,999 episodes per entity
_EARTH_RADIUS_M = 6_371_000.0


class _StayRecord(TypedDict):
    type: str
    first_idx: int
    last_idx: int
    anchor_lat: float
    anchor_lon: float
    envelope_radius_m: float


class _TransitRecord(TypedDict):
    type: str
    first_idx: int
    last_idx: int


def detect_episodes(
    segments_df: pd.DataFrame, params: EpisodeParams | None = None
) -> pd.DataFrame:
    """Group segments into STAY / TRANSIT episodes for one entity.

    Parameters
    ----------
    segments_df
        Output of ``trajkit.segment.aggregate_segments`` for one entity.
        Sorted by ``start_ts``.
    params
        Frozen ``EpisodeParams``; defaults are scale-class agnostic.

    Returns
    -------
    pd.DataFrame
        New frame conforming to ``EpisodesSchema``.
    """
    p = params if params is not None else EpisodeParams()
    if len(segments_df) == 0:
        return _empty_episodes_frame()

    df = segments_df.sort_values("start_ts").reset_index(drop=True)
    entity_id = str(df["entity_id"].iloc[0])

    centroid_lat = ((df["start_lat"] + df["end_lat"]) / 2.0).to_numpy(dtype=np.float64)
    centroid_lon = ((df["start_lon"] + df["end_lon"]) / 2.0).to_numpy(dtype=np.float64)
    durations = df["duration_s"].to_numpy(dtype=np.float64)
    start_ts = df["start_ts"].to_numpy()
    end_ts = df["end_ts"].to_numpy()

    stays = _find_stays(centroid_lat, centroid_lon, durations, start_ts, end_ts, p)
    transits = _find_transits(start_ts, end_ts, stays, len(df), p)

    return _build_episodes_frame(df, stays, transits, entity_id, p.h3_resolution)


# ── Pass 1: stays ────────────────────────────────────────────────────


def _find_stays(
    cent_lat: np.ndarray,
    cent_lon: np.ndarray,
    durations: np.ndarray,
    start_ts: np.ndarray,
    end_ts: np.ndarray,
    p: EpisodeParams,
) -> list[_StayRecord]:
    """Greedy left-to-right scan for STAY episodes."""
    n = len(cent_lat)
    stays: list[_StayRecord] = []
    i = 0
    while i < n:
        first_idx, last_idx, anchor_lat, anchor_lon, max_radius = _grow_stay(
            i, cent_lat, cent_lon, durations, start_ts, end_ts, p
        )
        stay_duration_s = float(
            (end_ts[last_idx] - start_ts[first_idx]) / np.timedelta64(1, "s")
        )

        if stay_duration_s >= p.min_stay_s:
            stays.append(
                _StayRecord(
                    type="STAY",
                    first_idx=first_idx,
                    last_idx=last_idx,
                    anchor_lat=anchor_lat,
                    anchor_lon=anchor_lon,
                    envelope_radius_m=max_radius,
                )
            )
            i = last_idx + 1
        else:
            i += 1

    return stays


def _grow_stay(
    i: int,
    cent_lat: np.ndarray,
    cent_lon: np.ndarray,
    durations: np.ndarray,
    start_ts: np.ndarray,
    end_ts: np.ndarray,
    p: EpisodeParams,
) -> tuple[int, int, float, float, float]:
    """Try to grow a stay starting at index ``i``; return stay extent + anchor."""
    n = len(cent_lat)
    anchor_lat = float(cent_lat[i])
    anchor_lon = float(cent_lon[i])
    last_inside = i
    time_outside = 0.0

    inside_lat = [float(cent_lat[i])]
    inside_lon = [float(cent_lon[i])]

    for j in range(i + 1, n):
        # Inter-segment gap also closes the stay (gotcha #5).
        gap_s = float((start_ts[j] - end_ts[j - 1]) / np.timedelta64(1, "s"))
        if gap_s > p.T_s:
            break

        d = _haversine(anchor_lat, anchor_lon, float(cent_lat[j]), float(cent_lon[j]))
        if d <= p.R_m:
            inside_lat.append(float(cent_lat[j]))
            inside_lon.append(float(cent_lon[j]))
            anchor_lat = float(np.mean(inside_lat))
            anchor_lon = float(np.mean(inside_lon))
            last_inside = j
            time_outside = 0.0
        else:
            time_outside += float(durations[j])
            if time_outside >= p.T_s:
                break

    # Recompute max observed radius against the FINAL anchor — running mean
    # drifted while inside-points were added, so per-step radii are stale.
    inside_arr_lat = np.asarray(inside_lat, dtype=np.float64)
    inside_arr_lon = np.asarray(inside_lon, dtype=np.float64)
    radii = _haversine_array(
        anchor_lat, anchor_lon, inside_arr_lat, inside_arr_lon
    )
    max_radius = float(radii.max()) if len(radii) > 0 else 0.0

    return i, last_inside, anchor_lat, anchor_lon, max_radius


# ── Pass 2: transits ─────────────────────────────────────────────────


def _find_transits(
    start_ts: np.ndarray,
    end_ts: np.ndarray,
    stays: list[_StayRecord],
    n: int,
    p: EpisodeParams,
) -> list[_TransitRecord]:
    """Each maximal run of unclaimed segments becomes a TRANSIT, split on gaps."""
    in_stay = np.zeros(n, dtype=bool)
    for s in stays:
        in_stay[s["first_idx"] : s["last_idx"] + 1] = True

    transits: list[_TransitRecord] = []
    i = 0
    while i < n:
        if in_stay[i]:
            i += 1
            continue

        # Walk a run of non-stay segments, splitting where inter-segment gap > T.
        run_start = i
        chunk_start = i
        j = i
        while j < n and not in_stay[j]:
            if j > chunk_start:
                gap_s = float((start_ts[j] - end_ts[j - 1]) / np.timedelta64(1, "s"))
                if gap_s > p.T_s:
                    transits.append(
                        _TransitRecord(
                            type="TRANSIT",
                            first_idx=chunk_start,
                            last_idx=j - 1,
                        )
                    )
                    chunk_start = j
            j += 1
        # Emit the trailing chunk
        transits.append(
            _TransitRecord(
                type="TRANSIT",
                first_idx=chunk_start,
                last_idx=j - 1,
            )
        )
        i = j
        _ = run_start  # silence linter; documents that we tracked the run start

    return transits


# ── Output frame construction ────────────────────────────────────────


def _build_episodes_frame(
    df: pd.DataFrame,
    stays: list[_StayRecord],
    transits: list[_TransitRecord],
    entity_id: str,
    h3_resolution: int,
) -> pd.DataFrame:
    """Combine stays + transits, sort by start_ts, assign IDs, build a frame."""
    rows: list[dict[str, object]] = []

    for s in stays:
        rows.append(_stay_row(df, s, h3_resolution))
    for t in transits:
        rows.append(_transit_row(df, t))

    rows.sort(key=lambda r: r["start_ts"])  # type: ignore[arg-type,return-value]

    for n_, row in enumerate(rows, start=1):
        row["episode_id"] = f"ep_{entity_id}_{n_:0{_EPISODE_ID_DIGITS}d}"
        row["entity_id"] = entity_id

    if not rows:
        return _empty_episodes_frame()

    out = pd.DataFrame(rows)
    return _enforce_dtypes(out)


def _stay_row(
    df: pd.DataFrame, s: _StayRecord, h3_resolution: int
) -> dict[str, object]:
    """Build a STAY row from a stay record."""
    first, last = s["first_idx"], s["last_idx"]
    start_ts = df["start_ts"].iloc[first]
    end_ts = df["end_ts"].iloc[last]
    duration_s = float((end_ts - start_ts).total_seconds())
    duration_s = max(duration_s, 0.0)

    segment_ids = df["segment_id"].iloc[first : last + 1].astype(str).tolist()

    return {
        "episode_type": "STAY",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": duration_s,
        "segment_ids": segment_ids,
        "n_segments": last - first + 1,
        "anchor_lat": s["anchor_lat"],
        "anchor_lon": s["anchor_lon"],
        "anchor_h3": h3.latlng_to_cell(s["anchor_lat"], s["anchor_lon"], h3_resolution),
        "envelope_radius_m": s["envelope_radius_m"],
        "start_lat": None,
        "start_lon": None,
        "end_lat": None,
        "end_lon": None,
        "displacement_m": None,
        "path_length_m": None,
        "straightness": None,
    }


def _transit_row(df: pd.DataFrame, t: _TransitRecord) -> dict[str, object]:
    """Build a TRANSIT row from a transit record."""
    first, last = t["first_idx"], t["last_idx"]
    start_ts = df["start_ts"].iloc[first]
    end_ts = df["end_ts"].iloc[last]
    duration_s = float((end_ts - start_ts).total_seconds())
    duration_s = max(duration_s, 0.0)

    start_lat = float(df["start_lat"].iloc[first])
    start_lon = float(df["start_lon"].iloc[first])
    end_lat = float(df["end_lat"].iloc[last])
    end_lon = float(df["end_lon"].iloc[last])

    path_length_m = float(df["path_length_m"].iloc[first : last + 1].sum())
    displacement_m = _haversine(start_lat, start_lon, end_lat, end_lon)

    if path_length_m > 0.0:
        raw = displacement_m / path_length_m
        straightness = float(min(max(raw, 0.0), 1.0))
    else:
        straightness = 0.0

    segment_ids = df["segment_id"].iloc[first : last + 1].astype(str).tolist()

    return {
        "episode_type": "TRANSIT",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": duration_s,
        "segment_ids": segment_ids,
        "n_segments": last - first + 1,
        "anchor_lat": None,
        "anchor_lon": None,
        "anchor_h3": None,
        "envelope_radius_m": None,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "displacement_m": displacement_m,
        "path_length_m": path_length_m,
        "straightness": straightness,
    }


# ── Geo utilities ────────────────────────────────────────────────────


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Scalar great-circle distance in metres."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    )
    return float(2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


def _haversine_array(
    lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance from a single anchor to N points."""
    p0 = np.radians(lat0)
    pp = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p0) * np.cos(pp) * np.sin(dlmb / 2.0) ** 2
    result: np.ndarray = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return result


# ── Dtype enforcement and empty frame ────────────────────────────────


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a freshly built episodes frame to canonical EpisodesSchema dtypes."""
    df = df.copy()
    df["episode_id"] = df["episode_id"].astype("string")
    df["entity_id"] = df["entity_id"].astype("string")
    df["episode_type"] = df["episode_type"].astype("string")
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True).astype("datetime64[ns, UTC]")
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True).astype("datetime64[ns, UTC]")
    df["duration_s"] = df["duration_s"].astype(np.float32)
    df["n_segments"] = df["n_segments"].astype(np.int32)

    # STAY-only columns: cast non-null entries to canonical, leave None as null.
    df["anchor_lat"] = df["anchor_lat"].astype("Float64").astype(np.float64)
    df["anchor_lon"] = df["anchor_lon"].astype("Float64").astype(np.float64)
    df["anchor_h3"] = df["anchor_h3"].astype("string")
    df["envelope_radius_m"] = df["envelope_radius_m"].astype("Float32").astype(np.float32)

    # TRANSIT-only columns: same treatment.
    df["start_lat"] = df["start_lat"].astype("Float64").astype(np.float64)
    df["start_lon"] = df["start_lon"].astype("Float64").astype(np.float64)
    df["end_lat"] = df["end_lat"].astype("Float64").astype(np.float64)
    df["end_lon"] = df["end_lon"].astype("Float64").astype(np.float64)
    df["displacement_m"] = df["displacement_m"].astype("Float32").astype(np.float32)
    df["path_length_m"] = df["path_length_m"].astype("Float32").astype(np.float32)
    df["straightness"] = df["straightness"].astype("Float32").astype(np.float32)

    column_order = [
        "episode_id",
        "entity_id",
        "episode_type",
        "start_ts",
        "end_ts",
        "duration_s",
        "segment_ids",
        "n_segments",
        "anchor_lat",
        "anchor_lon",
        "anchor_h3",
        "envelope_radius_m",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "displacement_m",
        "path_length_m",
        "straightness",
    ]
    return df[column_order]


def _empty_episodes_frame() -> pd.DataFrame:
    """Empty EpisodesSchema-shaped frame at canonical dtypes."""
    return pd.DataFrame(
        {
            "episode_id": pd.Series([], dtype="string"),
            "entity_id": pd.Series([], dtype="string"),
            "episode_type": pd.Series([], dtype="string"),
            "start_ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "end_ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "duration_s": pd.Series([], dtype=np.float32),
            "segment_ids": pd.Series([], dtype=object),
            "n_segments": pd.Series([], dtype=np.int32),
            "anchor_lat": pd.Series([], dtype=np.float64),
            "anchor_lon": pd.Series([], dtype=np.float64),
            "anchor_h3": pd.Series([], dtype="string"),
            "envelope_radius_m": pd.Series([], dtype=np.float32),
            "start_lat": pd.Series([], dtype=np.float64),
            "start_lon": pd.Series([], dtype=np.float64),
            "end_lat": pd.Series([], dtype=np.float64),
            "end_lon": pd.Series([], dtype=np.float64),
            "displacement_m": pd.Series([], dtype=np.float32),
            "path_length_m": pd.Series([], dtype=np.float32),
            "straightness": pd.Series([], dtype=np.float32),
        }
    )
