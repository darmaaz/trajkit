"""End-to-end pipeline test on a synthetic pedestrian trace.

Synthesises a realistic pedestrian trace — stays at locations
interleaved with walking transits — and exercises the full pipeline:

    raw pings → clean → segment → aggregate → episode → embed_segments

Runs in CI without external data dependencies. A real Geolife `.plt`
example lives under ``examples/geolife/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trajkit.clean import CleanParams, clean
from trajkit.embed import EmbedParams, embed_segments
from trajkit.episode import EpisodeParams, detect_episodes
from trajkit.segment import SegmentParams, aggregate_segments, segment

# Walking-scale calibration: tighter envelope (R=30m) and shorter
# transit-grace (T=120s) than the module defaults.
PEDESTRIAN_SEGMENT = SegmentParams(
    stop_speed_kmh=1.0, resume_speed_kmh=3.0, max_stop_displacement_m=50.0
)
PEDESTRIAN_EPISODE = EpisodeParams(R_m=30.0, T_s=120.0, min_stay_s=120.0)
PEDESTRIAN_EMBED = EmbedParams(spatial_bounds=(39.95, 40.02, 116.28, 116.36))


# ── Synthetic pedestrian trace ──────────────────────────────────────


def _make_pedestrian_day(entity_id: str = "p1") -> pd.DataFrame:
    """Build a realistic pedestrian trace: home → coffee → home.

    Pings at 1 Hz. Walking pace ~6 km/h (1.67 m/s). Stays are at clearly
    distinct locations so the episode detector can separate them under
    ``R_m=30``.
    """
    home_lat, home_lon = 39.990, 116.320
    coffee_lat, coffee_lon = 39.998, 116.320  # ~890 m north of home

    parts: list[tuple[np.ndarray, np.ndarray]] = []
    parts.append(_stay_block(home_lat, home_lon, 300))
    parts.append(_walk_block(home_lat, home_lon, coffee_lat, coffee_lon, 180))
    parts.append(_stay_block(coffee_lat, coffee_lon, 480))
    parts.append(_walk_block(coffee_lat, coffee_lon, home_lat, home_lon, 180))
    parts.append(_stay_block(home_lat, home_lon, 300))

    lats = np.concatenate([p[0] for p in parts])
    lons = np.concatenate([p[1] for p in parts])
    n = len(lats)
    ts = pd.date_range("2026-01-01 08:00:00", periods=n, freq="1s", tz="UTC").astype(
        "datetime64[ns, UTC]"
    )

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


def _stay_block(lat: float, lon: float, duration_s: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(duration_s, lat, dtype=np.float64),
        np.full(duration_s, lon, dtype=np.float64),
    )


def _walk_block(
    lat0: float, lon0: float, lat1: float, lon1: float, duration_s: int
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(lat0, lat1, duration_s, dtype=np.float64),
        np.linspace(lon0, lon1, duration_s, dtype=np.float64),
    )


def _run_pipeline(pings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Compose the L1 stages on a single entity."""
    cleaned = clean(pings, CleanParams())
    segs = aggregate_segments(segment(cleaned, PEDESTRIAN_SEGMENT), PEDESTRIAN_SEGMENT)
    eps = detect_episodes(segs, PEDESTRIAN_EPISODE)
    vectors, _ = embed_segments(segs, PEDESTRIAN_EMBED)
    return segs, eps, vectors


# ── Tests ───────────────────────────────────────────────────────────


def test_pedestrian_pipeline_runs_end_to_end() -> None:
    pings = _make_pedestrian_day()
    segs, eps, vectors = _run_pipeline(pings)
    assert len(segs) > 0
    assert len(eps) > 0
    assert vectors.shape[0] == len(segs)


def test_pedestrian_episodes_have_both_stays_and_transits() -> None:
    pings = _make_pedestrian_day()
    _, eps, _ = _run_pipeline(pings)
    types = set(eps["episode_type"].astype(str).unique())
    assert "STAY" in types
    assert "TRANSIT" in types


def test_pedestrian_detects_separate_home_and_coffee_stays() -> None:
    pings = _make_pedestrian_day()
    _, eps, _ = _run_pipeline(pings)
    stays = eps[eps["episode_type"] == "STAY"]
    assert len(stays) >= 2


def test_pedestrian_stays_are_geographically_separable() -> None:
    pings = _make_pedestrian_day()
    _, eps, _ = _run_pipeline(pings)
    stays = eps[eps["episode_type"] == "STAY"]
    anchors = stays[["anchor_lat", "anchor_lon"]].dropna().to_numpy()
    assert len(anchors) >= 2
    # Pairwise lat differences span the home-coffee gap (~880 m ≈ 0.0079°)
    assert float(anchors[:, 0].max() - anchors[:, 0].min()) > 0.005


def test_pedestrian_segment_vectors_have_expected_dim() -> None:
    pings = _make_pedestrian_day()
    _, _, vectors = _run_pipeline(pings)
    assert vectors.shape[1] == PEDESTRIAN_EMBED.expected_dim()
    assert vectors.dtype == np.float32


@pytest.mark.parametrize("entity_id", ["p1", "different_id"])
def test_pipeline_does_not_crash_on_varying_entity_ids(entity_id: str) -> None:
    pings = _make_pedestrian_day(entity_id=entity_id)
    _run_pipeline(pings)
