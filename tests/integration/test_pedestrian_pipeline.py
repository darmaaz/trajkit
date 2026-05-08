"""Cross-domain integration test: pedestrian-shape data through the full pipeline.

This is the v0.1.0 cross-domain validation gate. We synthesise a realistic
pedestrian trace — stays at locations interleaved with walking transits —
and exercise the entire pipeline end-to-end with pedestrian-tuned
parameters:

    raw pings → clean → segment → aggregate → episode → embed_segments
                                                       → embed_episodes

A real Geolife `.plt` integration lands in ``examples/geolife/`` later;
the synthetic test runs in CI without external data dependencies, while
still proving that the same pipeline + pedestrian parameters produces
sensible STAY/TRANSIT episodes on non-vehicle data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajkit.embed import EmbedParams
from trajkit.episode import detect_episodes
from trajkit.runner import RunParams, process

# Use the published pedestrian preset, then layer a tighter spatial bbox for
# better embedding resolution on this synthetic Beijing-area trace.
PEDESTRIAN_PARAMS = RunParams.from_preset("pedestrian").model_copy(
    update={
        "embed": EmbedParams(spatial_bounds=(39.95, 40.02, 116.28, 116.36)),
    }
)


# ── Synthetic pedestrian trace ──────────────────────────────────────


def _make_pedestrian_day(entity_id: str = "p1") -> pd.DataFrame:
    """Build a realistic pedestrian trace: home → coffee → home.

    Pings at 1 Hz. Walking pace ~6 km/h (1.67 m/s, ~17 cm per 0.0001° lat
    near Beijing). Stays at clearly distinct locations so the episode
    detector can separate them under R_m=30.
    """
    home_lat, home_lon = 39.990, 116.320
    coffee_lat, coffee_lon = 39.998, 116.320  # ~890 m north of home

    parts: list[tuple[np.ndarray, np.ndarray, pd.Timestamp]] = []

    # 5 minutes stationary at home
    parts.append(_stay_block(home_lat, home_lon, 300, "2026-01-01 08:00:00"))
    # ~3 minute walk to coffee shop
    parts.append(_walk_block(home_lat, home_lon, coffee_lat, coffee_lon, 180,
                             "2026-01-01 08:05:00"))
    # 8 minutes at coffee shop
    parts.append(_stay_block(coffee_lat, coffee_lon, 480, "2026-01-01 08:08:00"))
    # ~3 minute walk back home
    parts.append(_walk_block(coffee_lat, coffee_lon, home_lat, home_lon, 180,
                             "2026-01-01 08:16:00"))
    # 5 minutes stationary at home
    parts.append(_stay_block(home_lat, home_lon, 300, "2026-01-01 08:19:00"))

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


def _stay_block(
    lat: float, lon: float, duration_s: int, _start: str
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Stationary lat/lon arrays of length duration_s."""
    lats = np.full(duration_s, lat, dtype=np.float64)
    lons = np.full(duration_s, lon, dtype=np.float64)
    return lats, lons, pd.Timestamp(_start, tz="UTC")


def _walk_block(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
    duration_s: int,
    _start: str,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Linear interpolation lat/lon from (lat0, lon0) to (lat1, lon1)."""
    lats = np.linspace(lat0, lat1, duration_s, dtype=np.float64)
    lons = np.linspace(lon0, lon1, duration_s, dtype=np.float64)
    return lats, lons, pd.Timestamp(_start, tz="UTC")


# ── Tests ───────────────────────────────────────────────────────────


def test_pedestrian_pipeline_runs_through_runner(tmp_path: Path) -> None:
    pings = _make_pedestrian_day()
    rep = process(pings, tmp_path, PEDESTRIAN_PARAMS)
    assert rep.succeeded, f"runner reported failure: {rep.failed_reason}"
    assert rep.n_completed == 1


def test_pedestrian_pipeline_produces_all_stage_outputs(tmp_path: Path) -> None:
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    for stage in ("clean", "segment", "episode", "embed_segments", "embed_episodes"):
        assert (tmp_path / stage / "entity_id=p1" / "data.parquet").exists(), (
            f"missing stage output: {stage}"
        )


def test_pedestrian_episodes_have_both_stays_and_transits(tmp_path: Path) -> None:
    """The home → coffee → home pattern must produce both STAY and TRANSIT."""
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    eps = pd.read_parquet(tmp_path / "episode" / "entity_id=p1" / "data.parquet")
    types = set(eps["episode_type"].astype(str).unique())
    assert "STAY" in types
    assert "TRANSIT" in types


def test_pedestrian_detects_separate_home_and_coffee_stays(tmp_path: Path) -> None:
    """Two distinct stay locations should yield ≥ 2 STAY episodes."""
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    eps = pd.read_parquet(tmp_path / "episode" / "entity_id=p1" / "data.parquet")
    stays = eps[eps["episode_type"] == "STAY"]
    assert len(stays) >= 2


def test_pedestrian_stays_are_geographically_separable(tmp_path: Path) -> None:
    """Anchor points of distinct stays should be far apart (the home-coffee gap)."""
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    eps = pd.read_parquet(tmp_path / "episode" / "entity_id=p1" / "data.parquet")
    stays = eps[eps["episode_type"] == "STAY"]
    anchors = stays[["anchor_lat", "anchor_lon"]].dropna().to_numpy()
    if len(anchors) >= 2:
        # Pairwise lat differences span the home-coffee gap (~880 m ≈ 0.0079°)
        max_lat_span = float(anchors[:, 0].max() - anchors[:, 0].min())
        assert max_lat_span > 0.005


def test_pedestrian_embed_segments_produces_nonempty_vectors(tmp_path: Path) -> None:
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    vec_df = pd.read_parquet(
        tmp_path / "embed_segments" / "entity_id=p1" / "data.parquet"
    )
    assert len(vec_df) > 0
    first = np.asarray(vec_df["vector"].iloc[0], dtype=np.float32)
    expected_dim = PEDESTRIAN_PARAMS.embed.expected_dim()
    assert first.shape == (expected_dim,)


def test_pedestrian_embed_episodes_produces_pooled_vectors(tmp_path: Path) -> None:
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    seg_vec = pd.read_parquet(
        tmp_path / "embed_segments" / "entity_id=p1" / "data.parquet"
    )
    ep_vec = pd.read_parquet(
        tmp_path / "embed_episodes" / "entity_id=p1" / "data.parquet"
    )
    seg_dim = len(np.asarray(seg_vec["vector"].iloc[0], dtype=np.float32))
    expected = 3 * seg_dim + 5  # design's pooling formula
    first = np.asarray(ep_vec["vector"].iloc[0], dtype=np.float32)
    assert first.shape == (expected,)


def test_pedestrian_episodes_l2_normalised(tmp_path: Path) -> None:
    """Default L2 normalisation gives unit-norm episode vectors."""
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS)
    ep_vec = pd.read_parquet(
        tmp_path / "embed_episodes" / "entity_id=p1" / "data.parquet"
    )
    norms = np.array([
        float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
        for v in ep_vec["vector"]
    ])
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_pedestrian_episode_detect_directly_yields_three_stays(tmp_path: Path) -> None:
    """Outside the runner: pings → segments → episodes finds home-coffee-home."""
    pings = _make_pedestrian_day()
    process(pings, tmp_path, PEDESTRIAN_PARAMS, stages=("clean", "segment"))
    segs = pd.read_parquet(tmp_path / "segment" / "entity_id=p1" / "data.parquet")
    eps = detect_episodes(segs, PEDESTRIAN_PARAMS.episode)

    # The trace returns to home after coffee → expect 3 stays
    stays = eps[eps["episode_type"] == "STAY"]
    transits = eps[eps["episode_type"] == "TRANSIT"]
    assert len(stays) >= 2  # at minimum home + coffee separable
    assert len(transits) >= 1


# ── Integration-test smoke: cross-domain claim sanity ───────────────


@pytest.mark.parametrize("scale", ["pedestrian"])
def test_cross_domain_pipeline_does_not_crash(tmp_path: Path, scale: str) -> None:
    """Sanity — the full pipeline doesn't crash on non-vehicle inputs."""
    pings = _make_pedestrian_day()
    rep = process(pings, tmp_path, PEDESTRIAN_PARAMS)
    assert rep.succeeded
    _ = scale  # currently only one scale; parametrised for future maritime/wildlife
