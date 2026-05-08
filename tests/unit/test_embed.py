"""Unit tests for ``trajkit.embed``.

End-to-end through clean → segment → aggregate → embed_segments →
episode → embed_episodes; plus block-level and contract checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit.clean import clean
from trajkit.embed import (
    EmbedParams,
    FeaturePlugin,
    baseline_zscores,
    embed_episodes,
    embed_segments,
)
from trajkit.episode import detect_episodes
from trajkit.segment import aggregate_segments, segment

# ── End-to-end fixture: produce real segments + episodes ────────────


def _ts(start: str, n: int, freq: str = "1s") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq, tz="UTC").astype(
        "datetime64[ns, UTC]"
    )


def _pings(
    n: int,
    *,
    lat_path: np.ndarray | None = None,
    lon_path: np.ndarray | None = None,
    speed_ms: np.ndarray | None = None,
    bearing_deg: np.ndarray | None = None,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    if lat_path is None:
        lat_path = np.full(n, 19.4, dtype=np.float64)
    if lon_path is None:
        lon_path = np.full(n, -99.2, dtype=np.float64)
    if speed_ms is None:
        speed_ms = np.full(n, np.nan, dtype=np.float32)
    if bearing_deg is None:
        bearing_deg = np.full(n, np.nan, dtype=np.float32)
    return pd.DataFrame(
        {
            "entity_id": pd.Series(["v1"] * n, dtype="string"),
            "ts": _ts(start, n),
            "lat": lat_path.astype(np.float64),
            "lon": lon_path.astype(np.float64),
            "speed_ms": speed_ms.astype(np.float32),
            "bearing_deg": bearing_deg.astype(np.float32),
        }
    )


def _segments_for(n_pings: int = 600) -> pd.DataFrame:
    """End-to-end through clean → segment → aggregate."""
    pings = _pings(n_pings, lat_path=19.4 + np.arange(n_pings) * 0.0001)
    return aggregate_segments(segment(clean(pings)))


# ── embed_segments: contract ────────────────────────────────────────


def test_embed_segments_returns_float32_contiguous_array() -> None:
    segs = _segments_for()
    vectors, ids = embed_segments(segs)
    assert vectors.dtype == np.float32
    assert vectors.flags["C_CONTIGUOUS"]
    assert vectors.shape[0] == len(segs)
    assert vectors.shape[0] == len(ids)


def test_embed_segments_dimension_matches_expected_dim() -> None:
    segs = _segments_for()
    p = EmbedParams()
    vectors, _ = embed_segments(segs, p)
    assert vectors.shape[1] == p.expected_dim()


def test_embed_segments_handles_empty_input() -> None:
    empty = aggregate_segments(segment(clean(_pings(0))))
    vectors, ids = embed_segments(empty)
    assert vectors.shape == (0, EmbedParams().expected_dim())
    assert ids == []


def test_embed_segments_l2_normalises_when_enabled() -> None:
    segs = _segments_for()
    vectors, _ = embed_segments(segs, EmbedParams(l2_normalize=True))
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_segments_skips_l2_when_disabled() -> None:
    segs = _segments_for()
    vectors, _ = embed_segments(segs, EmbedParams(l2_normalize=False))
    norms = np.linalg.norm(vectors, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_embed_segments_ids_align_with_input_order() -> None:
    segs = _segments_for()
    _, ids = embed_segments(segs)
    assert ids == segs["segment_id"].astype(str).tolist()


# ── embed_segments: block correctness ───────────────────────────────


def test_embed_segments_segment_type_one_hot_block_is_one_hot() -> None:
    segs = _segments_for()
    vectors, _ = embed_segments(segs, EmbedParams(l2_normalize=False))
    p = EmbedParams()
    cyclic_dim = p.cyclic_harmonics * 2 * 2
    type_start = 8 + cyclic_dim  # kinematic(8) + cyclic
    type_end = type_start + 4
    one_hot = vectors[:, type_start:type_end]
    assert (one_hot.sum(axis=1) == 1.0).all()
    assert ((one_hot == 0.0) | (one_hot == 1.0)).all()


def test_embed_segments_spatial_block_clips_to_unit_interval() -> None:
    segs = _segments_for()
    vectors, _ = embed_segments(
        segs,
        EmbedParams(l2_normalize=False, spatial_bounds=(19.0, 20.0, -100.0, -99.0)),
    )
    p = EmbedParams()
    cyclic_dim = p.cyclic_harmonics * 2 * 2
    spatial_start = 8 + cyclic_dim + 4
    spatial_end = spatial_start + 4
    spatial = vectors[:, spatial_start:spatial_end]
    assert (spatial >= 0.0).all()
    assert (spatial <= 1.0).all()


def test_embed_segments_spatial_block_rejects_invalid_bounds() -> None:
    segs = _segments_for()
    with pytest.raises(ValueError, match="spatial_bounds"):
        embed_segments(segs, EmbedParams(spatial_bounds=(20.0, 19.0, -100.0, -99.0)))


# ── Plugin contract ─────────────────────────────────────────────────


class _ConstantPlugin:
    """Test plugin emitting a fixed value across all segments."""

    name = "constant"
    dim = 3

    def compute(self, segments_df: pd.DataFrame) -> np.ndarray:
        return np.full((len(segments_df), self.dim), 0.5, dtype=np.float32)


class _BadShapePlugin:
    name = "bad-shape"
    dim = 3

    def compute(self, segments_df: pd.DataFrame) -> np.ndarray:
        # Wrong number of columns → must raise
        return np.zeros((len(segments_df), 5), dtype=np.float32)


def test_embed_segments_plugin_extends_dimension() -> None:
    segs = _segments_for()
    plugin = _ConstantPlugin()
    p = EmbedParams()
    vectors, _ = embed_segments(segs, p, features=(plugin,))
    assert vectors.shape[1] == p.expected_dim((plugin,))
    assert vectors.shape[1] == 32 + 3


def test_embed_segments_plugin_shape_mismatch_raises() -> None:
    segs = _segments_for()
    with pytest.raises(ValueError, match="bad-shape"):
        embed_segments(segs, features=(_BadShapePlugin(),))


def test_feature_plugin_protocol_is_runtime_checkable() -> None:
    assert isinstance(_ConstantPlugin(), FeaturePlugin)


# ── embed_episodes ──────────────────────────────────────────────────


def test_embed_episodes_returns_3d_plus_5_columns() -> None:
    segs = _segments_for()
    vectors, ids = embed_segments(segs)
    eps = detect_episodes(segs)
    out, out_ids = embed_episodes(eps, vectors, ids)
    expected_dim = 3 * vectors.shape[1] + 5
    assert out.shape[1] == expected_dim
    assert out.dtype == np.float32
    assert len(out_ids) == out.shape[0]


def test_embed_episodes_l2_normalised_by_default() -> None:
    segs = _segments_for()
    vectors, ids = embed_segments(segs)
    eps = detect_episodes(segs)
    out, _ = embed_episodes(eps, vectors, ids)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_episodes_handles_single_segment_episode() -> None:
    segs = _segments_for(n_pings=300)
    eps = detect_episodes(segs)
    # Force a single-segment episode by mocking a tiny episodes_df
    one = eps.iloc[[0]].copy()
    one.at[one.index[0], "segment_ids"] = [segs["segment_id"].iloc[0]]  # noqa: PD008
    one.at[one.index[0], "n_segments"] = np.int32(1)  # noqa: PD008
    vectors, ids = embed_segments(segs)
    out, _ = embed_episodes(one, vectors, ids)
    assert out.shape[0] == 1
    assert out.shape[1] == 3 * vectors.shape[1] + 5


def test_embed_episodes_drops_episodes_with_no_segment_overlap() -> None:
    segs = _segments_for()
    vectors, ids = embed_segments(segs)
    eps = detect_episodes(segs)
    # Replace segment_ids with bogus IDs not in the segment_ids list
    eps_modified = eps.copy()
    eps_modified.at[eps_modified.index[0], "segment_ids"] = ["nonexistent_seg"]  # noqa: PD008
    out, out_ids = embed_episodes(eps_modified, vectors, ids)
    # First episode dropped; subsequent ones kept
    assert "ep_v1_00001" not in out_ids


def test_embed_episodes_handles_empty_episodes() -> None:
    segs = _segments_for()
    vectors, ids = embed_segments(segs)
    eps = detect_episodes(segs).iloc[:0]
    out, out_ids = embed_episodes(eps, vectors, ids)
    assert out.shape == (0, 3 * vectors.shape[1] + 5)
    assert out_ids == []


def test_embed_episodes_rejects_misaligned_segment_inputs() -> None:
    vectors = np.zeros((3, 32), dtype=np.float32)
    ids = ["a", "b"]  # one fewer than vectors
    with pytest.raises(ValueError, match="disagree"):
        embed_episodes(pd.DataFrame(columns=["episode_id"]), vectors, ids)


# ── baseline_zscores ────────────────────────────────────────────────


def test_baseline_zscores_adds_metric_columns() -> None:
    segs = _segments_for()
    baselines = pd.DataFrame(
        {
            "entity_id": ["v1"],
            "metric": ["duration_s"],
            "mean": np.array([60.0], dtype=np.float32),
            "std": np.array([10.0], dtype=np.float32),
            "n_samples": np.array([100], dtype=np.int32),
            "is_fallback": [False],
        }
    )
    out = baseline_zscores(segs, baselines, cohort_keys=["entity_id"])
    assert "duration_s_z" in out.columns
    expected = (segs["duration_s"].astype(np.float64) - 60.0) / 10.0
    np.testing.assert_allclose(
        out["duration_s_z"].astype(np.float64).to_numpy(),
        expected.to_numpy(),
        rtol=1e-3,
    )


def test_baseline_zscores_emits_nan_for_unknown_cohort() -> None:
    segs = _segments_for()
    # Baseline has only entity_id="v999" but segments are entity_id="v1"
    baselines = pd.DataFrame(
        {
            "entity_id": ["v999"],
            "metric": ["duration_s"],
            "mean": np.array([60.0], dtype=np.float32),
            "std": np.array([10.0], dtype=np.float32),
            "n_samples": np.array([100], dtype=np.int32),
            "is_fallback": [False],
        }
    )
    out = baseline_zscores(segs, baselines, cohort_keys=["entity_id"])
    assert out["duration_s_z"].isna().all()


def test_baseline_zscores_skips_unknown_metric() -> None:
    segs = _segments_for()
    baselines = pd.DataFrame(
        {
            "entity_id": ["v1"],
            "metric": ["nonexistent_metric"],
            "mean": np.array([1.0], dtype=np.float32),
            "std": np.array([1.0], dtype=np.float32),
            "n_samples": np.array([100], dtype=np.int32),
            "is_fallback": [False],
        }
    )
    out = baseline_zscores(segs, baselines, cohort_keys=["entity_id"])
    # Metric not present in segments_df → silently skipped, no new column
    assert "nonexistent_metric_z" not in out.columns


def test_baseline_zscores_floors_std_with_epsilon() -> None:
    segs = _segments_for().head(3)
    baselines = pd.DataFrame(
        {
            "entity_id": ["v1"],
            "metric": ["duration_s"],
            "mean": np.array([0.0], dtype=np.float32),
            "std": np.array([0.0], dtype=np.float32),  # collapsed cohort
            "n_samples": np.array([100], dtype=np.int32),
            "is_fallback": [False],
        }
    )
    out = baseline_zscores(segs, baselines, cohort_keys=["entity_id"], epsilon=1.0)
    # No infinities — epsilon floored the std
    assert np.isfinite(out["duration_s_z"]).all()


def test_baseline_zscores_rejects_missing_cohort_key() -> None:
    segs = _segments_for()
    baselines = pd.DataFrame(
        {
            "metric": ["duration_s"],
            "mean": np.array([60.0], dtype=np.float32),
            "std": np.array([10.0], dtype=np.float32),
            "n_samples": np.array([100], dtype=np.int32),
            "is_fallback": [False],
        }
    )
    with pytest.raises(ValueError, match="not in baselines"):
        baseline_zscores(segs, baselines, cohort_keys=["entity_id"])


# ── EmbedParams ─────────────────────────────────────────────────────


def test_embed_params_are_frozen() -> None:
    p = EmbedParams()
    with pytest.raises(ValidationError):
        p.cyclic_harmonics = 99  # type: ignore[misc]


def test_embed_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        EmbedParams(junk="oops")  # type: ignore[call-arg]


def test_embed_params_expected_dim_includes_plugins() -> None:
    p = EmbedParams()
    plugin = _ConstantPlugin()
    assert p.expected_dim((plugin,)) == p.base_dim() + plugin.dim
