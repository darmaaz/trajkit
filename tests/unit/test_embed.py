"""Unit tests for ``trajkit.embed``.

End-to-end through clean → segment → aggregate → embed_segments,
plus block-level and contract checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit.clean import clean
from trajkit.embed import EmbedParams, FeaturePlugin, embed_segments
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
