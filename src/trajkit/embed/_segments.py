"""Per-segment embedding: base recipe + feature-plugin extension.

Implements ``embed_segments``. The base recipe builds four blocks
(kinematic, cyclic, segment-type, spatial), concatenates them, then
appends each plugin's contribution. Output is a contiguous float32
array shaped ``(n_segments, total_dim)`` plus the row-aligned segment
ID list.

The base recipe applies ``log1p`` to kinematic features without
further standardisation. Cohort-relative scaling is a user concern,
applied (if desired) as a separate post-processing step.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from trajkit.embed._params import (
    CYCLIC_FEATURES,
    KINEMATIC_FEATURES,
    SEGMENT_TYPE_ORDER,
    EmbedParams,
    FeaturePlugin,
)

_F32 = np.float32


def embed_segments(
    segments_df: pd.DataFrame,
    params: EmbedParams | None = None,
    features: tuple[FeaturePlugin, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    """Vectorise a segments frame to a fixed-width float32 matrix.

    Parameters
    ----------
    segments_df
        Output of ``trajkit.segment.aggregate_segments``.
    params
        ``EmbedParams`` controlling cyclic harmonics, spatial bounds, and
        L2-normalisation.
    features
        Tuple of user-supplied ``FeaturePlugin`` instances. Each contributes
        a ``(n_segments, plugin.dim)`` block appended to the base recipe.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(vectors, segment_ids)``: ``vectors`` is a contiguous ``float32``
        array of shape ``(n_segments, params.expected_dim(features))``;
        ``segment_ids`` is row-aligned.
    """
    p = params if params is not None else EmbedParams()
    n = len(segments_df)
    if n == 0:
        return np.zeros((0, p.expected_dim(features)), dtype=_F32), []

    df = segments_df.reset_index(drop=True)

    blocks: list[np.ndarray] = [
        _kinematic_block(df),
        _cyclic_block(df, p.cyclic_harmonics),
        _segment_type_block(df),
        _spatial_block(df, p.spatial_bounds),
    ]
    for plugin in features:
        block = _validated_plugin_block(plugin, df)
        blocks.append(block)

    vectors = np.concatenate(blocks, axis=1, dtype=_F32)

    if p.l2_normalize:
        vectors = _l2_normalize(vectors, p.epsilon)

    expected = p.expected_dim(features)
    if vectors.shape != (n, expected):
        msg = (
            f"embed_segments: produced shape {vectors.shape}, "
            f"expected {(n, expected)}"
        )
        raise RuntimeError(msg)

    segment_ids = df["segment_id"].astype(str).tolist()
    return np.ascontiguousarray(vectors, dtype=_F32), segment_ids


# ── Base recipe blocks ──────────────────────────────────────────────


def _kinematic_block(df: pd.DataFrame) -> np.ndarray:
    """log1p of selected kinematic columns; NaNs filled with 0 first.

    The fillna-then-log1p ordering is mathematically identical to log1p-
    then-fillna(0) for non-negative inputs, but is faster (one operation
    over the array) and clearer about intent: we treat absent kinematics
    as a neutral zero rather than propagating NaN through the embedding.
    """
    n = len(df)
    out = np.zeros((n, len(KINEMATIC_FEATURES)), dtype=_F32)
    for j, col in enumerate(KINEMATIC_FEATURES):
        values = df[col].astype(np.float64).fillna(0.0).to_numpy()
        out[:, j] = np.log1p(np.maximum(values, 0.0)).astype(_F32)
    return out


def _cyclic_block(df: pd.DataFrame, harmonics: int) -> np.ndarray:
    """Sinusoidal encoding of hour-of-day and day-of-week.

    For each feature ``f`` with period ``P`` and harmonic ``k ∈ 1..K``:
    ``[sin(2πk·f/P), cos(2πk·f/P)]``. Captures cyclical structure without
    sharp boundaries at the period edge.
    """
    n = len(df)
    out = np.zeros((n, len(CYCLIC_FEATURES) * 2 * harmonics), dtype=_F32)
    start_ts = pd.DatetimeIndex(df["start_ts"])
    feature_values: dict[str, np.ndarray] = {
        "hour_of_day": (start_ts.hour.to_numpy() + start_ts.minute.to_numpy() / 60.0)
        / 24.0,
        "day_of_week": start_ts.dayofweek.to_numpy() / 7.0,
    }
    col = 0
    for feature in CYCLIC_FEATURES:
        normalised = feature_values[feature].astype(np.float64)
        for k in range(1, harmonics + 1):
            angle = 2.0 * np.pi * k * normalised
            out[:, col] = np.sin(angle).astype(_F32)
            out[:, col + 1] = np.cos(angle).astype(_F32)
            col += 2
    return out


def _segment_type_block(df: pd.DataFrame) -> np.ndarray:
    """One-hot encoding of segment_type in canonical order."""
    n = len(df)
    out = np.zeros((n, len(SEGMENT_TYPE_ORDER)), dtype=_F32)
    types = df["segment_type"].astype(str).to_numpy()
    for j, t in enumerate(SEGMENT_TYPE_ORDER):
        out[:, j] = (types == t).astype(_F32)
    return out


def _spatial_block(
    df: pd.DataFrame, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Lat/lon mapped to [0, 1] using the cohort bounding box.

    Out-of-bounds values are clipped (rather than raising) so a single
    out-of-bounds segment doesn't fail the whole batch; users who want
    strict-mode validation should validate against ``SegmentsSchema``
    upstream.
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    if lat_max <= lat_min or lon_max <= lon_min:
        msg = f"EmbedParams.spatial_bounds invalid: {bounds}"
        raise ValueError(msg)
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    def _norm(values: np.ndarray, lo: float, rng: float) -> np.ndarray:
        return np.clip((values.astype(np.float64) - lo) / rng, 0.0, 1.0)

    out = np.column_stack(
        [
            _norm(df["start_lat"].to_numpy(), lat_min, lat_range),
            _norm(df["start_lon"].to_numpy(), lon_min, lon_range),
            _norm(df["end_lat"].to_numpy(), lat_min, lat_range),
            _norm(df["end_lon"].to_numpy(), lon_min, lon_range),
        ]
    ).astype(_F32)
    return cast(np.ndarray, out)


# ── Plugin shape validation ─────────────────────────────────────────


def _validated_plugin_block(
    plugin: FeaturePlugin, df: pd.DataFrame
) -> np.ndarray:
    """Run a plugin and assert its output shape and dtype."""
    block = plugin.compute(df)
    expected_shape = (len(df), plugin.dim)
    if not isinstance(block, np.ndarray):
        msg = f"FeaturePlugin {plugin.name!r} returned {type(block).__name__}, expected ndarray"
        raise TypeError(msg)
    if block.shape != expected_shape:
        msg = (
            f"FeaturePlugin {plugin.name!r}: shape {block.shape}, "
            f"expected {expected_shape}"
        )
        raise ValueError(msg)
    if block.dtype != _F32:
        # Coerce silently for plugin authors who didn't explicitly set dtype.
        block = block.astype(_F32)
    return block


# ── L2 normalisation ────────────────────────────────────────────────


def _l2_normalize(vectors: np.ndarray, epsilon: float) -> np.ndarray:
    """Row-wise L2 normalisation with epsilon floor for zero-vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.maximum(norms, epsilon)
    result: np.ndarray = (vectors / safe).astype(_F32)
    return result
