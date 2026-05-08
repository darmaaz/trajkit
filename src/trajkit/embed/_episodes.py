"""Per-episode embedding: pool segment vectors + episode-level scalars.

Implements ``embed_episodes``. For each episode, the constituent segment
vectors are pooled to a fixed width via concatenation of (mean, std,
max-by-magnitude). Five episode-level scalars are appended:
``[log1p(duration_s), log1p(path_length_m), n_segments, STAY-1hot,
TRANSIT-1hot]``. The result is L2-normalised.

Output dim is ``3 × segment_dim + 5`` per the design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trajkit.embed._params import EmbedParams
from trajkit.embed._segments import _l2_normalize

_F32 = np.float32
_EPISODE_SCALAR_DIM = 5  # log1p(duration) + log1p(path) + n_segments + 2-hot


def embed_episodes(
    episodes_df: pd.DataFrame,
    segment_vectors: np.ndarray,
    segment_ids: list[str],
    params: EmbedParams | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Pool segment vectors into episode vectors.

    Parameters
    ----------
    episodes_df
        Output of ``trajkit.episode.detect_episodes``.
    segment_vectors
        ``(n_segments, segment_dim)`` float32 array from ``embed_segments``.
    segment_ids
        Row-aligned segment IDs for ``segment_vectors``.
    params
        Used only for ``l2_normalize`` and ``epsilon`` here.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(vectors, episode_ids)``: ``vectors`` is float32 with shape
        ``(n_episodes_with_pool, 3 * segment_dim + 5)``. Episodes whose
        ``segment_ids`` produce no overlap with ``segment_ids`` are dropped
        from the output.
    """
    p = params if params is not None else EmbedParams()

    if segment_vectors.ndim != 2:
        msg = f"segment_vectors must be 2-D, got shape {segment_vectors.shape}"
        raise ValueError(msg)
    if segment_vectors.shape[0] != len(segment_ids):
        msg = (
            f"segment_vectors rows ({segment_vectors.shape[0]}) and "
            f"segment_ids length ({len(segment_ids)}) disagree"
        )
        raise ValueError(msg)

    segment_dim = segment_vectors.shape[1]
    output_dim = 3 * segment_dim + _EPISODE_SCALAR_DIM

    if len(episodes_df) == 0:
        return np.zeros((0, output_dim), dtype=_F32), []

    id_to_row = {sid: i for i, sid in enumerate(segment_ids)}

    out_vectors: list[np.ndarray] = []
    out_ids: list[str] = []

    for _, episode in episodes_df.iterrows():
        ids = list(episode["segment_ids"])
        rows = [id_to_row[sid] for sid in ids if sid in id_to_row]
        if not rows:
            continue

        sub = segment_vectors[rows]
        pooled = _pool(sub)
        scalars = _episode_scalars(episode)

        full = np.concatenate([pooled, scalars]).astype(_F32)
        out_vectors.append(full)
        out_ids.append(str(episode["episode_id"]))

    if not out_vectors:
        return np.zeros((0, output_dim), dtype=_F32), []

    matrix = np.vstack(out_vectors).astype(_F32)
    if p.l2_normalize:
        matrix = _l2_normalize(matrix, p.epsilon)

    return np.ascontiguousarray(matrix, dtype=_F32), out_ids


# ── Pooling ─────────────────────────────────────────────────────────


def _pool(sub_vectors: np.ndarray) -> np.ndarray:
    """Concatenate (mean, std, max-by-magnitude) along the segment axis."""
    if sub_vectors.shape[0] == 1:
        # Single-segment episode: std is 0, max-by-mag equals the only row.
        only = sub_vectors[0]
        return np.concatenate(
            [only, np.zeros_like(only), only], dtype=_F32
        )
    mean = sub_vectors.mean(axis=0)
    std = sub_vectors.std(axis=0)
    abs_vals = np.abs(sub_vectors)
    argmax = abs_vals.argmax(axis=0)
    cols = np.arange(sub_vectors.shape[1])
    max_by_mag = sub_vectors[argmax, cols]
    return np.concatenate([mean, std, max_by_mag], dtype=_F32)


def _episode_scalars(episode: pd.Series) -> np.ndarray:
    """Five episode-level scalar features, dtype float32."""
    duration_s = float(episode["duration_s"])
    raw_path = episode.get("path_length_m")
    path_length_m = (
        float(raw_path) if raw_path is not None and not pd.isna(raw_path) else 0.0
    )
    n_segments = float(episode["n_segments"])
    episode_type = str(episode["episode_type"])
    is_stay = 1.0 if episode_type == "STAY" else 0.0
    is_transit = 1.0 if episode_type == "TRANSIT" else 0.0
    return np.array(
        [
            np.log1p(max(duration_s, 0.0)),
            np.log1p(max(path_length_m, 0.0)),
            n_segments,
            is_stay,
            is_transit,
        ],
        dtype=_F32,
    )
