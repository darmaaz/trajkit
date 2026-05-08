"""``trajkit.embed`` — base recipe, plugins, episode pooling, baselines.

Public API:

* ``embed_segments(segments_df, params, features)`` — float32 vectors per
  segment from the base recipe + optional plugins.
* ``embed_episodes(episodes_df, segment_vectors, segment_ids, params)`` —
  pool segment vectors and append episode-level scalars.
* ``baseline_zscores(segments_df, baselines, cohort_keys, epsilon)`` —
  apply pass-2 baselines as ``<metric>_z`` columns.
* ``EmbedParams`` — frozen Pydantic v2 hyperparameters.
* ``FeaturePlugin`` — protocol for user-supplied embedding extensions.

See ``docs/design/embed.md`` for the full specification.
"""

from __future__ import annotations

from trajkit.embed._baselines import baseline_zscores
from trajkit.embed._episodes import embed_episodes
from trajkit.embed._params import EmbedParams, FeaturePlugin
from trajkit.embed._segments import embed_segments

__all__ = [
    "EmbedParams",
    "FeaturePlugin",
    "baseline_zscores",
    "embed_episodes",
    "embed_segments",
]
