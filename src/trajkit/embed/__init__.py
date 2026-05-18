"""``trajkit.embed`` — fixed-width float32 vectors per segment.

Public API:

* ``embed_segments(segments_df, params, features)`` — base recipe plus
  optional plugin blocks.
* ``EmbedParams`` — frozen hyperparameters.
* ``FeaturePlugin`` — protocol for user-supplied embedding extensions.

See ``docs/design/embed.md`` for the recipe details and plugin contract.
"""

from __future__ import annotations

from trajkit.embed._params import EmbedParams, FeaturePlugin
from trajkit.embed._segments import embed_segments

__all__ = ["EmbedParams", "FeaturePlugin", "embed_segments"]
