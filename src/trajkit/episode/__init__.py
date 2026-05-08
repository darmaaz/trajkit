"""``trajkit.episode`` — spatial-envelope STAY/TRANSIT closure rule.

Public API:

* ``detect_episodes(segments_df, params)`` — per-entity episode detection.
* ``EpisodeParams`` — frozen Pydantic v2 parameter model.

See ``docs/design/episode.md`` for the full specification.
"""

from __future__ import annotations

from trajkit.episode._detect import detect_episodes
from trajkit.episode._params import EpisodeParams

__all__ = ["EpisodeParams", "detect_episodes"]
