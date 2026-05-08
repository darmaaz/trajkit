"""``trajkit.segment`` — hysteresis state machine, 4-state taxonomy, aggregation.

Public API:

* ``segment(cleaned_pings_df, params)`` — adds ``segment_id`` and
  ``segment_type`` to a cleaned per-ping frame.
* ``aggregate_segments(segmented_pings_df, params)`` — collapses the
  per-ping frame into one row per segment.
* ``SegmentParams`` — frozen Pydantic v2 parameter model.

See ``docs/design/segment.md`` for the full specification.
"""

from __future__ import annotations

from trajkit.segment._aggregate import aggregate_segments
from trajkit.segment._params import SegmentParams
from trajkit.segment._segment import segment

__all__ = ["SegmentParams", "aggregate_segments", "segment"]
