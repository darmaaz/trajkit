"""``trajkit.clean`` — quality flags, dedup, stale-position merge.

Public API:

* ``clean(pings_df, params)`` — per-ping quality flagging and kinematic
  derivation. Single-entity input.
* ``merge_stale_positions(cleaned_df, params, clean_params)`` — collapse
  duplicate-position runs; opt-in for entities whose GPS device pings
  more often than it updates position.
* ``detect_stale_pattern(pings_df, params)`` — returns True iff the
  entity exhibits the stale-position pattern.
* ``CleanParams``, ``StaleMergeParams`` — frozen Pydantic v2 parameter
  models.

See ``docs/design/clean.md`` for the full specification and rationale.
"""

from __future__ import annotations

from trajkit.clean._clean import clean
from trajkit.clean._params import CleanParams, StaleMergeParams
from trajkit.clean._stale import detect_stale_pattern, merge_stale_positions

__all__ = [
    "CleanParams",
    "StaleMergeParams",
    "clean",
    "detect_stale_pattern",
    "merge_stale_positions",
]
