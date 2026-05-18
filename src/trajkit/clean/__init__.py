"""``trajkit.clean`` — kinematic derivation and quality flagging.

Public API:

* ``clean(pings_df, params)`` — per-ping quality flagging and derived
  kinematic columns. Single-entity input.
* ``CleanParams`` — frozen Pydantic v2 parameter model.

See ``docs/design/clean.md`` for the precedence rule and per-flag
rationale.
"""

from __future__ import annotations

from trajkit.clean._clean import clean
from trajkit.clean._params import CleanParams

__all__ = ["CleanParams", "clean"]
