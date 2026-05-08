"""``trajkit.io`` — L2 entity iterator and parquet read/write helpers.

Public API:

* ``iter_entities(source, *, format)`` — yields ``(entity_id, pings_df)``
  tuples from parquet, Arrow, DataFrame, or CSV sources. Validates each
  yielded frame against ``PingsSchema``.

See ``docs/design/LIBRARY.md`` §5 (L2 layer) and §7 (I/O conventions).
"""

from __future__ import annotations

from trajkit.io._iter import iter_entities

__all__ = ["iter_entities"]
