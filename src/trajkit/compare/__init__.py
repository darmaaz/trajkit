"""``trajkit.compare`` — FAISS-backed similarity index and search.

Public API:

* ``build_index(vectors, ids, metric, normalize)`` — build a FAISS index.
* ``search(index, query, k, filter_ids)`` — top-k nearest neighbours.
* ``save_index(index, path)`` / ``load_index(path, mmap)`` — persistence.
* ``Index``, ``Hit`` — public types.

FAISS is imported lazily so ``import trajkit.compare`` works without
FAISS installed; functions raise ``ImportError`` only when invoked.

See ``docs/design/compare.md`` for the metric and normalisation details.
"""

from __future__ import annotations

from trajkit.compare._index import (
    Hit,
    Index,
    build_index,
    load_index,
    save_index,
    search,
)

__all__ = [
    "Hit",
    "Index",
    "build_index",
    "load_index",
    "save_index",
    "search",
]
