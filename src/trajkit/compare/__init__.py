"""``trajkit.compare`` — FAISS index, similarity search, persistence, anomaly.

Public API:

* ``build_index(vectors, ids, metric, normalize)`` — build a FAISS-backed
  similarity index.
* ``search(index, query, k, filter_ids)`` — top-k nearest neighbours.
* ``save_index(index, path)`` / ``load_index(path, mmap)`` — persist and
  reload via FAISS native serialisation.
* ``anomaly_score(vectors, contamination)`` — per-call IsolationForest.
* ``Index``, ``Hit`` — public types.

FAISS itself is imported lazily so ``import trajkit.compare`` works
without the ``[search]`` extra installed; functions raise a clear
``ImportError`` only when actually invoked. Install via
``pip install 'trajkit[search]'``.

See ``docs/design/compare.md`` for the full specification.
"""

from __future__ import annotations

from trajkit.compare._anomaly import anomaly_score
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
    "anomaly_score",
    "build_index",
    "load_index",
    "save_index",
    "search",
]
