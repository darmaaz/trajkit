"""FAISS-backed similarity index, search, and persistence.

Exposes ``IndexFlatIP`` (cosine) and ``IndexFlatL2`` via ``build_index``,
top-k ``search``, and FAISS-native ``save_index`` / ``load_index``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np

_F32 = np.float32
_VALID_METRICS = ("cosine", "l2")
_INDEX_FILENAME = "index.faiss"
_META_FILENAME = "meta.json"


@dataclass(frozen=True)
class Hit:
    """A single similarity-search result.

    ``score`` is the inner product for cosine (higher = more similar) or
    the negative L2 distance, depending on the index's metric.
    """

    id: str
    score: float
    rank: int


class Index:
    """Wraps a FAISS index plus an id mapping and metric tag.

    Not picklable: FAISS indexes use their own serialisation. Use
    ``save_index`` and ``load_index`` for persistence.
    """

    def __init__(self, faiss_index: Any, ids: list[str], metric: str) -> None:
        self._faiss = faiss_index
        self._ids: list[str] = list(ids)
        self._metric = metric

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def dim(self) -> int:
        return int(self._faiss.d)

    def __len__(self) -> int:
        return int(self._faiss.ntotal)


# ── Public functions ────────────────────────────────────────────────


def build_index(
    vectors: np.ndarray,
    ids: list[str],
    *,
    metric: Literal["cosine", "l2"] = "cosine",
    normalize: Literal["auto", "always", "never"] = "auto",
) -> Index:
    """Build an ``Index`` over ``vectors`` keyed by ``ids``.

    ``normalize="auto"`` row-normalises when cosine is requested and rows
    aren't already unit-norm; ``"always"`` forces it; ``"never"`` skips
    it even for cosine.
    """
    if metric not in _VALID_METRICS:
        msg = f"unknown metric {metric!r}; valid options: {_VALID_METRICS}"
        raise ValueError(msg)

    if vectors.ndim != 2:
        msg = f"vectors must be 2-D, got shape {vectors.shape}"
        raise ValueError(msg)
    if vectors.shape[0] != len(ids):
        msg = (
            f"vectors rows ({vectors.shape[0]}) and ids length ({len(ids)}) disagree"
        )
        raise ValueError(msg)

    arr = np.ascontiguousarray(vectors, dtype=_F32)

    if (
        metric == "cosine"
        and normalize != "never"
        and (normalize == "always" or not _is_unit_norm(arr))
    ):
        arr = _normalize_rows(arr)

    # Annotate to the base class — `IndexFlatIP` and `IndexFlatL2` are
    # distinct types in stricter faiss stubs (e.g. the Linux wheel), so
    # an inferred type from the first branch would reject the second.
    faiss_index: faiss.Index
    if metric == "cosine":
        faiss_index = faiss.IndexFlatIP(arr.shape[1])
    else:
        faiss_index = faiss.IndexFlatL2(arr.shape[1])
    faiss_index.add(arr)

    return Index(faiss_index, list(ids), metric)


def search(
    index: Index,
    query: np.ndarray,
    k: int = 10,
    filter_ids: frozenset[str] | None = None,
) -> list[Hit]:
    """Top-``k`` nearest neighbours for a single query vector.

    ``filter_ids`` is applied as a post-search rerank — fewer than ``k``
    hits may be returned if the filtered pool is small.
    """
    if query.ndim == 1:
        query_arr = query.reshape(1, -1)
    elif query.ndim == 2 and query.shape[0] == 1:
        query_arr = query
    else:
        msg = f"query must be 1-D (D,) or 2-D (1, D); got {query.shape}"
        raise ValueError(msg)
    if query_arr.shape[1] != index.dim:
        msg = (
            f"query dim {query_arr.shape[1]} doesn't match index dim {index.dim}"
        )
        raise ValueError(msg)

    query_arr = np.ascontiguousarray(query_arr, dtype=_F32)
    # Normalise the query to match the (build-time) row-normalised index
    # so the inner product is true cosine similarity.
    if index.metric == "cosine":
        query_arr = _normalize_rows(query_arr)

    overshoot = min(len(index), k * 4 if filter_ids else k)
    if overshoot == 0:
        return []
    distances, indices = index._faiss.search(query_arr, overshoot)

    hits: list[Hit] = []
    rank = 0
    for raw_score, raw_idx in zip(distances[0], indices[0], strict=True):
        if raw_idx < 0:
            continue
        sid = index._ids[int(raw_idx)]
        if filter_ids is not None and sid not in filter_ids:
            continue
        score = (
            float(raw_score)
            if index.metric == "cosine"
            else float(-raw_score)  # L2: lower distance = better; flip sign
        )
        hits.append(Hit(id=sid, score=score, rank=rank))
        rank += 1
        if len(hits) >= k:
            break
    return hits


def save_index(index: Index, path: str | Path) -> None:
    """Persist ``index`` to a directory containing the FAISS file + meta."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index._faiss, str(target / _INDEX_FILENAME))
    meta = {"metric": index.metric, "ids": index._ids}
    (target / _META_FILENAME).write_text(json.dumps(meta))


def load_index(path: str | Path, *, mmap: bool = False) -> Index:
    """Load an ``Index`` previously saved with ``save_index``."""
    src = Path(path)
    if not src.exists():
        msg = f"index path does not exist: {src}"
        raise FileNotFoundError(msg)
    flag = faiss.IO_FLAG_MMAP if mmap else 0
    faiss_index = faiss.read_index(str(src / _INDEX_FILENAME), flag)
    meta = json.loads((src / _META_FILENAME).read_text())
    return Index(faiss_index, meta["ids"], meta["metric"])


# ── Helpers ─────────────────────────────────────────────────────────


def _is_unit_norm(vectors: np.ndarray, tol: float = 1e-3) -> bool:
    """Cheap check: are rows already L2-normalised?"""
    if vectors.shape[0] == 0:
        return True
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows preserved."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-8)
    result: np.ndarray = (vectors / safe).astype(_F32)
    return result
