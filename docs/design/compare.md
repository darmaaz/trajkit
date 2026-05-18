# `trajkit.compare`

## Purpose

Vector similarity search and persistence. Given the float32 vectors
produced by `embed`, this layer builds FAISS indices, performs top-k
queries with optional id filters, and persists indices to disk.

## Assumptions

- Input vectors are `float32`, contiguous. L2-normalization for cosine
  is auto-applied by `build_index(..., normalize="auto")` when the rows
  aren't already unit-norm; the user can force the choice with
  `normalize="always"` or opt out with `normalize="never"`.
- Indices are in-memory at use time; persistence uses FAISS's native
  serialization (not pickle). Indices may be memory-mapped on load for
  large-corpus / low-memory consumers.
- Only `IndexFlatIP` (cosine) and `IndexFlatL2` are exposed. Approximate
  indices (HNSW, IVF) have tuning surface that isn't worth the cost at
  the scales the rest of the pipeline produces.
- Id filtering is post-search rerank, not pre-filtered into sub-indices.
  Small efficiency hit; simpler API.

## Architecture

```python
build_index(
    vectors: np.ndarray,
    ids: list[str],
    *,
    metric: Literal["cosine", "l2"] = "cosine",
    normalize: Literal["auto", "always", "never"] = "auto",
) -> Index

search(
    index: Index,
    query: np.ndarray,
    k: int = 10,
    filter_ids: frozenset[str] | None = None,
) -> list[Hit]

save_index(index: Index, path: str | Path) -> None
load_index(path: str | Path, *, mmap: bool = False) -> Index
```

`Index` is a small Python wrapper around a FAISS index plus an
`id`-to-row mapping. It carries no DataFrame metadata; consumers join
hits back on `id`.

`Hit` is a frozen dataclass: `id: str`, `score: float`, `rank: int`.

## Efficiency

- `IndexFlatIP` query is O(N × D) per query.
- Build is O(N × D) memory copy into FAISS-managed buffer. No training.
- Persistence is FAISS native — fast read/write, version-stable across
  FAISS minor releases.
- `mmap=True` on load avoids materialising the full index for large
  corpora at the cost of slower per-query latency.

## Usage

```python
from trajkit.compare import build_index, save_index, load_index, search

index = build_index(segment_vectors, segment_ids, metric="cosine")
hits  = search(index, query_vec, k=10)

save_index(index, "out/segments.index")
loaded = load_index("out/segments.index", mmap=True)
```

## Not in this layer

- Vector construction — [`embed`](embed.md).
- Approximate indices (HNSW, IVF), cohort-stable anomaly models —
  out of scope.
