# `trajkit.compare`

## Purpose

Vector similarity search and persistence. Given the float32 vectors produced
by `embed`, this layer builds FAISS indices, performs top-k queries with
metadata filters, and persists indices to disk. It also exposes a
single-call anomaly score utility for users who want a quick density-based
ranking. Cohort-stable anomaly model fitting (which would require pass-2
state and persisted models) is deferred to v1.1.

## Assumptions

- Input vectors are `float32`, contiguous, and L2-normalized when the user
  is using cosine similarity. The library does not silently re-normalize —
  `build_index(..., metric="cosine", normalize="auto")` is explicit.
- Indices are in-memory at use time; persistence uses FAISS's native
  serialization (not pickle). Indices may be memory-mapped on load for
  large-corpus / low-memory consumers.
- For v1, only `IndexFlatIP` is exposed. Approximate indices (HNSW, IVF)
  have tuning surface and our scale (~1 M vectors per cohort) doesn't
  require them. Approximate indices are deferred to v2.
- Metadata filters are post-search (re-rank), not pre-filtered into
  sub-indices. We accept the small efficiency hit for a simpler API and
  fewer auxiliary indices to manage.
- The `Index` object is not used inside L1 functions. It lives in pass-2
  / single-process consumer code. This sidesteps FAISS's pickle hostility.

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
    filter: dict[str, set] | None = None,
) -> list[Hit]

save_index(index: Index, path: str | Path) -> None
load_index(path: str | Path, *, mmap: bool = False) -> Index

anomaly_score(
    vectors: np.ndarray,
    contamination: float = 0.01,
) -> np.ndarray
```

`Index` is a small Python wrapper around a FAISS index plus an `id`-to-row
mapping. It carries no DataFrame metadata; metadata for filters is supplied
at search time as `{column: allowed_values}`. The wrapper exposes only the
methods the library uses; it does not re-export FAISS internals.

`Hit` is a frozen dataclass: `id: str`, `score: float`, `rank: int`. It
carries no metadata payload — consumers join back to their segments /
episodes frame on `id`.

`anomaly_score` fits an `IsolationForest(n_estimators=200,
contamination=contamination)` in-call and returns `-decision_function`
scores so that more positive = more anomalous. This is deliberately a
one-call utility; cohort-stable scoring requires a fitted-and-persisted
model, which is `fit_anomaly_model` in v1.1.

## Efficiency

- `IndexFlatIP` query is O(N × D) per query. Target: < 50 ms over 1 M
  vectors, 80 dims, single thread.
- Build is O(N × D) memory copy into FAISS-managed buffer. No training.
- Persistence is FAISS native — fast read/write, version-stable across
  FAISS minor releases.
- `mmap=True` on load avoids materializing the full index for large
  corpora at the cost of slower per-query latency.
- Filter post-rerank is O(k) per result, negligible.
- `anomaly_score` is sklearn IsolationForest at the user's chosen scale;
  documented as O(N · n_estimators) build, O(n_estimators) per query.

## Usage

```python
import trajkit
from trajkit import compare

# Build + search
index = compare.build_index(segment_vectors, segment_ids, metric="cosine")
hits = compare.search(index, query_vec, k=10, filter={"segment_type": {"MOVE"}})

# Persist
compare.save_index(index, "out/segments.index")
loaded = compare.load_index("out/segments.index", mmap=True)

# Anomaly (per-call, not cohort-stable)
scores = compare.anomaly_score(segment_vectors, contamination=0.01)
```

## Successful deliverable

- [ ] `build_index`, `search`, `save_index`, `load_index`, `anomaly_score`
      with explicit signatures matching the architecture above.
- [ ] FAISS dependency declared as `[search]` extra (LIBRARY.md D5; verified
      via Apple Silicon + Linux install test before v1 release).
- [ ] `Hit` dataclass; metadata stays in the consumer's frame, joined on `id`.
- [ ] Save/load round-trip test with byte-equality on serialized form.
- [ ] Property test: top-1 hit for a query that exactly matches an indexed
      vector is that vector itself, with score ≈ 1.0 (cosine).
- [ ] Filter test: `filter` reduces returned hits to a strict subset.
- [ ] `anomaly_score` test on synthetic mixture (mostly normal + a few
      planted outliers) — outliers consistently rank in the top scores.
- [ ] ≥ 70% line coverage (FAISS itself is not re-tested here).

## Not in this layer

- Cohort-stable anomaly model fitting (`fit_anomaly_model`) — v1.1.
- Approximate indices (HNSW, IVF) — v2.
- Vector construction — `embed`.
- Cohort baselines — `embed.baseline_zscores` and pass-2 `fit_baselines`.
- Cross-entity pre-filtered indices — deferred; post-search filter is the
  v1 contract.
