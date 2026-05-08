# `trajkit.embed`

## Purpose

Convert per-segment and per-episode frames into fixed-width float32 vectors
that capture behavioral content — kinematic profile, cyclic temporal
position, sequence context, and (via plugins) domain-specific context like
POI category or marine zone. Vectors are the substrate for `compare`'s
similarity search and downstream clustering. This layer also pools segment
vectors into episode vectors and applies cohort baselines computed in
pass-2.

## Assumptions

- Input segments / episodes match their respective schemas in `schemas.md`.
- Segments are independent at L1 — no cross-entity context required for the
  base recipe.
- The base vector blocks are domain-agnostic: kinematic, cyclic-temporal,
  segment-type one-hot, normalized lat/lon. Domain-specific context comes
  from feature plugins supplied by the user.
- Plugins are pure callables with a fixed output shape declared in advance.
  Variable-width output breaks the concat contract and is rejected.
- Vector dimension is determined by `EmbedParams` plus the plugins passed in,
  not by data inspection. `expected_dim()` is callable before any data flows.

## Architecture

```python
embed_segments(
    segments_df: pd.DataFrame,
    params: EmbedParams,
    features: Sequence[FeaturePlugin] = (),
) -> tuple[np.ndarray, list[str]]                 # (vectors, segment_ids)

embed_episodes(
    episodes_df: pd.DataFrame,
    segment_vectors: np.ndarray,
    segment_ids: list[str],
) -> tuple[np.ndarray, list[str]]                 # (vectors, episode_ids)

baseline_zscores(
    segments_df: pd.DataFrame,
    baselines: Baselines,
    cohort_keys: list[str],
) -> pd.DataFrame                                  # adds *_z columns
```

Base recipe blocks (always emitted):
- **Kinematic** (linear): log1p+StandardScaler over duration, distance, mean
  speed, max speed, path length, displacement, straightness, bearing variance.
- **Cyclic** (sinusoidal): sin/cos × K harmonics over hour-of-day,
  day-of-week, mean bearing.
- **Segment type** (one-hot): 4 dims.
- **Spatial position** (normalized): lat/lon mapped to a [0, 1] bounding
  box derived from the cohort, shipped in `EmbedParams.spatial_bounds`.

Plugin contract (`Protocol`):
```python
class FeaturePlugin(Protocol):
    name: str
    dim: int
    def compute(self, segments_df: pd.DataFrame) -> np.ndarray: ...
```
Output must be `shape == (len(segments_df), self.dim)`, dtype `float32`.
Validated at every call (cheap shape check). Plugins are picklable so they
work under multiprocessing — lambda plugins fail explicitly at registration,
not silently in a worker.

Episode pooling: for each episode, gather its segment vectors and concat
[mean, std, max-by-magnitude] over them, then append episode-level scalars
[log1p(duration_s), log1p(path_length_m), n_segments, episode_type one-hot
(2 dims)]. Output L2-normalized.

Baseline application: `baseline_zscores` looks up each segment's cohort key,
computes (value − baseline.mean) / max(baseline.std, ε), with sample-count
fallback to a parent cohort when `n_samples < params.min_cohort_n`. Returns
the segments frame with `*_z` columns added; this is then passed back into
`embed_segments` as if those columns were base inputs (registered as a
synthetic block).

## Efficiency

- Base recipe is fully vectorized numpy. Target: 50 K segments/sec/entity
  per plugin block on the developer laptop.
- Plugins inherit responsibility for their own efficiency. The library
  validates the shape contract; the plugin author must ensure linearity.
  A spatial-join plugin without an h3 hash will dominate runtime — this
  is documented loudly in the `FeaturePlugin` Protocol docstring.
- Episode pooling is O(N_segments × D); negligible compared to embed.
- Output buffers are preallocated `(N, total_dim) float32` once, plugin
  blocks fill into slices. No intermediate concatenation, no Python lists
  of arrays.
- Memory: `(N_segments × total_dim × 4)` bytes. For 1 M segments × 80 dims
  = 320 MB per entity — fits.
- L2-normalize once at the end for cosine compatibility; FAISS expects this.

## Usage

```python
import trajkit

# Base recipe, no plugins
vectors, ids = trajkit.embed_segments(segments_df, trajkit.EmbedParams())

# With plugins (user-provided, bring-your-own-data)
vectors, ids = trajkit.embed_segments(
    segments_df,
    trajkit.EmbedParams(),
    features=[POIPlugin(poi_table), RoadClassPlugin(road_table)],
)

# Episodes
ep_vectors, ep_ids = trajkit.embed_episodes(episodes_df, vectors, ids)

# Cohort baselines (after pass-2 fit_baselines has produced Baselines)
seg_with_z = trajkit.baseline_zscores(segments_df, baselines, cohort_keys=["entity_id"])
vectors, ids = trajkit.embed_segments(seg_with_z, trajkit.EmbedParams(include_baselines=True))
```

## Successful deliverable

- [ ] `embed_segments` — base recipe, with at least the four base blocks.
      Output is `np.ndarray` of dtype `float32`, contiguous, FAISS-ready.
- [ ] `embed_episodes` — pooling + episode-level scalars + L2-normalize.
- [ ] `baseline_zscores` — cohort lookup with sample-count fallback.
- [ ] `FeaturePlugin` Protocol with shape-contract validation, and one
      reference plugin (in `examples/`, not bundled into `trajkit.embed`)
      to demonstrate the integration.
- [ ] `EmbedParams` model with `expected_dim(features=[...]) -> int`
      callable before data flows.
- [ ] Picklability test: every Params and every shipped reference plugin
      survives `pickle.dumps`/`loads`.
- [ ] Property test: vectors are unit-norm to within float32 epsilon when
      `params.l2_normalize=True`.
- [ ] ≥ 80% line coverage.

## Not in this layer

- Computing baselines — pass-2 `fit_baselines`.
- FAISS index — `compare`.
- Learned / contrastive embedding training — out of v1 entirely; the
  existing `methods/contrastive` is exploration scaffolding, not core.
- Persistence of vectors — L3 runner; vectors written as fixed-size list
  parquet columns per `schemas.md`.
