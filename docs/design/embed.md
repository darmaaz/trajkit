# `trajkit.embed`

## Purpose

Convert a per-segment frame into a fixed-width `float32` vector that
captures behavioural content — kinematic profile, cyclic temporal
position, segment-type, and normalised spatial position. The vector is
the substrate for [`compare`](compare.md)'s similarity search; users
extend it via the `FeaturePlugin` protocol for domain-specific signals
(POI category, marine zone, road class, etc.).

## Assumptions

- Input matches `SegmentsSchema` (see [`schemas.md`](schemas.md)).
- The base blocks are domain-agnostic. Domain context comes from user-
  supplied plugins.
- Plugins have a fixed output `dim` declared up front. Variable-width
  output breaks the concat contract and is rejected at runtime.
- Vector dimension is determined by `EmbedParams` plus the plugins
  passed in. `EmbedParams.expected_dim(features)` is callable before
  any data flows.

## Architecture

```python
embed_segments(
    segments_df: pd.DataFrame,
    params: EmbedParams,
    features: tuple[FeaturePlugin, ...] = (),
) -> tuple[np.ndarray, list[str]]   # (vectors, segment_ids)
```

### Base recipe blocks (always emitted)

- **Kinematic** (8 dims): `log1p` of `duration_s`, `path_length_m`,
  `displacement_m`, `mean_speed_ms`, `max_speed_ms`, `straightness`,
  `bearing_variance`, `n_pings`.
- **Cyclic** (8 dims at default `cyclic_harmonics=2`; 16 at default
  `cyclic_harmonics=4`): sin/cos × K harmonics over hour-of-day and
  day-of-week.
- **Segment type** (4 dims): one-hot of `MOVE / MOVE_BRIEF / STOP_BRIEF
  / STOP_DWELL`.
- **Spatial position** (4 dims): `start_lat`, `start_lon`, `end_lat`,
  `end_lon` normalised to `EmbedParams.spatial_bounds`.

### Plugin contract

```python
class FeaturePlugin(Protocol):
    name: str
    dim: int
    def compute(self, segments_df: pd.DataFrame) -> np.ndarray: ...
```

Output must be `shape == (len(segments_df), self.dim)`, dtype
`float32`. Validated at every call. Plugins must be top-level
importable (no lambdas) so they survive process boundaries.

The full vector is the concat of base blocks + each plugin's block,
L2-normalised by default for cosine compatibility with FAISS.

## Efficiency

- Base recipe is fully vectorised numpy.
- Plugins inherit responsibility for their own efficiency. The library
  validates the shape contract; the plugin author must ensure
  linearity. A spatial-join plugin without an h3 hash will dominate
  runtime.
- Output is a single preallocated `(N, total_dim)` float32 buffer.
- L2-normalisation runs once at the end.

## Usage

```python
from trajkit.embed import embed_segments, EmbedParams

# Base recipe, no plugins
vectors, ids = embed_segments(segments_df, EmbedParams())

# With user-supplied plugins
vectors, ids = embed_segments(
    segments_df,
    EmbedParams(),
    features=(POIPlugin(poi_table), RoadClassPlugin(road_table)),
)
```

## Deliverable

- [x] `embed_segments` — base recipe with four blocks. Output is a
      contiguous `float32` array, FAISS-ready.
- [x] `FeaturePlugin` Protocol with shape-contract validation at every
      call.
- [x] `EmbedParams` with `expected_dim(features)` callable before data
      flows.
- [x] Property test: vectors are unit-norm to within float32 epsilon
      when `params.l2_normalize=True`.

## Not in this layer

- FAISS index — [`compare`](compare.md).
- Learned / contrastive embedding training — out of scope.
- Episode-level pooling — user concern; the per-segment vector is the
  primitive, and episodes carry segment ids that join back to vectors
  for downstream pooling at the user's chosen scheme.
