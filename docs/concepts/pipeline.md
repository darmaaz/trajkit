# The Pipeline

The pipeline turns one entity's GPS pings into segment vectors that can be
indexed for similarity search.

```
pings → clean → segment → aggregate → episode
                       ↘
                         embed_segments → compare.build_index → search
```

## The single-entity contract

Each stage's L1 function takes one entity's frame in and returns one frame
out. No I/O, no global state, no implicit grouping. If you have many
entities, you iterate them yourself — `pandas.groupby` or a manual loop is
the entire story.

## Stage by stage

### `clean`

**Input:** raw pings (`PingsSchema`). **Output:** cleaned pings
(`CleanedPingsSchema`).

- Deduplicates exact-duplicate rows.
- Derives `dt_seconds`, `displacement_m`, `speed_ms`, `bearing_deg` from
  positions in a single vectorised geodesic call.
- Flags each ping with `quality_flag ∈ {VALID, DRIFT, SPEED_OUTLIER,
  GAP_FOLLOWS, DEVICE_FAULT}` using a fixed precedence
  (`DEVICE_FAULT > SPEED_OUTLIER > GAP_FOLLOWS > DRIFT > VALID`). The
  precedence rule is the design point — see [`clean`](../design/clean.md)
  for the rationale, especially why GAP_FOLLOWS outranks DRIFT.

### `segment`

**Input:** cleaned pings. **Output (after `aggregate_segments`):**
`SegmentsSchema` — one row per segment.

- A hysteresis state machine with two speed thresholds (`stop_speed_kmh`,
  `resume_speed_kmh`) classifies each ping as moving or stopped without
  flicker at the boundary.
- A circular-R bearing detector over distance-based sliding windows splits
  a `MOVE` segment where direction sustainedly changes. Multi-scale
  (short + long windows) catches both street corners and arterial sweeps;
  Schmitt-trigger hysteresis prevents flicker. See
  [`segment`](../design/segment.md) for the full method.
- Each segment is one of `MOVE`, `MOVE_BRIEF`, `STOP_BRIEF`, `STOP_DWELL`,
  classified by duration and ping-count thresholds.

### `episode`

**Input:** segments. **Output:** `EpisodesSchema` — one row per episode.

- A `STAY` is a maximal run of segments whose endpoints stay within radius
  `R_m` of the running anchor centroid, allowing a `T_s`-second grace
  window outside the envelope before closure.
- Anything between two stays is a `TRANSIT`. Transits split where the
  inter-segment gap exceeds `T_s`.
- Two qualification gates: `duration ≥ min_stay_s` AND
  `envelope_radius ≤ R_m`. The radius gate rejects spatially extended
  single-segment "stays" that the time gate would otherwise admit. See
  [`episode`](../design/episode.md).

### `embed_segments`

**Input:** segments. **Output:** `(vectors: ndarray, ids: list[str])`.

Per-segment recipe (32 dims with default `cyclic_harmonics=4`):

- **Kinematic block** (8 dims): `log1p` of `duration_s`, `path_length_m`,
  `displacement_m`, `mean_speed_ms`, `max_speed_ms`, `straightness`,
  `bearing_variance`, `n_pings`.
- **Cyclic block** (16 dims): sin/cos × 4 harmonics over hour-of-day and
  day-of-week from `start_ts`.
- **Segment-type block** (4 dims): one-hot of the four taxonomy values.
- **Spatial block** (4 dims): `start_lat`/`start_lon`/`end_lat`/`end_lon`
  normalised to `EmbedParams.spatial_bounds`.

Plugins extend the recipe by appending their own fixed-width block — see
the `FeaturePlugin` protocol in [`embed`](../design/embed.md).

### `compare`

**Input:** `(vectors, ids)`. **Output:** an `Index` you can `search`.

```python
from trajkit.compare import build_index, search

index = build_index(vectors, ids, metric="cosine")
hits  = search(index, vectors[0], k=10)
```

`build_index` wraps FAISS (`IndexFlatIP` for cosine, `IndexFlatL2` for L2).
`search` returns frozen `Hit` records — consumers join hits back to their
data on `id`.

## Composing many entities

The pipeline is per-entity. To process a fleet of N vehicles, iterate:

```python
import numpy as np
import pandas as pd
from trajkit.clean import clean
from trajkit.segment import segment, aggregate_segments
from trajkit.embed import embed_segments
from trajkit.compare import build_index

all_vectors: list[np.ndarray] = []
all_ids: list[str] = []
for entity_id, group in raw_pings.groupby("entity_id"):
    cleaned  = clean(group.sort_values("ts"))
    segs     = aggregate_segments(segment(cleaned))
    vectors, ids = embed_segments(segs)
    all_vectors.append(vectors)
    all_ids.extend(ids)

matrix = np.vstack(all_vectors)
index  = build_index(matrix, all_ids, metric="cosine")
```

Production-shape orchestration (parallel workers, atomic writes, resumable
runs) is out of scope for this repository.
