# The Pipeline

A trajkit pipeline turns one entity's stream of GPS pings (or equivalent
spatial-temporal observations) into queryable behavioural artifacts. The
stages are linear and each one's output is the next one's input.

```
pings → clean → segment → aggregate → episode → embed_segments
                                              ↘
                                                embed_episodes
```

## The natural unit is one entity

Each layer operates on a single entity at a time — one vehicle, one
vessel, one animal, one phone. Operations on different entities are
independent, so the runner parallelises across entities trivially. The
library never holds the whole dataset in memory; it processes
per-entity, writes per-entity.

## Stage by stage

### `clean`

Input: raw pings (`PingsSchema`). Output: cleaned pings
(`CleanedPingsSchema`).

* Deduplicates exact-duplicate rows.
* Derives `dt_seconds`, `displacement_m`, `speed_ms`, `bearing_deg`
  from positions in a single vectorised geodesic call.
* Flags each ping with `quality_flag ∈ {VALID, DRIFT, SPEED_OUTLIER,
  GAP_FOLLOWS, DEVICE_FAULT}` using a fixed precedence.
* Optional `merge_stale_positions` collapses runs of identical (lat,
  lon) for GPS devices that ping more often than they update position.

### `segment`

Input: cleaned pings. Output (after `aggregate_segments`):
`SegmentsSchema` — one row per segment.

* A hysteresis state machine with two speed thresholds (`stop_speed`,
  `resume_speed`) classifies each ping as moving or stopped without
  flicker.
* Sustained bearing changes split a `MOVE` into multiple segments when
  the rolling-mean bearing delta exceeds `bearing_change_deg` for
  `bearing_sustain_s` continuously.
* Each segment is one of `MOVE`, `MOVE_BRIEF`, `STOP_BRIEF`,
  `STOP_DWELL`, classified by duration + ping-count thresholds.

### `episode`

Input: segments. Output: `EpisodesSchema` — one row per episode.

* Spatial-envelope closure rule: a `STAY` is a maximal run of segments
  whose endpoints stay within radius `R_m` of the running anchor
  centroid, allowing a `T_s`-second grace window outside the envelope
  before closure.
* Anything between two stays is a `TRANSIT`. Transits split where the
  inter-segment time gap exceeds `T_s`.
* Two qualification gates: `duration ≥ min_stay_s` AND
  `envelope_radius ≤ R_m`. The radius gate rejects spatially extended
  single-segment "stays" that the time gate would otherwise admit.

### `embed_segments`

Input: segments. Output: `(vectors: ndarray, ids: list[str])`.

Per-segment recipe (32 dims with default `cyclic_harmonics=4`):

* **Kinematic block** (8 dims): `log1p` of `duration_s`,
  `path_length_m`, `displacement_m`, `mean_speed_ms`, `max_speed_ms`,
  `straightness`, `bearing_variance`, `n_pings`.
* **Cyclic block** (16 dims): sin/cos × 4 harmonics over hour-of-day
  and day-of-week from `start_ts`.
* **Segment-type block** (4 dims): one-hot of the four taxonomy values.
* **Spatial block** (4 dims): `start_lat`/`start_lon`/`end_lat`/`end_lon`
  normalised to `EmbedParams.spatial_bounds`.

Plugins extend the base recipe by appending their own fixed-width block.
Output is L2-normalised by default (FAISS cosine prerequisite).

### `embed_episodes`

Input: episodes + segment vectors. Output: `(vectors, ids)`.

Pools each episode's constituent segment vectors via concatenation of
`(mean, std, max-by-magnitude)`, then appends five episode-level
scalars: `[log1p(duration_s), log1p(path_length_m), n_segments,
STAY-1hot, TRANSIT-1hot]`. Result dim is `3 × segment_dim + 5`,
L2-normalised.

## Orchestration: `process`

The L3 runner ties everything together:

```python
from trajkit.runner import process, RunParams
from trajkit.testing import make_pings

pings = make_pings(n=600, motion="stop_then_move")
report = process(pings, "out/", RunParams.from_preset("pedestrian"))
```

* Iterates entities via `iter_entities`.
* Applies stages in canonical order (clean → segment → episode →
  embed_segments → embed_episodes).
* Writes Hive-partitioned parquet per stage (atomic file-level rename).
* Resumes by skipping entities whose final stage output already exists.
* Multiprocessing pool over entities when given a parquet path source;
  in-memory sources force `n_workers=1`.

## Cohort baselines (pass-2)

Some statistics need a global view: cohort-relative anomaly scores
require per-cohort means and stddevs across all entities.

```python
from trajkit.baselines import fit_baselines
from trajkit.embed import baseline_zscores

# After process(...) has produced sink/segment/...
baselines = fit_baselines("sink/segment/", cohort_keys=["entity_id"])
zscored = baseline_zscores(segments_df, baselines, cohort_keys=["entity_id"])
```

Baselines are computed once across the cohort, then applied per-entity
as a separate L1 step. Cohorts with too few samples fall back to global
statistics with `is_fallback=True`.

## Compare: similarity + anomaly

```python
from trajkit.compare import build_index, search, anomaly_score

index = build_index(vectors, ids, metric="cosine")
hits = search(index, vectors[0], k=10)
scores = anomaly_score(vectors, contamination=0.01)
```

`build_index` wraps FAISS (`IndexFlatIP` for cosine, `IndexFlatL2` for
L2). `search` returns frozen `Hit` records — consumers join hits back
to their data on `id`. `anomaly_score` is a per-call IsolationForest
helper; cohort-stable model fitting lands in v1.1.
