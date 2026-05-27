# Architecture notes

Cross-cutting design discussion: how the modules fit together, what the
data contract is, and which choices were deliberate constraints rather
than accidental defaults. Per-module specifications live alongside this
file ([`clean`](clean.md), [`segment`](segment.md), [`episode`](episode.md),
[`embed`](embed.md), [`compare`](compare.md)).

## 1. What the pipeline does

Three operations, in order:

1. **Discretize** — raw pings → typed segments → episodes.
2. **Embed** — segments → fixed-width float32 vectors.
3. **Compare** — FAISS similarity search over the segment vectors.

## 2. Explicit non-goals

To prevent feature creep, the following are deliberately out of scope:

- Hosted UI / SaaS / web service.
- Streaming / online detection. Batch only.
- Geometry-heavy GIS (buffers, intersections, polygon ops) — use shapely
  or geopandas separately.
- ML model training. The pipeline emits vectors; downstream training is a
  user concern.
- Trajectory storage. The pipeline reads/writes parquet; it does not
  replace a database.
- Coordinate reference handling beyond WGS84 in / out. Users project
  before or after as needed.

## 3. Module shape

| Module | Responsibility | Explicitly NOT |
|---|---|---|
| `trajkit.clean` | Quality flags, drift detection, duplicate handling, gap flagging | OSM enrichment, projection conversion |
| `trajkit.segment` | Hysteresis state machine + 4-state taxonomy + bearing-split → segments | Per-segment feature engineering beyond the type label |
| `trajkit.episode` | Spatial-envelope episode detection (STAY/TRANSIT) | Place clustering across entities |
| `trajkit.embed` | Per-segment vectorisation, plugin extension point | Learned/contrastive embeddings |
| `trajkit.compare` | FAISS index construction + similarity search | Anomaly explanation, dashboards |

Plus:

- `trajkit.types` — Pandera + Arrow schemas, kept consistent by tests.
- `trajkit.testing` — synthetic trace generators. Used by the test suite
  and by example code.

## 4. The single-entity contract

The natural unit is one entity's pings. Operations on different entities
are independent, and a single entity's history fits in memory comfortably
even for huge fleets.

Each L1 function takes one entity's frame in and returns one frame out.
No global state, no I/O, no logging side effects. The function trusts the
schema; multi-entity inputs are a user error rather than a silent
groupby. This keeps composition explicit at the call site and avoids the
"silent groupby semantics change with parameter order" footgun.

Composing many entities is the user's job (`pandas.groupby` + a loop, or
your own orchestrator).

## 5. Data contract

The DataFrame column names are part of the public surface. Renames are
breaking. Schemas are codified twice — Pandera for runtime validation,
Arrow for parquet I/O — and a test asserts they agree. This avoids the
silent drift where "valid by Pandera" doesn't survive parquet
round-trip.

**Canonical input — pings frame:**

| col | dtype | required | notes |
|---|---|---|---|
| `entity_id` | str | yes | vehicle / vessel / animal / device id |
| `ts` | datetime64[ns, UTC] | yes | tz-aware, UTC |
| `lat`, `lon` | float64 | yes | WGS84 |
| `speed_ms` | float32 | no | derived if absent |
| `bearing_deg` | float32 | no | derived if absent |

Sorted by (`entity_id`, `ts`). Duplicate `(entity_id, ts)` rows raise.

Gotchas baked into the schema:

- Naive timestamps reject with a clear error. Mixed TZ is a footgun.
- `lat` / `lon` ranges validated. Catches the (lon, lat) reversal bug.
- Sort assertion is a check, not a side effect. Silent sort would break
  streaming-style chunked use.
- `entity_id` must be string. Numeric ids cause grief downstream (FAISS
  payload encoding, groupby performance).

## 6. Parameter contract

Each module exposes a frozen Pydantic v2 model with `extra='forbid'`.
Reasoning:

- **Frozen** — params are immutable in a run; mutation across a pipeline
  is a debugging nightmare we've all seen.
- **`extra='forbid'`** — silently ignored YAML typos are the worst class
  of bug. Make them errors.

```python
class EpisodeParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    R_m: PositiveFloat
    T_s: PositiveFloat
    min_stay_s: PositiveFloat
```

When loading from YAML, use `EpisodeParams.model_validate(yaml.safe_load(...))`
rather than passing a dict directly to a function — that bypasses
validation.

## 7. Domain extensibility

Three plug points, no others:

1. **Per-stage parameter overrides.** Every module's threshold is a
   `Params` field; users construct their own instance for their domain
   rather than relying on shipped defaults.
2. **Feature plugins** (embed only). A `FeaturePlugin` protocol contributes
   a fixed-width block to the segment vector. Users supply their own (e.g.
   a marine-traffic-zone block, an animal habitat-class block) by
   implementing `name`, `dim`, `compute(segments_df) -> np.ndarray`. The
   core recipe ships as a fixed bundle; user plugins compose alongside.
3. **Custom segmentation taxonomy** is *not* a plug point. The 4 states
   are fixed; adding more is a code-fork conversation.

Plugin contracts:

- Plugins declare their `dim` up front; blocks concat by index, so a
  variable-width plugin breaks the contract. Schema-checked at runtime.
- Plugins must be top-level importable (no lambdas) for any future
  multiprocessing path to work.

## 8. Dependency policy

Lean by default.

| Dep | Status | Reason |
|---|---|---|
| `pandas>=2.1` | required | core DataFrame |
| `numpy` | required | vectors |
| `pydantic>=2` | required | params + validation |
| `pandera` | required | schema validation |
| `pyarrow` | required | parquet round-trip |
| `h3>=4` | required | spatial indexing; pinned major to avoid the v3 → v4 API break |
| `pyproj` | required | great-circle / projection helpers |
| `faiss-cpu` | required | similarity search; fragile pip install on Apple Silicon — conda-forge is the reliable path |
| `geopandas` | **forbidden** | pulls GDAL; install hell on Windows and Apple Silicon |

Gotcha: `faiss-cpu` on Apple Silicon — works cleanly via conda-forge,
fragile via pip. Document the install path; don't pretend it's
transparent.

## 9. Performance posture

The L1 functions are vectorised pandas where possible, with a single
Python-level pass for state transitions in `segment`. No claimed
benchmarks ship with the repository — performance has not been measured
on a representative corpus, and the synthetic test suite is too small to
extrapolate from.

If you find a hot path on real data, profile first; the obvious targets
are the per-ping state-transition loop in `segment` and the
`iterrows`-shaped pooling in any custom feature plugin you write.

## 10. Testing posture

| Layer | Lives in | Speed |
|---|---|---|
| Unit | `tests/unit/` | <1s each, synthetic only |
| Integration | `tests/integration/` | <5s total, synthetic |
| Real-data example | `examples/geolife/` | minutes; Geolife `.plt` download required |

The synthetic generators in `trajkit.testing` are used by both the test
suite and the example notebooks. There is no real-data CI gate — the
example runs locally on demand.
