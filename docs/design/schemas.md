# Schemas

Single source of truth for the public DataFrame schemas. Every other module
document refers back to this one.

Each schema is declared **twice** in `trajkit.types`: once as a Pandera
`DataFrameSchema` for runtime validation, once as a `pyarrow.Schema` for
storage round-trip. A test asserts the two declarations are consistent
(column names, dtypes, nullability). Validation runs at the L2 boundary
on read, and at pass-2 boundaries on read; L1 pure functions trust the
contract for speed.

## Versioning

Each schema carries a `schema_version: str` carried as Pandera metadata and
written as parquet key-value metadata under `trajkit.schema_version`. Readers
tolerate missing-with-default for columns added in later versions. v0.1.0
schemas start at version `1`.

---

## Pings — input

Output of `iter_entities`. Single-entity frames sorted by `ts`.

| col | dtype | nullable | constraint | notes |
|---|---|---|---|---|
| `entity_id` | `string` | no | unique value per frame | string only — numeric ids cause grief |
| `ts` | `datetime64[ns, UTC]` | no | tz-aware UTC, monotonic non-decreasing | naive ts rejected at L2 |
| `lat` | `float64` | no | [-90, 90] | range-checked |
| `lon` | `float64` | no | [-180, 180] | range-checked — catches reversed (lon, lat) |
| `speed_ms` | `float32` | yes | ≥ 0 | derived in `clean` if absent |
| `bearing_deg` | `float32` | yes | [0, 360) | derived in `clean` if absent |

## Pings (cleaned) — output of `clean`

Same as Pings, plus:

| col | dtype | nullable | constraint | notes |
|---|---|---|---|---|
| `dt_seconds` | `float32` | yes | ≥ 0 | gap to previous ping |
| `displacement_m` | `float32` | yes | ≥ 0 | great-circle from previous ping |
| `is_duplicate` | `bool` | no | | identical (lat, lon) to previous |
| `quality_flag` | `string` | no | enum below | dominant flag for the row |
| `merge_count` | `int32` | yes | ≥ 1 | only when stale-merge ran |
| `run_duration_s` | `float32` | yes | ≥ 0 | only when stale-merge ran |

`quality_flag` enum: `VALID`, `DRIFT`, `SPEED_OUTLIER`, `GAP_FOLLOWS`, `DEVICE_FAULT`.

## Pings (segmented) — output of `segment`

Cleaned pings plus:

| col | dtype | nullable | constraint | notes |
|---|---|---|---|---|
| `segment_id` | `string` | no | format `<entity_id>_seg_<NNNNN>` | monotonic per entity |
| `segment_type` | `string` | no | enum below | one value per consecutive run |

`segment_type` enum: `MOVE`, `MOVE_BRIEF`, `STOP_BRIEF`, `STOP_DWELL`. Fixed in v1.

## Segments — output of `aggregate_segments`

One row per `segment_id`. The canonical input to `episode` and `embed`.

| col | dtype | nullable | notes |
|---|---|---|---|
| `segment_id` | `string` | no | primary key |
| `entity_id` | `string` | no | |
| `segment_type` | `string` | no | enum |
| `start_ts`, `end_ts` | `datetime64[ns, UTC]` | no | |
| `duration_s` | `float32` | no | > 0 |
| `start_lat`, `start_lon`, `end_lat`, `end_lon` | `float64` | no | |
| `start_h3`, `end_h3` | `string` | no | h3 v4 cell, resolution per `EmbedParams` |
| `path_length_m` | `float32` | no | sum of inter-ping displacements |
| `displacement_m` | `float32` | no | great-circle start→end |
| `straightness` | `float32` | no | displacement / path_length, clipped [0, 1] |
| `mean_speed_ms`, `max_speed_ms` | `float32` | yes | null only for STOP types |
| `bearing_variance` | `float32` | yes | circular variance, [0, 1] |
| `n_pings` | `int32` | no | ≥ 1 |

## Episodes — output of `detect_episodes`

One row per `episode_id`. STAY and TRANSIT episodes share columns; type-specific
columns are nullable on the off-type. See `episode.md` for the closure rule.

| col | dtype | nullable | type | notes |
|---|---|---|---|---|
| `episode_id` | `string` | no | both | format `ep_<entity_id>_<NNNNN>` |
| `entity_id` | `string` | no | both | |
| `episode_type` | `string` | no | both | enum: `STAY`, `TRANSIT` |
| `start_ts`, `end_ts` | `datetime64[ns, UTC]` | no | both | |
| `duration_s` | `float32` | no | both | |
| `segment_ids` | `list<string>` | no | both | constituents in order |
| `n_segments` | `int32` | no | both | ≥ 1 |
| `anchor_lat`, `anchor_lon` | `float64` | yes | STAY only | running-mean centroid |
| `anchor_h3` | `string` | yes | STAY only | h3 v4 cell |
| `envelope_radius_m` | `float32` | yes | STAY only | observed max segment-to-anchor |
| `start_lat`, `start_lon` | `float64` | yes | TRANSIT only | first segment centroid |
| `end_lat`, `end_lon` | `float64` | yes | TRANSIT only | last segment centroid |
| `displacement_m` | `float32` | yes | TRANSIT only | great-circle |
| `path_length_m` | `float32` | yes | TRANSIT only | sum |
| `straightness` | `float32` | yes | TRANSIT only | displacement / path |

## Vectors — output of `embed_segments` / `embed_episodes`

One row per `id`. Stored as parquet with a fixed-size list column.

| col | dtype | nullable | notes |
|---|---|---|---|
| `id` | `string` | no | `segment_id` or `episode_id` |
| `entity_id` | `string` | no | for partitioning |
| `vector` | `list<float32>` (fixed size) | no | dimension declared by `EmbedParams.expected_dim()` |

## Baselines — output of `fit_baselines`

Cohort statistics keyed by the user's `cohort_keys` (e.g., `["road_class"]`,
`["entity_id", "segment_type"]`). One row per cohort.

| col | dtype | nullable | notes |
|---|---|---|---|
| (cohort key columns) | string/string | no | from `cohort_keys` |
| `metric` | `string` | no | enum: `duration_s`, `mean_speed_ms`, etc. |
| `mean` | `float32` | no | |
| `std` | `float32` | no | |
| `n_samples` | `int32` | no | for sample-count fallback |
| `is_fallback` | `bool` | no | true when cohort drew from fleet baseline |

## Constraints applied at L2

`iter_entities` validates each yielded frame against `PingsSchema`. Failures
raise `SchemaError` with the offending row range and column. No silent
coercion. No silent sort.

## Constraints applied at L1 entry

Each L1 function asserts:
- Single unique `entity_id` value (multi-entity input is a user error).
- Sorted by `ts` (assertion, not silent fix).
- Required input columns present per the relevant schema above.

These checks are O(1) on the sorted-monotonic side and O(N) on the entity-id
uniqueness check. Acceptable; trust restored after.

## Out of scope

- Pyarrow extension types (e.g., custom h3 type). Strings are the round-trip-safe
  interchange representation.
- DataFrame-engine-agnostic schema (Polars LazyFrame, DuckDB Relation). Pandas
  is the v1 in-memory engine; Polars is v1.1+.
- Schema migration tooling. Adding a column with a default is the only kind of
  change v1 schemas support; renames or type-changes are breaking and require
  a major bump.
