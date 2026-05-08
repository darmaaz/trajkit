# Library Plan — Shape, Scope, Extraction

Cross-cutting design for the publishable package. Per-module design docs live alongside this one (`episode_layer/DESIGN.md` etc.). This file decides *what the library is*, not *how each algorithm works*.

## 1. Goal

A pip-installable Python library that turns continuous, noisy spatial-temporal traces into a queryable space of comparable behavioral primitives — usable by someone who has never seen our fleet data. Three primitive operations:

1. **Discretize** — pings → typed segments → episodes (multi-scale).
2. **Embed** — episodes (and segments) → fixed-width vectors with context-aware baselining.
3. **Compare** — vector similarity search and cohort-relative anomaly scoring.

Defaults survive at minimum two non-fleet domains (target: pedestrian + maritime). If they don't, we don't publish — we have an internal tool, not a library.

## 2. Out of scope (explicit)

To prevent feature creep, these are **not** library responsibilities:

- Hosted UI / SaaS / web service. (The current FastAPI + Leaflet stays in the consumer repo as an example.)
- Streaming / online detection. Batch only in v1.
- Geometry-heavy GIS (buffers, intersections, polygon ops). Use shapely/geopandas separately.
- ML training framework. We ship vectors; users train downstream models.
- Trajectory storage. We read/write parquet; we don't replace a database.
- Map visualization. We may ship a tiny `viz` extra (matplotlib/folium), nothing heavier.
- Coordinate-reference handling beyond WGS84 in/out. Users project before/after if they need to.

## 3. Library name

**Criteria.** ≤ 8 chars, pronounceable, low-collision on PyPI + GitHub, evokes the discretize-and-search semantics, acceptable in a corporate stack.

**Shortlist.**

| Name | For | Against |
|---|---|---|
| `trajkit` | descriptive, short | generic; PyPI status unverified |
| `trajix` | distinctive, pronounceable | meaningless until you know |
| `dwellkit` | evokes the stay primitive | overweights stays vs. transits |
| `passage` | journeys + visits | too common a word, likely taken |
| `wayward` | trace-shaped, available-feeling | semantic mismatch (means "off-path") |
| `kinet` | "kinetic" stem, short | ambiguous |
| `trkit` | very short | unreadable |

**Recommendation.** `trajkit` for the working name; do a PyPI/GitHub/Google check before committing. If `trajkit` collides, `trajix`. The choice gates nothing technical — just rename later — but PyPI registration is first-come.

**Gotcha.** PyPI squatting is real. Reserve the chosen name with a stub release before announcing publicly.

## 4. Module shape

Single package, five modules, monolithic install with `extras_require` for optional features. Splitting into independent packages is a future migration, not a v1 concern.

| Module | Responsibility | Explicitly NOT |
|---|---|---|
| `trajkit.clean` | Quality flags, drift detection, duplicate handling, gap flagging, stale-position merge | OSM enrichment, projection conversion |
| `trajkit.segment` | Hysteresis state machine + 4-state taxonomy + sustained-bearing splits → segments | Per-segment feature engineering beyond the type label |
| `trajkit.episode` | Spatial-envelope episode detection (STAY/TRANSIT) | Place clustering across entities |
| `trajkit.embed` | Per-segment + per-episode vectorization, context-aware z-score baselining | Learned/contrastive embeddings (later as a separate extra) |
| `trajkit.compare` | FAISS index construction + similarity search + cohort-relative anomaly scores | Anomaly explanation UI |

Plus:

- `trajkit.types` — Pydantic param models, dataframe schemas (Pandera).
- `trajkit.presets` — `SCALE_PRESETS` for known domain classes.
- `trajkit.testing` — synthetic-trace generators, fixture builders. Importable by users for their own tests.
- `trajkit.cli` — thin argparse wrappers around the modules. Not the import surface; opt-in.

**Out of the package, into a sibling repo (`trajkit-fleet/` or just `examples/fleet/`):** mission grouping, route derivation, destination clustering, OSM road/POI enrichment. These are opinionated applications, not primitives. Keeping them out of the core preserves the cross-domain story.

**Gotcha.** If we put missions/routes inside `trajkit`, we silently bake in the fleet-vehicle scale (≥30min dwells, road-class features) and lose maritime/wildlife/mobile users. Hard line.

## 5. API layers

Three layers, one job each. The natural unit is **one entity's pings** — operations on different entities are independent, and a single entity's history fits in memory comfortably even for huge fleets. The library is designed around iterating over entities, not around a single dataset DataFrame.

```
L1  Pure functions    one entity in, one entity out, no I/O
L2  Entity iterator   yields (entity_id, pings_df) from a source; validates at boundary
L3  Runner            iterator + L1 + atomic per-entity write, parallel
                      ──────────────────────────────────────────────
L1.5  Pass-2 functions cohort-level reads of L3 output (baselines, etc.)
```

### L1 — pure functions

**WHY.** Composable, unit-testable, framing-agnostic (notebook, custom pipeline, third-party orchestrator). The contract is the data shape, not the call site.

**HOW.** Each function takes one entity's frame plus a Params instance; returns one entity's frame. No global state, no I/O, no logging side effects. Trusts the schema — validation lives in L2. Single-entity invariant is asserted (one unique `entity_id` value); multi-entity inputs are a user error, not silently grouped.

### L2 — entity iterator

**WHY.** The streaming layer the user touches when L3 is too opinionated. Decouples sources (parquet, Arrow, in-memory frame, custom DB query) from the L1 contract.

**HOW.** `iter_entities(source) -> Iterator[(str, pd.DataFrame)]`. Source is a parquet path (Hive-partitioned by `entity_id` is the fast path), an Arrow `Table`, or an existing DataFrame. `format="csv"` is tolerated for tutorial datasets like Geolife. Custom sources (Postgres, BigQuery, S3 manifests) are user-supplied iterators conforming to the same tuple contract — no plugin system, just any callable that yields tuples. Schema validation runs once per yielded frame using `PingsSchema`; failure raises (no skip mode in v1).

### L3 — runner

**WHY.** Most users don't want to write the iterator + apply + write loop themselves. Single entry point that turns "input parquet, output dir, params" into a complete pipeline run.

**HOW.** `process(source, sink_dir, params, stages=("clean","segment","episode","embed_segments"), n_workers=1) -> RunReport`. Iterates entities, applies stages in fixed order, writes Hive-partitioned parquet per stage. Multiprocessing pool over the iterator (`Pool.imap_unordered`); per-entity write to `sink_dir/<stage>/_tmp/entity_id=<X>/` then atomic rename for crash safety; existing final files are skipped on re-run for cheap resume. Per-entity exceptions abort the whole run in v1 (no skip-on-failure mode); the report names the failing entity and stage.

### L1.5 — pass-2 functions (cross-entity)

**WHY.** Some computations need a global view: fleet-wide z-score baselines, anomaly model fitting, destination clustering. They don't fit per-entity iteration. Surfacing them as a separate phase keeps the per-entity contract clean and makes the cohort dependency visible.

**HOW.** Named functions that read pass-1's Hive output via `pyarrow.dataset` or `pandas.read_parquet`, compute a small artifact, persist it. No parallelism — these are whole-cohort operations by definition. Pass-2 input (segments, episodes) is 1–3 orders of magnitude smaller than pings, so memory pressure is not a concern.

```python
trajkit.fit_baselines(segments_dir, out_path, cohort_keys=["entity_id"]) -> Baselines  # v1
# trajkit.fit_anomaly_model(...)                                                       # v1.1
# trajkit.cluster_destinations(...)                                                    # extras, not core
```

### v1 public skeleton

```python
# L1 — pure, per-entity
trajkit.clean(pings_df, params)                              -> pd.DataFrame
trajkit.merge_stale_positions(pings_df, params)              -> pd.DataFrame
trajkit.segment(pings_df, params)                            -> pd.DataFrame
trajkit.aggregate_segments(pings_df)                         -> pd.DataFrame
trajkit.detect_episodes(segments_df, params)                 -> pd.DataFrame
trajkit.embed_segments(segments_df, params, features=())     -> (np.ndarray, list[str])
trajkit.embed_episodes(episodes_df, seg_vectors, seg_ids)    -> (np.ndarray, list[str])

# L2 — iterator
trajkit.io.iter_entities(source, *, format="auto")           -> Iterator[(str, pd.DataFrame)]

# L3 — runner
trajkit.process(source, sink_dir, params, *, stages=..., n_workers=1) -> RunReport

# Pass-2 — cohort
trajkit.fit_baselines(segments_dir, out_path, cohort_keys)   -> Baselines

# Compare — vector ops, neither per-entity nor pass-2
trajkit.compare.build_index(vectors, ids, metric="cosine")   -> Index
trajkit.compare.search(index, query, k=10, filter=None)      -> list[Hit]

# Presets
trajkit.presets.SCALE_PRESETS                                # logistics_vehicle, pedestrian
```

### Gotchas

- **L1 single-entity invariant.** Asserted on entry; multi-entity inputs raise. Avoids silent `groupby` semantics that would change with parameter order.
- **Pickling for multiprocessing.** L1 functions, Params, and any user-supplied feature plugins must be top-level importable. Lambdas in feature plugins silently break L3 with confusing `PicklingError` traces. Documented; tested against in CI.
- **Per-worker memory.** Each worker holds one entity. With `n_workers=8` and ~500 MB max per entity, peak ≈ 4 GB. Document so users with massive entities know to drop `n_workers`.
- **Resumability is per-entity, not per-stage.** Existing `<stage>/entity_id=<X>/data.parquet` causes the entity to be skipped for that stage. To re-run, delete that path. Document; no `--force` flag in v1.
- **Pass-2 reads partial cohorts silently.** If pass-1 partially failed, `fit_baselines` computes against whatever entities completed. Surface the count in the log; do not silently proceed without it.
- **Mutable default Params.** Use `params: Params | None = None` + `params or Params()` rather than `params: Params = Params()` to avoid Python's class-level default trap, even though Pydantic v2 frozen models are technically safe.

## 6. Data contract

The library's public API surface includes its DataFrame column names. Renames are breaking changes. Schemas are codified twice — once for runtime validation, once for storage — kept consistent by tests.

**Tooling.**
- **Pandera** schemas for runtime validation at the L2 iterator boundary and at pass-2 function boundaries.
- **Arrow** schemas for parquet I/O — types must round-trip through storage cleanly.
- Both definitions live side-by-side in `trajkit.types`. A test asserts they are consistent (column names, dtypes, nullability). This avoids the silent drift where "valid by Pandera" doesn't survive parquet write/read.
- Schemas are exported so consumers can validate their own frames before passing them in.

**Validation runs at exactly one place: the L2 boundary.** L1 pure functions trust the contract for speed. Pass-2 functions re-validate their reads, since pass-1 output has crossed a parquet round-trip.

**Canonical input — pings frame:**

| col | dtype | required | notes |
|---|---|---|---|
| `entity_id` | str | yes | vehicle/vessel/animal/device id |
| `ts` | datetime64[ns, UTC] | yes | tz-aware, UTC |
| `lat`, `lon` | float64 | yes | WGS84 |
| `speed_ms` | float64 | no | derived if absent |
| `bearing_deg` | float64 | no | derived if absent |

Sorted by (`entity_id`, `ts`). Duplicate (`entity_id`, `ts`) rows raise.

**Gotchas (each enforced by schema):**

- Naive timestamps → reject with a clear error. Mixed TZ is a footgun.
- Multiple entities per frame is the default, not an error. Avoids forcing per-entity DataFrames on users who already have a fleet table.
- `lat`/`lon` ranges validated. Subtle bug: someone passes (lon, lat) reversed — schema range check catches it.
- Sort assertion is a check, not a side-effect. The library does not silently sort; it raises if unsorted. Silent sort breaks streaming-style chunked use.
- `entity_id` must be string. Numeric ids cause grief downstream (FAISS payload encoding, group-by performance).

## 7. I/O & storage conventions

**WHY.** A per-entity contract needs an I/O contract. Without one, every consumer reinvents partitioning and discovers it differs across environments.

**Format.** Parquet, snappy compression, for everything the library writes. CSV is tolerated as input via `iter_entities(..., format="csv")` for tutorial datasets but is never produced by a library write.

**Layout — Hive-partitioned by `entity_id`, per stage:**

```
sink_dir/
├── pings_clean/entity_id=A/data.parquet
├── pings_clean/entity_id=B/data.parquet
├── segments/entity_id=A/data.parquet
├── episodes/entity_id=A/data.parquet
├── segment_vectors/entity_id=A/data.parquet   # vectors as fixed-size list columns
└── episode_vectors/entity_id=A/data.parquet
```

**WHY Hive.** `pyarrow.dataset`, `pandas.read_parquet`, `polars.scan_parquet`, and DuckDB all read this layout natively with no custom code. Pass-2 functions trivially scan a stage root. Per-entity directories (`sink_dir/entity_id=A/segments.parquet`) work but force every downstream tool to know our custom layout.

**Vectors as parquet, not `.npz`.** Arrow fixed-size list columns round-trip cleanly. Mild storage overhead vs `.npz`; the gain is layout consistency — every artifact is parquet, every tool reads parquet.

**Atomicity.** Per-entity writes go to `sink_dir/<stage>/_tmp/entity_id=<X>/data.parquet`, then renamed to the final path on success. Crash mid-write leaves the target unmoved; re-run resumes by skipping entities whose final file exists.

**Vector dtype.** `float32`, not `float64`. Halves storage; FAISS expects `float32` anyway.

### Gotchas

- **Schema evolution.** Add a column to a stage in v0.x → old per-entity files don't have it. Each schema carries a `schema_version` field; readers tolerate missing-with-default for that version. Plan from v0.1 release, not retrofitted.
- **Empty entities.** An entity with zero rows after `clean` still produces a (potentially zero-row) per-entity file so resumability works. Don't skip empty entities silently — that breaks idempotent re-runs.
- **Un-partitioned input.** If the source parquet isn't Hive-partitioned by `entity_id` and isn't sorted, `iter_entities` falls back to a `groupby` pass that loads everything. Detected at read time; logged as a warning. Repartition helper deferred to v1.1; users with un-partitioned input do the one-time repartition themselves in v1.
- **CSV is slow and dangerous.** Type inference, mixed encodings, null sentinels. CSV is for tutorials and Geolife, not for production. Document; warn on first CSV read in a process.
- **`_tmp` directory cleanup.** A crashed run leaves `_tmp` directories behind. The runner cleans them on next start before iterating. Don't let them accumulate silently.

## 8. Parameter contract

Each module exposes a Pydantic v2 model. Frozen, `extra='forbid'`. Reasoning:

- **Frozen** — params are immutable in a run; mutation across a pipeline is a debugging nightmare we've all seen.
- **`extra='forbid'`** — silently ignored typos in YAML configs are the worst class of bug. Make them errors.

```python
class EpisodeParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    R_m: PositiveFloat
    T_s: PositiveFloat
    min_stay_s: PositiveFloat
    @classmethod
    def from_preset(cls, name: str) -> "EpisodeParams": ...
```

**Gotcha.** YAML config loading: use `EpisodeParams.model_validate(yaml.safe_load(...))`. Don't accept dicts directly into the public function — that bypasses validation.

## 9. Domain extensibility

The library generalizes via three plug points, no others:

1. **Scale presets.** `SCALE_PRESETS["maritime_vessel"]` returns a bundle of Params instances tuned for that scale. Users override fields without rewriting the bundle.
2. **Feature plugins (embed only).** A `FeaturePlugin` Protocol contributes a fixed-width block to the segment vector. Users supply their own (e.g., a marine-traffic-zone block, an animal habitat-class block) by implementing `name`, `dim`, `compute(segments_df) -> np.ndarray`. The core 81-dim recipe ships as a bundle of default plugins; nothing is hard-coded.
3. **Cohort keys (baselining).** `baseline_zscores(segments_df, cohort_keys=["road_class", "entity_id"])` lets the user specify what "peer group" means for their domain. Default is `["entity_id"]` plus a global fleet baseline.

**What is NOT a plug point:** the segmentation taxonomy (4 states are fixed in v1), the episode closure rule (spatial-envelope is fixed in v1). Adding more is a v2 conversation.

**Gotcha.** Feature plugins must declare their output `dim` up front; we concat blocks by index, so a plugin that returns a variable-width vector breaks the contract. Schema-check every plugin output in tests.

## 10. Dependency policy

Lean by default; heavy deps go in extras.

| Dep | Status | Reason |
|---|---|---|
| `pandas>=2.1` | required | core dataframe |
| `numpy` | required | vectors |
| `pydantic>=2` | required | params + validation |
| `pandera` | required | schema validation |
| `h3>=4` | required | spatial indexing; pin major to avoid the v3→v4 API break |
| `pyproj` | required | great-circle / projection helpers |
| `scikit-learn` | required | IsolationForest, scalers (lightweight slice) |
| `faiss-cpu` | **extra `[search]`** | FAISS wheels are platform-fragile; not everyone needs it |
| `matplotlib` | extra `[viz]` | optional histogram helpers |
| `polars` | extra `[fast]` | future perf path; not exercised in v1 |
| `geopandas` | **forbidden** | pulls GDAL; install hell on Windows + Apple Silicon |
| `osmnx` | **out of package** | OSM enrichment lives in the fleet examples repo |

**Gotcha.** `faiss-cpu` on Apple Silicon: works via conda-forge, fragile via pip. Document the install path in the README, don't pretend it's transparent.

**Gotcha.** `h3` v3 vs v4: function names differ (`grid_ring` vs `k_ring`). Pin to v4 and document; the existing fleet code uses v4 already.

## 11. Repo & packaging

**Single repo, single package, src-layout.** Standard modern Python.

```
<library_repo_root>/
├── pyproject.toml
├── src/trajkit/
│   ├── clean/
│   ├── segment/
│   ├── episode/
│   ├── embed/
│   ├── compare/
│   ├── types.py
│   ├── presets.py
│   └── testing/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/             # MkDocs Material
├── examples/         # runnable notebooks: geolife, ais
└── CHANGELOG.md
```

**`pyproject.toml`** points at `src/trajkit`, declares core + extras, sets `python_requires>=3.10` (typing improvements + pandas 2.0 floor).

**Versioning.** Start at `0.1.0`. SemVer with the convention that `0.x.y` allows breaking changes in minor; we promote to `1.0.0` only after at least one external user has shipped against us.

**Gotcha.** Do not vendor or re-export DataFrame schemas from external libs (e.g., movingpandas). Coupling our public API to someone else's release schedule is a maintenance trap.

## 12. Performance contract

Per-entity computation is the unit. The library never holds the full dataset in memory. Multiprocessing parallelism is over `iter_entities` at L3.

**Targets** (developer laptop, 8 GB free RAM, single entity at a time):

| Stage | Throughput | Notes |
|---|---|---|
| `clean` | 1M pings/sec on the dedup/derive path | 100K pings/sec when stale-position merge runs |
| `segment` | 500K pings/sec | |
| `detect_episodes` | 10K segments/sec | |
| `embed_segments` | 50K segments/sec per plugin block | |
| `compare.search` | < 50 ms over 1M vectors, 81-dim | `IndexFlatIP` |

**Runner with `n_workers=8`.** ~8 entities resident simultaneously, ~4 GB peak.

**Pass-2 `fit_baselines`.** Reads all segments into memory; for fleet scale (~750 entities, ~2M segments), ~500 MB working set. No parallelism — cohort operation by definition.

**Engine.** Vectorized pandas for v1. Polars rewrite is gated on profiling that shows pandas is the bottleneck (likely true for `clean` and `segment` at fleet scale). v1.1+ as `[fast]` extra.

**Out of scope:**
- Streaming sub-entity (one entity that doesn't fit in RAM). Workaround: time-shard before `iter_entities`.
- Distributed multi-node. Single-node multiprocessing only.
- GPU FAISS, GPU embedding. Considered if a real user needs it.

### Gotchas

- **I/O contention at worker startup.** All workers read parquet at once and saturate disk, not CPU. Tune `n_workers` to disk throughput on large entities, not core count.
- **`pyarrow.dataset` overhead on small files.** Per-entity files ≪ 1 MB pay non-trivial dataset-read overhead. For very small entities, batched `pd.read_parquet` is faster. Profile before optimizing.
- **Hidden quadratic features.** Any spatial join in a feature plugin (POI lookup, marine zone) is O(N × M) without an index. h3 cell hashing keeps it linear; verify with a large synthetic frame in the plugin's own tests.
- **Pickling cost.** Sending a 500 MB DataFrame between processes via pickle is slow. The runner does NOT broadcast data — each worker reads its own entity from parquet given an `(entity_id, source_path)` tuple. Don't pre-load and ship.

## 13. Testing strategy

Three layers; hard rules about what each does.

| Layer | Lives in | Speed | Real data? |
|---|---|---|---|
| Unit | `tests/unit/` | <1s each | no — synthetic only |
| Integration | `tests/integration/` | <30s total | small open dataset bundled (≤ 1MB) |
| Cross-domain validation | separate repo `trajkit-validation/` | minutes | Geolife + AIS + MoveBank slices |

**Synthetic trace generator** lives in `trajkit.testing` and is part of the public API — users test their own pipelines with it. Generates parametrized traces: simple stay, simple transit, depot-loop, oscillating-boundary, gap-split, multi-entity batch, drift-flagged pings, stale-position run.

**Coverage targets.** `episode.py` and `segment.py` each ≥ 80% line coverage at v1. Other modules ≥ 60%.

**Gotcha.** Snapshot tests against synthetic data for whole-pipeline output: useful, but every parameter tweak rewrites the snapshot. Use sparingly; prefer explicit assertions on properties (n_episodes, type distribution, anchor convergence).

**Gotcha.** Property-based tests (`hypothesis`) are valuable here — strategies for "monotonic timestamps within entity" and "lat/lon in bounds" catch invariant violations the static fixtures miss. Plan to add but don't gate v1 on them.

## 14. Documentation

Three surfaces, three audiences:

| Surface | Audience | Content |
|---|---|---|
| `README.md` | someone evaluating the library in 60s | one-paragraph pitch + 10-line example + supported domains + install |
| MkDocs site `docs/` | someone implementing | Concept guides (what's an episode, what's calibration, what's a cohort), full API reference (auto-generated from docstrings), per-domain quickstarts (fleet, AIS, Geolife) |
| `CHANGELOG.md` | someone upgrading | every breaking change called out with migration path |

**Convention.** Every public function has a docstring with: one-line summary, Args, Returns, Raises, Example. Lint with `pydocstyle` in CI.

**Gotcha.** The API reference has to render correctly on read-the-docs / mkdocs without us hand-curating every page. Tooling: `mkdocstrings[python]` with type hint introspection. Set this up early; retrofitting docstring style on 50 functions is painful.

## 15. Extraction plan

The existing `fleet-intelligence` repo cannot break during extraction. It is the first consumer of the new library; it remains a runnable case study.

**Phased migration:**

1. **Phase 0 — scaffold.** New repo `trajkit/` exists with `pyproject.toml`, empty src layout, CI green, schema types defined. No behavior moved. Existing `fleet-intelligence` unchanged.
2. **Phase 1 — episode in trajkit.** The episode_layer (which is greenfield) lands in `trajkit.episode` directly, never in `fleet_intelligence/layers/`. This validates the library's package conventions on something with no migration cost.
3. **Phase 2 — clean migration.** Lift `cleaning.py` + `stale_position_merge.py` into `trajkit.clean`. In `fleet-intelligence`, replace `from fleet_intelligence.layers.cleaning import …` with `from trajkit.clean import …`. Tests in both repos green before merging.
4. **Phase 3 — segment, embed, compare** in that order, same pattern.
5. **Phase 4 — strip the duplicate code paths** in `fleet-intelligence`. The `fleet_intelligence/layers/` package shrinks to only the opinionated, fleet-specific layers (mission_grouping, mission_aggregation, destination_embedding, feature_enrichment).

**Gotcha.** Tempting shortcut: keep shim modules in `fleet_intelligence` re-exporting from `trajkit` indefinitely. Cost: every upgrade has two import paths people use. Plan: shims allowed during a phase, deleted at phase end.

**Gotcha.** During phases 2–4, the existing API and frontend may need patching as columns/imports shift. Flag this in each PR; don't merge a phase that breaks `api/main.py`.

## 16. Licensing & attribution

**Default license:** MIT for the library code. Reasoning: lowest friction, encourages adoption, mirrors `pandas`/`numpy` ecosystem norms.

**Gotcha (must resolve before publishing).** The fleet operator's contract may restrict open-sourcing IP developed using their data. Check the contract for derivative-works clauses. If restrictive: either (a) a clean-room rewrite from the design docs without referencing the existing repo, or (b) negotiate an MIT carve-out for the algorithmic code (data stays the operator's). This is a legal call, not an engineering one.

**Attribution:** if any module ever ingests OSM-derived data, ODbL attribution applies to that data. Documented in the relevant module's docstring; out of scope for v1 since OSM enrichment lives in the fleet examples repo.

**Third-party licenses inventory:** required deps' licenses must be compatible with MIT. `faiss` is MIT, `pandera` is MIT, `pydantic` is MIT, `h3` is Apache-2.0, `scikit-learn` is BSD-3 — all fine.

## 17. v1 scope

Pinned scope for the first release. Each item not in v1 is documented as a v1.1+ opt-in addition that does not change the v1 contract — so users who upgrade later don't pay a migration cost.

### Ships in v1

- All five core modules at L1: `clean`, `segment`, `episode`, `embed`, `compare`.
- L2 `iter_entities` accepting parquet path, Arrow `Table`, and in-memory DataFrame; `format="csv"` tolerated for tutorial datasets.
- L3 `process` runner with `n_workers` parameter, multiprocessing pool, Hive-partitioned output, atomic per-entity writes, resume-by-skipping-existing-files.
- Pass-2: `fit_baselines` only.
- Pandera + Arrow schemas for `pings`, `segments`, `episodes`, with consistency test.
- Validate-by-default at the L2 boundary; raise on invalid (no `validate=` opt-out, no `on_invalid="skip"|"warn"` modes).
- Per-entity exception aborts the run (no skip-on-failure mode).
- `SCALE_PRESETS` for two domains: `logistics_vehicle` (built from fleet defaults), `pedestrian` (calibrated on Geolife).
- Synthetic trace generator in `trajkit.testing`, public API from v1.
- Geolife integration test as the cross-domain validation gate.
- MkDocs site: concept guides + auto-generated API reference.

### Deferred to v1.1+

Each is purely additive. v1 users do not pay for these features.

- `fit_anomaly_model` + persisted-model anomaly scoring.
- `validate="first" | "sample" | "none"` opt-outs for million-entity datasets.
- `on_invalid="skip" | "warn"` tolerant iterator modes.
- `repartition_by_entity` helper for non-Hive parquet inputs.
- `time_shard` helper for entities too large to fit in memory.
- Resilient runner (catch + continue + per-entity failure report).
- Polars in-memory engine (`[fast]` extra).
- Maritime preset + AIS-slice validation (a second cross-domain gate).
- `trajkit-fleet` sibling package (missions, routes, destination clustering, OSM enrichment).

### Out of scope indefinitely

- Streaming sub-entity computation.
- Distributed multi-node.
- Hosted UI / SaaS.
- ML training framework.

### Gotchas

- **"v1.1" stays vague until v1 ships.** Ordering by demand from real first users is the only sensible prioritization. No dates promised.
- **Geolife is the publication gate.** Without Geolife passing, v1 doesn't ship. AIS in v1.1 reinforces the cross-domain claim but is not gating.
- **No silent feature drift.** Every item on the deferred list stays deferred until a real user asks for it. Engineering-justified-in-isolation is not a sufficient reason to add scope back.

## 18. Decision register (open)

Tracks unresolved decisions. When resolved, decisions move out of this section into the relevant section above.

| # | Question | Owner | Blocker for | Status |
|---|---|---|---|---|
| D1 | Final library name | user | PyPI registration | open |
| D2 | Repo location (new GitHub org? personal?) | user | Phase 0 scaffold | open |
| D3 | License — MIT vs Apache-2.0 | user, after fleet-operator contract review | first public release | open |
| D4 | Validation datasets — Geolife alone, or + AIS slice | — | publication claim | **resolved**: Geolife in v1, AIS in v1.1 (§17) |
| D5 | Whether `compare` ships with FAISS bundled vs `[search]` extra | based on Apple Silicon install testing | first public release | open |
| D6 | Whether segmentation taxonomy stays at 4 states or becomes plugin-extensible | empirical — does maritime want different states? | v2 | **deferred to v2** |
| D7 | Synthetic trace generator API stability — public from v1 or `_internal` | — | v1 release | **resolved**: public from v1 (§17) |
| D8 | Whether to ship `trajkit-fleet` sibling on day 1 vs `examples/fleet/` | adoption signal | first external user | **resolved**: deferred to v1.1+ (§17) |

**Gotcha.** Decision registers rot. Each open item gets a target resolution date, reviewed every two weeks until closed.

## What "complete" looks like for this plan

This document is **the plan**, not the implementation. It is complete when:

- Each section has been read and challenged by the user (you).
- The decision register has owners against every row.
- The naming check (PyPI + GitHub + Google) has been done.
- Per-module design docs exist for at least `episode` (✓), `clean`, `segment`, `embed`, `compare` before any code lands in the library repo.

**Until those are in place, no code goes into the library repo.** The most expensive mistake we can make is committing module skeletons that bake in unexamined assumptions.
