# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-08

First feature-complete release. The full pipeline ships: discretize →
embed → compare, with L2 entity iterator and L3 process orchestrator.

### Added

- `trajkit.types`: Pandera DataFrameModel + Arrow Schema declarations for
  `Pings`, `CleanedPings`, `SegmentedPings`, `Segments`, `Episodes`,
  `Vectors`, plus `make_baselines_schema(cohort_keys)` factory. Schema
  versioning via parquet metadata.
- `trajkit.clean`: `clean()` (dedup, kinematic derivation, quality flags)
  and `merge_stale_positions()` for stale-position GPS devices.
  `CleanParams`, `StaleMergeParams`. Quality-flag precedence:
  `DEVICE_FAULT > SPEED_OUTLIER > DRIFT > GAP_FOLLOWS > VALID`.
- `trajkit.segment`: hysteresis state machine + 4-state taxonomy
  (`MOVE` / `MOVE_BRIEF` / `STOP_BRIEF` / `STOP_DWELL`),
  sustained-bearing splits, false-stop override, `aggregate_segments()`
  building `SegmentsSchema`. `SegmentParams`.
- `trajkit.episode`: spatial-envelope STAY/TRANSIT detection with
  endpoint-aware containment + max-radius qualification gate.
  `EpisodeParams` with `R_m`, `T_s`, `min_stay_s`, `h3_resolution`.
- `trajkit.embed`: 32-dim base recipe (kinematic + cyclic + segment-type
  + spatial), `FeaturePlugin` Protocol for extensions, `embed_episodes`
  pooling (`mean + std + max-by-magnitude` + 5 episode scalars),
  `baseline_zscores` cohort applier. `EmbedParams`.
- `trajkit.compare`: FAISS `IndexFlatIP` (cosine) and `IndexFlatL2`
  via `build_index`, `search` with optional `filter_ids`, `save_index`/
  `load_index` with FAISS native serialisation, per-call
  `anomaly_score` via IsolationForest. `Hit` dataclass.
- `trajkit.io`: `iter_entities` L2 boundary — accepts parquet path,
  Arrow Table, DataFrame, or CSV; coerces canonical dtypes; sorts;
  validates `PingsSchema` per yield.
- `trajkit.runner`: L3 `process` orchestrator with multiprocessing pool,
  Hive-partitioned atomic per-entity writes, resume-by-skipping-existing,
  per-entity-failure abort. `RunParams`, `RunReport`,
  `DEFAULT_STAGES`. `RunParams.from_preset(name)` ergonomic API.
- `trajkit.baselines`: pass-2 `fit_baselines` with cohort grouping,
  global-fallback for sparse cohorts, parquet persistence.
  `BaselineParams`.
- `trajkit.testing`: `make_pings` + `make_segments` minimal builders
  for sanity checks and quickstart examples.
- `trajkit.presets`: `SCALE_PRESETS` dict with `logistics_vehicle` and
  `pedestrian` v0.1.0 domain bundles. `get_preset(name)` lookup.
- 268 unit + integration tests (ruff strict + mypy strict clean).
- MkDocs site: 3 concept guides (overview, pipeline, parameters) + 11
  auto-generated API reference pages via mkdocstrings.
- Pedestrian cross-domain integration test
  (`tests/integration/test_pedestrian_pipeline.py`) — synthetic
  Geolife-shape trace through the full pipeline.

### Changed

- `EpisodesSchema.duration_s` and `SegmentsSchema.duration_s`: relaxed
  from `gt=0` to `ge=0` to accept legitimate zero-duration single-ping
  segments.

## [0.0.1] - 2026-05-08

### Added

- Initial repository scaffold with module skeletons (`clean`, `segment`, `episode`, `embed`, `compare`, `io`, `runner`, `testing`).
- Design documents under `docs/design/`:
  - `LIBRARY.md` — cross-cutting plan (shape, scope, extraction).
  - `schemas.md` — canonical column schemas (single source of truth).
  - `clean.md`, `segment.md`, `episode.md`, `embed.md`, `compare.md` — per-module designs.
- `pyproject.toml` with core dependencies and `[search]`, `[viz]`, `[fast]`, `[dev]` extras.
- GitHub Actions workflows: `ci.yml` (lint, typecheck, test) and `docs.yml` (mkdocs build).
- MkDocs configuration with material theme and `mkdocstrings`.
- Smoke test asserting package import + version exposure.
