# Concepts

trajkit turns continuous, noisy spatial-temporal traces into a queryable
space of comparable behavioral primitives. Three operations:

| Operation | Module | What it produces |
|---|---|---|
| **Discretize** | `clean` → `segment` → `episode` | typed episodic units of behaviour |
| **Embed** | `embed` | fixed-width float32 vectors per segment / per episode |
| **Compare** | `compare` | similarity search + anomaly scoring |

Plus a cross-cutting orchestration layer:

| Layer | Module | What |
|---|---|---|
| Per-entity execution | `runner.process` | iterates entities → applies stages → atomic Hive parquet writes |
| Cohort statistics | `baselines.fit_baselines` | pass-2 mean/std for cohort-aware z-scoring |

## Reading order

* [Pipeline](pipeline.md) — end-to-end flow from raw pings through to
  searchable vectors.
* [Parameters](parameters.md) — how to tune for your domain, including
  the two v0.1.0 presets and how to override.

For algorithmic detail and rationale (the *why* behind each design
decision), see the per-module documents under `docs/design/`.
