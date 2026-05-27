# trajkit

A reference implementation of an end-to-end pipeline for turning noisy GPS
pings into searchable trajectory primitives.

The pipeline runs in five stages:

```
raw pings → clean → segment → episode → embed → similarity search
```

Each module is small and self-contained. The design notes below capture
the non-obvious decisions that make the pipeline more than glue code.

## Design notes

- **[clean](design/clean.md)** — kinematic derivation and the 5-flag quality
  precedence rule.
- **[segment](design/segment.md)** — hysteresis state machine and the
  multi-scale circular-R bearing detector.
- **[episode](design/episode.md)** — spatial-envelope STAY detection with a
  dual qualification gate.
- **[embed](design/embed.md)** — base segment-vector recipe and the
  `FeaturePlugin` extension contract.
- **[compare](design/compare.md)** — FAISS index, metric choice, and search.
- **[schemas](design/schemas.md)** — Pandera + Arrow data contracts.
- **[Architecture notes](design/LIBRARY.md)** — repo-wide conventions:
  parameter models, dependencies, testing posture.

## Scope

This repository is a reference design, not a maintained library. The defaults
shipped here are calibrated for two domains (pedestrian and 60-second vehicle
cadence) and will need retuning for other data shapes.
