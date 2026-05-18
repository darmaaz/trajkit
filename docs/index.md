# trajkit

A reference implementation of an end-to-end pipeline for turning noisy GPS
pings into searchable trajectory primitives.

The pipeline runs in five stages:

```
raw pings → clean → segment → episode → embed → similarity search
```

Each module is small and self-contained. The design choices that make the
pipeline more than glue code are documented under [Design](design/clean.md).

## Where to start

- **[Pipeline](concepts/pipeline.md)** — end-to-end walkthrough of what each
  stage takes and produces.
- **[Parameters](concepts/parameters.md)** — the parameter model per stage and
  how to tune them for your data.
- **[Design notes](design/clean.md)** — the reasoning behind the non-obvious
  choices (flag precedence, hysteresis + circular-R bearing detection,
  spatial-envelope episode rule, segment vector recipe).
- **[API reference](reference/index.md)** — auto-generated from docstrings.

## Scope

This repository is a reference design, not a maintained library. The defaults
shipped here are calibrated for two domains (pedestrian and 60-second vehicle
cadence) and will need retuning for other data shapes.
