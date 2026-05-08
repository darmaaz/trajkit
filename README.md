# trajkit

> Behavioral primitives for spatial-temporal traces.

**Status:** pre-alpha scaffold (v0.0.1). Design docs in [`docs/design/`](docs/design/) describe the planned shape; no implementation yet. This repo is private pending v0.1.0.

## What this is (when complete)

A pip-installable Python library that turns continuous, noisy spatial-temporal traces — GPS pings, AIS, animal collars, mobile location data — into a queryable space of comparable behavioral primitives. Three operations:

1. **Discretize** — pings → typed segments → episodes.
2. **Embed** — fixed-width vectors with context-aware baselining.
3. **Compare** — vector similarity search and cohort-relative anomaly scoring.

See [`docs/design/LIBRARY.md`](docs/design/LIBRARY.md) for the full plan.

## Install (when published)

```bash
pip install trajkit
pip install "trajkit[search]"   # FAISS similarity search
pip install "trajkit[viz]"      # matplotlib, folium
pip install "trajkit[fast]"     # polars in-memory engine
```

## Roadmap

- **v0.0.1** — scaffold, design docs, CI green (this commit).
- **v0.1.0** — `clean`, `segment`, `episode`, `embed`, `compare` core; `iter_entities` + `process` runner; `fit_baselines` pass-2; Geolife integration test passing.
- **v1.0.0** — first stable release after at least one external user has shipped against it.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src
mkdocs serve
```

## License

License pending. See [`LICENSE`](LICENSE).
