# trajkit

> Behavioral primitives for spatial-temporal traces.

**Status:** v0.1.0 in active development. The full pipeline — clean, segment, episode, embed, compare, plus the L3 process runner and pass-2 baselines — is implemented and tested. Documentation and the first public release are in flight.

## What it does

A pip-installable Python library that turns continuous, noisy spatial-temporal traces — GPS pings, AIS, animal collars, mobile location data — into a queryable space of comparable behavioral primitives. Three operations:

1. **Discretize** — pings → typed segments (MOVE / MOVE_BRIEF / STOP_BRIEF / STOP_DWELL) → episodes (STAY / TRANSIT).
2. **Embed** — fixed-width float32 vectors per segment and per episode, with context-aware baselining.
3. **Compare** — FAISS-backed similarity search and per-call anomaly scoring.

See [`docs/design/LIBRARY.md`](docs/design/LIBRARY.md) for the full plan.

## Quickstart

```python
import trajkit
from trajkit.runner import RunParams, process
from trajkit.testing import make_pings

# 1. A synthetic single-entity trace (real users pass parquet/Arrow/DataFrame)
pings = make_pings(n=600, motion="stop_then_move")

# 2. Choose a domain preset (or pass a custom RunParams)
params = RunParams.from_preset("pedestrian")  # or "logistics_vehicle"

# 3. Run the full pipeline. Outputs land as Hive-partitioned parquet.
report = process(pings, "out/", params, n_workers=1)
print(report.succeeded, report.completed_entity_ids)

# 4. Read any stage's output back
import pandas as pd
episodes = pd.read_parquet("out/episode/entity_id=v1/data.parquet")
print(episodes[["episode_type", "duration_s", "n_segments"]])
```

For finer-grained use, the L1 functions are pure and composable:

```python
from trajkit.clean import clean
from trajkit.segment import segment, aggregate_segments
from trajkit.episode import detect_episodes
from trajkit.embed import embed_segments

cleaned = clean(pings)
segments = aggregate_segments(segment(cleaned))
episodes = detect_episodes(segments)
vectors, ids = embed_segments(segments)
```

For cohort-aware operations:

```python
from trajkit.baselines import fit_baselines
from trajkit.embed import baseline_zscores
from trajkit.compare import build_index, search

baselines = fit_baselines("out/segment/", cohort_keys=["entity_id"])
zscored = baseline_zscores(segments, baselines, cohort_keys=["entity_id"])

index = build_index(vectors, ids, metric="cosine")
hits = search(index, vectors[0], k=10)
```

## Install (when published)

```bash
pip install trajkit
pip install "trajkit[search]"   # FAISS similarity search
pip install "trajkit[viz]"      # matplotlib, folium
pip install "trajkit[fast]"     # polars in-memory engine (v1.1+)
```

## Concepts

| Layer | Unit | Schema | What |
|---|---|---|---|
| `clean` | one ping | `CleanedPingsSchema` | quality flags, dedup, optional stale-position merge |
| `segment` | one ping → one segment | `SegmentsSchema` | hysteresis state machine, 4-state taxonomy |
| `episode` | one episode (STAY/TRANSIT) | `EpisodesSchema` | spatial-envelope closure rule (R, T, min_stay_s) |
| `embed` | per-segment / per-episode | `VectorsSchema` | base recipe + plugins, L2-normalised float32 |
| `compare` | FAISS index | — | top-k similarity, per-call anomaly |
| `baselines` | cohort statistics | `BaselinesSchema` | pass-2 mean/std for z-score normalisation |

## Roadmap

- **v0.0.1** — scaffold, design docs, CI green.
- **v0.1.0** — full pipeline shipped (this branch). Geolife-shape integration test passing. Awaiting docs + first public release.
- **v1.1+** — anomaly-model fitting (`fit_anomaly_model`), Polars in-memory engine, `time_shard`/`repartition` helpers, expanded `trajkit.testing` scenarios.
- **v1.0.0** — first stable release after at least one external user has shipped against it.

## Development

```bash
git clone https://github.com/darmaaz/trajkit.git
cd trajkit
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest                # unit + integration
ruff check .
mypy
mkdocs serve          # docs preview
```

## License

License pending. See [`LICENSE`](LICENSE).
