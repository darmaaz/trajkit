# trajkit

> Reference implementation of an end-to-end pipeline for turning noisy GPS pings
> into searchable trajectory primitives.

The repository implements a five-stage pipeline:

```
raw pings → clean → segment → episode → embed → similarity search
```

Each stage is a small module with explicit design choices, documented in
`docs/design/`. The two non-obvious ideas the pipeline depends on:

- **Segmentation that splits on both motion state and sustained direction
  change.** A two-threshold hysteresis state machine handles stop/move
  transitions without flicker; a multi-scale circular-statistics detector
  over distance windows handles direction changes (street corners and
  arterial sweeps) without ping-rate sensitivity.
- **Similarity search over per-segment vectors.** Each segment is embedded
  as a fixed-width float32 vector and indexed with FAISS. Queries return
  rank-ordered nearest neighbours that join back to the original segment
  metadata.

## Modules

| Module | Stage | Highlights |
|---|---|---|
| `trajkit.clean` | clean | 5-flag quality taxonomy with explicit precedence |
| `trajkit.segment` | segment | hysteresis state machine + circular-R bearing detector |
| `trajkit.episode` | episode | spatial-envelope STAY detection with dual qualification gate |
| `trajkit.embed` | embed | base vector recipe + `FeaturePlugin` protocol for extensions |
| `trajkit.compare` | similarity | FAISS index + search |
| `trajkit.testing` | helpers | synthetic generators for tests and examples |
| `trajkit.types` | schemas | Pandera + Arrow data contracts |

## Reading order

The per-module design notes under `docs/design/` walk through the choices and
their alternatives. Each note maps 1:1 to a module:

- [`clean`](docs/design/clean.md) — kinematic derivation and the flag precedence rule
- [`segment`](docs/design/segment.md) — hysteresis and circular-R bearing detection
- [`episode`](docs/design/episode.md) — spatial envelope and dual qualification
- [`embed`](docs/design/embed.md) — base recipe and plugin contract
- [`compare`](docs/design/compare.md) — index, metric, and search

The Geolife example under `examples/geolife/` runs the full pipeline end-to-end
on real pedestrian data.

## Scope

This is a reference design, not a maintained library. Defaults here are
calibrated for two domains — pedestrian and 60-second vehicle cadence — and
will need retuning for other data shapes. If you find a pattern useful, copy
it; that's what the code is for.

## Run it

```bash
git clone https://github.com/darmaaz/trajkit.git
cd trajkit
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Synthetic quickstart:

```python
from trajkit.clean import clean
from trajkit.segment import segment, aggregate_segments
from trajkit.episode import detect_episodes
from trajkit.embed import embed_segments
from trajkit.compare import build_index, search
from trajkit.testing import make_pings

pings    = make_pings(n=600, motion="stop_then_move")
cleaned  = clean(pings)
segments = aggregate_segments(segment(cleaned))
episodes = detect_episodes(segments)
vectors, ids = embed_segments(segments)

index = build_index(vectors, ids, metric="cosine")
hits  = search(index, vectors[0], k=5)
```

Real-data example (pedestrian, Microsoft Geolife):

```bash
python examples/geolife/run.py /path/to/Geolife/Data --users 5
```

See [`examples/geolife/README.md`](examples/geolife/README.md) for download
instructions.

## License

MIT — see [`LICENSE`](LICENSE).
