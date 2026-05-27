# Geolife integration

End-to-end run of the trajkit pipeline on the Microsoft Geolife dataset
(pedestrian / multi-modal GPS, non-fleet).

## What's here

| File | Purpose |
|---|---|
| `reader.py` | `.plt` parser + per-user trajectory loader |
| `run.py` | end-to-end pipeline runner with summary printouts (CLI) |
| `explore.ipynb` | **Visual validation notebook** — every design decision tied to something visible on a real trace |
| `_explore_source.py` | jupytext source for `explore.ipynb` (edit here, regenerate notebook) |

The notebook also writes standalone interactive maps (`*_map.html`,
`*_anatomy.html`) next to it when executed, but they're regenerated
on every run and not committed.

## Visual validation

`explore.ipynb` is the recommended starting point for evaluating
trajkit on Geolife. Open it in JupyterLab / VS Code / on GitHub
and you get:

* histograms of segment-type and episode-type breakdowns
* interactive Folium maps of the trajectory coloured by `segment_type`
* a bearing-driven boundary anatomy with the multi-scale circular-R
  detector visualised alongside the raw bearings
* anatomy maps for one TRANSIT and one STAY episode (the running
  anchor rule in action)
* a segment-similarity demo (cosine over per-segment vectors)
* a TRANSIT-similarity demo (Euclidean over trip-native features,
  data-driven threshold, query auto-picked from the densest part
  of the corpus)
* a STAY-similarity demo (duration + cyclic time context)
* a neighbour-purity benchmark that quantifies whether the segment
  embedding does the job `compare` advertises

## Re-executing the notebook

```bash
uv pip install jupyter jupytext matplotlib folium
uv run jupytext --to ipynb examples/geolife/_explore_source.py \
    -o examples/geolife/explore.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
    examples/geolife/explore.ipynb
```

`_explore_source.py` is the canonical edit surface — re-run the two
commands above to refresh the `.ipynb` after edits.

## Download

Geolife is freely available for research from Microsoft Research Asia.
The download is ~1.6 GB.

1. Download from the [official page][geolife] (or one of the mirrored
   archives if the original is unavailable):

       https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

2. Unzip. The expected directory layout:

       Geolife Trajectories 1.3/
       ├── Data/
       │   ├── 000/
       │   │   └── Trajectory/
       │   │       ├── 20081023025304.plt
       │   │       └── ...
       │   ├── 001/...
       │   └── ...
       └── User Guide-1.3.pdf

3. Cite Geolife if you publish derived work — see the User Guide.

[geolife]: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

## Run

```bash
# 5 users by default
uv run python examples/geolife/run.py "/path/to/Geolife Trajectories 1.3/Data"

# specific users
uv run python examples/geolife/run.py "/path/to/Geolife Trajectories 1.3/Data" \
    --user-list 000,001,002
```

Expected runtime: 30 seconds to a few minutes per user, depending on the
number of `.plt` files (some users have hundreds of trajectories
spanning years). Everything runs in memory; nothing is persisted.

## What the script does

1. Discovers users with a `Trajectory/` subdirectory.
2. Loads selected users — concatenates all `.plt` files per user,
   produces a `PingsSchema`-compatible frame.
3. Runs the pipeline per user (`clean → segment → episode → embed_segments`)
   with pedestrian-tuned parameters (`R_m=30 m`, `T_s=120 s`,
   `min_stay_s=120 s`).
4. Concatenates segment vectors across users and builds a FAISS index.
5. Issues a similarity query for sanity.

## Why pedestrian

Geolife is multi-modal (Beijing residents walking, biking, driving,
taking taxis and trains), which exercises the pipeline well outside the
vehicle-cadence defaults. The pedestrian thresholds used here are
documented in `tests/integration/test_pedestrian_pipeline.py`. The
synthetic fixtures in `tests/integration/` prove the pipeline doesn't
crash; this script proves it produces sensible output on real-world
noisy GPS.
