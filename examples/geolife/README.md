# Geolife integration

End-to-end run of the trajkit pipeline on the Microsoft Geolife dataset
— the v0.1.0 cross-domain validation gate (pedestrian / multi-modal,
non-fleet).

## What's here

| File | Purpose |
|---|---|
| `reader.py` | `.plt` parser + per-user trajectory loader |
| `run.py` | end-to-end pipeline runner with summary printouts (CLI) |
| `explore.ipynb` | **Visual validation notebook** — segment maps, STAY anchors, similarity demo |
| `_explore_source.py` | jupytext source for `explore.ipynb` (edit here, regenerate notebook) |
| `segments_map.html` | standalone trajectory map (color-coded by segment_type) |
| `stays_map.html` | standalone STAY anchors map |
| `similarity_map.html` | standalone similarity-search visualisation |
| `README.md` | this file — download + run instructions |

## Visual validation

`explore.ipynb` is the recommended starting point for evaluating
trajkit on Geolife. Open it in JupyterLab / VS Code / on GitHub and you
get:

* histograms of segment-type breakdown and STAY duration
* an interactive Folium map of the trajectory color-coded by
  `segment_type`
* an interactive map of STAY anchors sized by duration
* a similarity-search demo (query segment + top-5 nearest neighbours
  on a map)

The standalone `*.html` files contain the same maps for direct browser
viewing without Jupyter.

## Re-executing the notebook

```bash
pip install jupyter jupytext matplotlib folium
jupytext --to ipynb examples/geolife/_explore_source.py \
    -o examples/geolife/explore.ipynb
jupyter nbconvert --to notebook --execute --inplace \
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
python examples/geolife/run.py "/path/to/Geolife Trajectories 1.3/Data"

# specific users
python examples/geolife/run.py "/path/to/Geolife Trajectories 1.3/Data" \
    --user-list 000,001,002

# persist outputs to a chosen directory
python examples/geolife/run.py "/path/to/Geolife Trajectories 1.3/Data" \
    --users 3 --sink ./geolife_out
```

Expected runtime: 30 seconds to a few minutes per user, depending on the
number of `.plt` files (some users have hundreds of trajectories
spanning years).

## What the script does

1. Discovers users with a `Trajectory/` subdirectory.
2. Loads selected users — concatenates all `.plt` files per user,
   produces a `PingsSchema`-compatible frame.
3. Runs `trajkit.runner.process` with the published pedestrian preset
   (`R_m=30 m`, `T_s=120 s`, `min_stay_s=120 s`).
4. Prints a per-stage summary (rows + key descriptors).
5. Builds a FAISS similarity index over segment vectors and runs a
   query.
6. Computes cohort baselines + z-scores.
7. Reports STAY anchor recurrence per user.

## Why this is the cross-domain gate

The pedestrian / multi-modal nature of Geolife (Beijing residents
walking, biking, driving, taking taxis and trains) sits outside the
trajkit defaults that were originally fleet-tuned. Running the same
pipeline + the published `pedestrian` preset on real Geolife data
either:

* validates that the library generalises beyond fleet vehicles, or
* surfaces concrete tuning issues we'd never have caught synthetically.

Either outcome is informative. Synthetic pedestrian fixtures (in the
`tests/integration/` suite) prove the pipeline doesn't crash; this
script proves it produces sensible output on real-world noisy GPS.
