# Parameters

Each module has its own frozen Pydantic `Params` model. Pass them
explicitly, or use a domain preset.

## Domain presets

v0.1.0 ships two presets:

| Preset | Episode `R_m` | Episode `T_s` | Episode `min_stay_s` | Use for |
|---|---|---|---|---|
| `logistics_vehicle` | 200 m | 300 s | 180 s | trucks, fleet vehicles |
| `pedestrian` | 30 m | 120 s | 120 s | walking-scale data (Geolife-shape) |

```python
from trajkit.runner import RunParams

params = RunParams.from_preset("pedestrian")
```

Other domains (maritime vessels, wildlife, mobile phone) are documented
in the design but deferred to v1.1+. For now, override the relevant
fields:

```python
from trajkit.episode import EpisodeParams

# Approximate maritime preset (1500 m envelope, 30-min persistence)
params = RunParams(
    episode=EpisodeParams(R_m=1500.0, T_s=1800.0, min_stay_s=1800.0),
)
```

## Per-stage parameters

| Stage | Params class | Key knobs |
|---|---|---|
| `clean` | `CleanParams` | `max_speed_kmh`, `drift_radius_m`, `gap_threshold_min`, device-fault thresholds |
| stale-merge (opt-in) | `StaleMergeParams` | `detection_ratio_threshold`, `min_pings_for_detection` |
| `segment` | `SegmentParams` | hysteresis (`stop_speed_kmh`, `resume_speed_kmh`), `bearing_change_deg`, `dwell_threshold_min`, `move_brief_*`, `h3_resolution` |
| `episode` | `EpisodeParams` | `R_m`, `T_s`, `min_stay_s`, `h3_resolution` |
| `embed` | `EmbedParams` | `cyclic_harmonics`, `spatial_bounds`, `l2_normalize` |

`RunParams` bundles all of the above for the L3 runner.

## Three knobs to know first

If you only want to think about a few parameters, these matter most:

### `EpisodeParams.R_m` — what counts as "here"?

The radius of the spatial envelope. Within `R_m` of the running anchor,
segments are part of the same stay; outside, they're departures. Set
to the typical operational scale of "same place" in your domain:

* Yard / loading-bay scale → **100–300 m** (logistics vehicles)
* Anchorage / port stay → **1–2 km** (maritime vessels)
* Foraging patch → **species-dependent**, 50 m–1 km (wildlife)
* Place stay → **50–200 m** (mobile phone)
* Building / room → **10–30 m** (pedestrian)

### `EpisodeParams.T_s` — how long a brief excursion can be

Once outside the envelope, the entity has `T_s` seconds to come back
before the stay closes. Captures "drove around the block and came
back" without splitting into two stays. Defaults: 300 s (vehicles), 120 s
(pedestrian).

### `EpisodeParams.min_stay_s` — the minimum time to count as staying

Below this, candidate stays are folded into the surrounding transit.
Filters "slow crawl through a parking lot" being labeled a stay.

## Embedding bounds

`EmbedParams.spatial_bounds` is `(lat_min, lat_max, lon_min, lon_max)`
for the spatial-normalisation block. Defaults to the world
`(-90, 90, -180, 180)` for a coarse global encoding. **Tighten to your
cohort's bounding box** for a denser representation:

```python
EmbedParams(spatial_bounds=(19.30, 19.55, -99.30, -99.05))  # Mexico City
```

## Frozen and explicit

All `Params` models are immutable (`frozen=True`) and reject unknown
fields (`extra="forbid"`). A typo in a kwarg name fails at construction
rather than silently doing nothing. `Params` instances are picklable, so
they thread through multiprocessing workers without issue.
