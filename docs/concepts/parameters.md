# Parameters

Each stage has its own frozen Pydantic v2 parameter model. Pass them
explicitly per call; defaults are documented inline on each `Params` class.

| Stage | Params class | Key knobs |
|---|---|---|
| `clean` | `CleanParams` | `max_speed_kmh`, `drift_radius_m`, `drift_speed_kmh`, `gap_threshold_min`, device-fault thresholds |
| `segment` | `SegmentParams` | hysteresis (`stop_speed_kmh`, `resume_speed_kmh`), bearing detector (`bearing_window_short_m`, `bearing_window_long_m`, `bearing_r_enter`, `bearing_r_exit`, `bearing_sustain_m`), `dwell_threshold_min`, `move_brief_*`, `h3_resolution` |
| `episode` | `EpisodeParams` | `R_m`, `T_s`, `min_stay_s`, `h3_resolution` |
| `embed` | `EmbedParams` | `cyclic_harmonics`, `spatial_bounds`, `l2_normalize` |

## Tuning to your data

The module defaults are calibrated for vehicle-scale traces with ~60s ping
cadence. Pedestrian-scale calibration lives in the integration test under
`tests/integration/test_pedestrian_pipeline.py`:

```python
from trajkit.segment import SegmentParams
from trajkit.episode import EpisodeParams

PEDESTRIAN_SEGMENT = SegmentParams(
    stop_speed_kmh=1.0,
    resume_speed_kmh=3.0,
    max_stop_displacement_m=50.0,
)
PEDESTRIAN_EPISODE = EpisodeParams(R_m=30.0, T_s=120.0, min_stay_s=120.0)
```

Other domains (maritime vessels, wildlife, mobile phones) need their own
calibration; the patterns to follow are in the test.

## Three knobs to know first

### `EpisodeParams.R_m` — what counts as "here"?

The radius of the spatial envelope. Within `R_m` of the running anchor,
segments are part of the same stay; outside, they're departures. Set to
the typical operational scale of "same place" in your domain:

- Building / café scale → **20–50 m** (pedestrians)
- Yard / loading-bay scale → **100–300 m** (road vehicles)
- Anchorage / port stay → **1–2 km** (maritime vessels)

### `SegmentParams.stop_speed_kmh` / `resume_speed_kmh` — when is the entity moving?

The two-threshold hysteresis. Below `stop_speed_kmh` the state moves to
stopped; above `resume_speed_kmh` it moves to moving. Between them is a
dead zone where the current state persists. The gap is what prevents
flicker. Module defaults are vehicle-scale (2 / 5 km/h); set both lower
for pedestrians and higher for marine vessels.

### `SegmentParams.bearing_window_short_m` / `bearing_window_long_m` — how far before a "turn"?

The two distance windows for the circular-R bearing detector. The short
window catches street-corner-scale turns; the long window catches
arterial / sustained sweeps. Defaults (75 m / 200 m) are calibrated for
typical road geometry. Pedestrians may want smaller windows; ships need
much larger ones. See [`segment`](../design/segment.md) for the full
method and why the windows are distance-based rather than time-based.
