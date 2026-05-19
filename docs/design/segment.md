# `trajkit.segment`

## Purpose

Convert a cleaned per-ping stream into typed atomic behavioral units:
sustained movement (`MOVE`), short repositioning (`MOVE_BRIEF`), brief
stationary periods (`STOP_BRIEF`), and qualifying dwells (`STOP_DWELL`).
Naive speed-thresholding produces flicker; this layer's value is the
hysteresis state machine and sustained-bearing detection that suppress it.
The output is the input to `episode` (multi-segment grouping) and `embed`
(vectorization).

## Assumptions

- Input is the output of `clean` for one entity, with `speed_ms` populated
  and `quality_flag` set.
- Sorted by `ts`; the state machine consumes ping order, not real-time.
- Time gaps within the trace terminate the current segment regardless of
  state (a gap is, by construction, not part of any segment's behavior).
- The four-state taxonomy is fixed.
- Hysteresis thresholds are user-decided; pass a custom `SegmentParams`
  to retune for your scale.
- Pings flagged `DEVICE_FAULT` or `SPEED_OUTLIER` are treated as missing
  (excluded from speed/bearing decisions but kept in the row count).

## Architecture

Two pure functions; the second consumes the first's output.

```python
segment(pings_df: pd.DataFrame, params: SegmentParams) -> pd.DataFrame
aggregate_segments(segmented_pings_df: pd.DataFrame) -> pd.DataFrame
```

`segment`:
- Hysteresis state machine. Two thresholds (`stop_speed_kmh`,
  `resume_speed_kmh`) define the move/stop transition with a dead zone
  between them, preventing flicker at the boundary.
- A bearing-change rule splits a `MOVE` based on circular statistics
  over distance-based sliding windows (see "Bearing change detection"
  below).
- Brief-vs-dwell distinction is by duration only (`dwell_threshold_min`).
- `MOVE_BRIEF` is recognized when a `MOVE` candidate has fewer than
  `move_brief_min_pings` or shorter than `move_brief_max_duration_s`.
- Output is the per-ping frame with `segment_id` and `segment_type` added.

### Bearing change detection

Bearing is angular. Linear statistics on bearing deltas confound a
"single sharp turn" (one big delta diluted by many zeros) with "noise
around a stable direction" (deltas oscillating around zero) — the
arithmetic mean of `|delta|` is similar in both cases. The detector
uses **mean resultant length** `R` of bearings inside a *distance-based*
sliding window, which correctly distinguishes the two: high `R` =
bearings cluster (going one direction); low `R` = bearings spread
around the unit circle (direction is changing).

Distance windows (rather than time windows) make the detector
ping-rate invariant: a 200 m window contains the same physical-
behaviour magnitude regardless of whether it was logged at 1 Hz or
1/min.

**Multi-scale**: `R` is computed at two window sizes
(`bearing_window_short_m=75 m` for street-corner-scale turns and
`bearing_window_long_m=200 m` for arterial / sustained turns).
Boundary entry signal: `R` falls below `bearing_r_enter` (default
0.80) in *either* window. Exit signal: `R` rises above `bearing_r_exit`
(default 0.92) in *both* windows. Asymmetric thresholds form a
Schmitt trigger that prevents flicker around the entry threshold.
`bearing_r_enter = 0.80` sits above the math floor `√0.5 ≈ 0.707`
for a clean 90° turn centred in the window, so the detector also
fires on sub-90° (arterial-bend-scale) direction changes; the high
`bearing_r_exit = 0.92` demands a clean straight run to release.

**Distance-based hysteresis**: state flips only when the relevant
signal has persisted for `bearing_sustain_m` (default 30 m) of
trajectory distance. A boundary fires on the rising edge of the
"direction-changing" state, restricted to moving pings.

**Sparse-window guard**: a per-window minimum-bearings count blocks
`R` from being computed on starved windows. The configured
`bearing_window_min_pings` (default 5) is a **ceiling**, not a fixed
threshold; it adapts down to a floor of 2 when the trace's observed
median per-ping displacement would otherwise make the configured
value unsatisfiable inside the window. This lets the detector
compute `R` on sparse vehicular-cadence data (e.g. 5 s pings at
50 km/h, where a 75 m window contains ~1 ping) instead of NaNing
the windows out across long stretches. The adaptive value is
derived once per `_bearing_boundaries` call from the median of
positive `displacement_m`.

**Cumulative distance is computed motion-only**: stops collapse to
zero distance, so a pause inside a journey doesn't burn the window
with no-progress pings. Stop-period bearings are also masked out of
the `R` computation (they're GPS jitter at near-stationary points and
would pollute the unit-circle vector mean).

`aggregate_segments`:
- Single `groupby(segment_id)` over the per-ping frame.
- Computes per-segment kinematics (mean/max speed, path length, displacement,
  straightness, bearing variance), spatial endpoints (start/end lat/lon, h3),
  temporal endpoints (start/end ts, duration), and `n_pings`.
- Output validates against the `Segments` schema.

The state machine is implemented as a vectorized scan where possible
(speed comparisons, bearing differences) with a single Python-level pass
for state transitions. Sustained-bearing detection uses a sliding window
on a circular-difference array, O(N).

## Efficiency

- O(N) over pings. State-transition pass is the only non-vectorized step.
- Target: 500 K pings/sec/entity on the developer laptop.
- Memory: one frame's worth, no copies; segment_id and segment_type are
  appended in place.
- `aggregate_segments` is a single groupby; circular variance over each
  group is the heaviest per-segment computation, still O(pings_in_segment).
- No spatial index; no cross-entity work.

## Usage

```python
from trajkit.segment import segment, aggregate_segments, SegmentParams

per_ping_segmented = segment(clean_df, SegmentParams())
segments_df = aggregate_segments(per_ping_segmented)
```

## Not in this layer

- Episode grouping — [`episode`](episode.md).
- Vector embedding — [`embed`](embed.md).
- Cross-entity z-score normalisation — user concern.
- Persistence — user concern.
