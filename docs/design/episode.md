# `trajkit.episode`

## Goal

Group segments into episodes — `STAY` or `TRANSIT` — using a spatial-
envelope rule. An episode is the smallest grouping of segments that
corresponds to a coherent operational unit a human would name: a visit
or a leg.

## Input / Output

**Input:** a segments frame matching `SegmentsSchema` (see
[`schemas.md`](schemas.md)), single-entity, sorted by `start_ts`.

**Output:** an episodes frame matching `EpisodesSchema`:

| col | for type | notes |
|---|---|---|
| `episode_id` | both | `ep_<entity>_<seq>` |
| `entity_id` | both | |
| `episode_type` | both | `STAY` or `TRANSIT` |
| `start_ts`, `end_ts`, `duration_s` | both | |
| `segment_ids` | both | list of constituent segment IDs |
| `n_segments` | both | |
| `anchor_lat`, `anchor_lon`, `anchor_h3` | STAY | running mean of inside segments |
| `envelope_radius_m` | STAY | observed max segment-to-anchor distance |
| `start_lat`/`start_lon`/`end_lat`/`end_lon` | TRANSIT | first / last segment endpoint |
| `displacement_m`, `path_length_m`, `straightness` | TRANSIT | great-circle / sum / ratio |

## The closure rule

**Why.** Behavioural run-length grouping (e.g. "three `MOVE_BRIEF`s in a
row") is brittle to taxonomy choice and produces episodes that don't
correspond to anything a human cares about. A spatial-envelope rule is
anchored to a physical fact (the entity was here vs. going somewhere)
that survives any segment-type taxonomy and generalises across domains.

**How.** Detect stays first; transits are residuals. A stay is a
maximal run of segments whose **endpoints** (both `start_lat/lon` and
`end_lat/lon`) stay within a radius `R_m` of the running anchor
centroid, allowing a grace window of `T_s` outside the envelope before
closure. Anything between two stays is a transit.

**Why endpoint-aware containment, not centroid-only.** The natural
intuition — "is this segment's centroid within `R_m`" — fails for
spatially extended `MOVE` segments. A 3-minute walk that traverses
800 m has a centroid that is one point ≤ `R_m` from itself, so it
would trivially anchor a single-segment "stay" if its duration meets
`min_stay_s`. Endpoints reject this: a segment whose start and end are
far apart can't be inside any small envelope. For stationary segments
where start ≈ end ≈ centroid, the endpoint check is equivalent to the
centroid check.

## Parameters

Three knobs. Two physical, one minimum-duration filter.

### `R_m` — envelope radius (metres)

Defines "here." Smaller than `R_m` = same place; larger and persistent
= departure. Anchor is the running mean of in-envelope segment
centroids, not the first one — handles parking-then-drift-to-bay.

### `T_s` — departure persistence (seconds)

Prevents "drove around the block and came back" from splitting one
stay into two. Also closes an episode when an inter-segment gap
exceeds `T_s`.

### `min_stay_s` — minimum stay duration (seconds)

Filters "slow crawl through a parking lot" being labelled a stay.
Anything shorter is folded into the surrounding `TRANSIT`.

## Algorithm

Two passes per entity trace.

**Pass 1 — stays.** Greedy left-to-right scan, with two qualification
gates: time (`stay_duration ≥ min_stay_s`) AND space
(`max_observed_radius ≤ R_m`).

```
i = 0
while i < n:
    anchor = centroid(seg[i])
    last_inside = i
    time_outside = 0
    inside_centroids = [centroid(seg[i])]
    inside_endpoints = [start(seg[i]), end(seg[i])]
    for j in i+1 .. n-1:
        if start_ts[j] - end_ts[j-1] > T_s:   # trace gap closes the stay
            break
        d = max(
            haversine(anchor, start(seg[j])),
            haversine(anchor, end(seg[j])),
        )
        if d <= R_m:
            inside_centroids.append(centroid(seg[j]))
            inside_endpoints += [start(seg[j]), end(seg[j])]
            anchor = mean(inside_centroids)   # running update
            last_inside = j
            time_outside = 0
        else:
            time_outside += seg[j].duration_s
            if time_outside >= T_s:
                break                         # stay ends at last_inside
    max_obs_radius = max(haversine(anchor, p) for p in inside_endpoints)
    stay_duration = end_ts[last_inside] - start_ts[i]
    # Both gates must hold: a single spatially-extended segment whose
    # endpoints exceed R fails the radius gate even if duration qualifies.
    if stay_duration >= min_stay_s and max_obs_radius <= R_m:
        emit STAY(i .. last_inside, anchor, max_obs_radius)
        i = last_inside + 1
    else:
        i += 1
```

**Pass 2 — transits.** Each maximal run of segments not claimed by
Pass 1 becomes one `TRANSIT`. Time gaps inside the trace exceeding
`T_s` split the transit at the gap.

## Edge cases

1. **Trace shorter than `min_stay_s` overall.** Result: one `TRANSIT`
   covering everything. Correct — no evidence of a stay.
2. **Boundary oscillation (drifts in/out at R±ε).** The running-mean
   anchor + grace window jointly absorb this. The final
   `envelope_radius_m` reflects the max observed reach; a downstream
   consumer can flag stays where this exceeds `R_m` as "loose."
3. **Long stationary segment (device offline / dwelling).** Already
   aggregated upstream. One segment = one stay if it meets
   `min_stay_s`. No special case.
4. **Trace gaps (device offline).** A gap ≥ `T_s` between consecutive
   `end_ts` and next `start_ts` closes the current episode regardless
   of envelope. Prevents a multi-day device outage from being absorbed
   into a stay.
5. **Drive-in-circles-around-depot.** Running centroid anchors at
   depot; all loops fit within `R_m`; one large `STAY` emitted.
6. **Pass-through a known destination without stopping.** Stay-duration
   threshold not met → folded into `TRANSIT`.
7. **Two adjacent stays at distinct sub-locations within `R_m` of each
   other.** Resolved by `R_m` choice. Documented as the calibration
   trade-off.

## Usage

```python
from trajkit.episode import detect_episodes, EpisodeParams

episodes = detect_episodes(segments_df, EpisodeParams(R_m=200, T_s=300, min_stay_s=180))
```

## Not in this layer

- Per-segment vectorisation — [`embed`](embed.md).
- Similarity search — [`compare`](compare.md).
- Place clustering across entities (resolving "this anchor is the same
  warehouse as that other entity's anchor") — user concern.
