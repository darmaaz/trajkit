# Episode Layer — Design Sketch

## Goal
Introduce a scale between **segment** (atomic typed unit, seconds–minutes) and **mission** (origin→destination journey, hours). An episode is the smallest grouping of segments that corresponds to a coherent operational unit a human would name — a *visit* or a *leg* — independent of domain.

## Input
A segments DataFrame with at minimum, per row:

| col | type | notes |
|---|---|---|
| `segment_id` | str | unique per trace |
| `entity_id` | str | vehicle/vessel/animal/device |
| `start_ts`, `end_ts` | datetime | ordered within entity |
| `start_lat`, `start_lon`, `end_lat`, `end_lon` | float | WGS84 |
| `segment_type` | enum | MOVE, MOVE_BRIEF, STOP_BRIEF, STOP_DWELL |
| `duration_s` | float | end - start |

Optionally a per-segment embedding `(N, D)` (numpy, row-aligned) for episode embedding.

## Output
An episodes DataFrame:

| col | type | for type | notes |
|---|---|---|---|
| `episode_id` | str | both | `ep_<entity>_<seq>` |
| `entity_id` | str | both | |
| `episode_type` | enum | both | STAY \| TRANSIT |
| `start_ts`, `end_ts`, `duration_s` | — | both | |
| `segment_ids` | list[str] | both | constituents |
| `n_segments` | int | both | |
| `anchor_lat`, `anchor_lon`, `anchor_h3` | — | STAY | running mean of inside segments |
| `envelope_radius_m` | float | STAY | observed max segment-to-anchor distance |
| `start_anchor_*`, `end_anchor_*` | — | TRANSIT | first/last segment centroid |
| `displacement_m`, `path_length_m`, `straightness` | float | TRANSIT | great-circle / sum / ratio |

Plus optional `(M, D')` episode embedding matrix, row-aligned with episodes.

## The closure rule

**WHY.** Behavioral run-length grouping ("three MOVE_BRIEFs in a row") is brittle to taxonomy choice and produces episodes that don't correspond to anything a human cares about. A spatial-envelope rule is anchored to a physical fact (the entity was here vs. going somewhere) that survives any segment-type taxonomy and generalizes across domains.

**HOW.** Detect stays first; transits are residuals. A stay is a maximal run of segments whose centroids stay within a radius `R` of the running anchor centroid, allowing a grace window of `T` outside the envelope before closure. Anything between two stays is a transit.

## Parameters

Three knobs. Two physical, one minimum-duration filter.

### `R` — envelope radius (meters)
- **WHY.** Defines "here." Smaller than R = same place; larger and persistent = departure.
- **HOW.** User sets per scale class, or the calibration helper suggests it (see below). Anchor is the running mean of in-envelope segment centroids, not the first one — handles parking-then-drift-to-bay.

### `T` — departure persistence (seconds)
- **WHY.** Prevents "drove around the block and came back" from splitting one stay into two.
- **HOW default.** `5 × median(segment.duration_s)`. Rationale: an excursion shorter than ~5 atomic segments is noise; longer is a real departure. Robust to fleet-mix; no manual tuning needed.

### `min_stay_s` — minimum stay duration (seconds)
- **WHY.** Filters "slow crawl through a parking lot" being labeled a stay.
- **HOW default.** `3 × median(STOP_BRIEF.duration_s)`. Anything shorter is folded into the surrounding TRANSIT.

## Algorithm

Two passes per entity trace.

**Pass 1 — stays.** Greedy left-to-right scan:

```
i = 0
while i < n:
    anchor = centroid(seg[i])
    last_inside = i
    time_outside = 0
    inside_centroids = [centroid(seg[i])]
    for j in i+1 .. n-1:
        d = haversine(anchor, centroid(seg[j]))
        if d <= R:
            inside_centroids.append(centroid(seg[j]))
            anchor = mean(inside_centroids)   # running update
            last_inside = j
            time_outside = 0
        else:
            time_outside += seg[j].duration_s
            if time_outside >= T:
                break                          # stay ends at last_inside
    stay_duration = end_ts[last_inside] - start_ts[i]
    if stay_duration >= min_stay_s:
        emit STAY(i .. last_inside, anchor, max_obs_radius)
        i = last_inside + 1
    else:
        i += 1                                 # not a stay, retry from next
```

**Pass 2 — transits.** Each maximal run of segments not claimed by Pass 1 = one TRANSIT. Time gaps inside the trace exceeding `T` split the transit at the gap.

## Edge cases (specific, not hand-waved)

1. **Trace shorter than `min_stay_s` overall.** Result: one TRANSIT covering everything. Correct — we don't have evidence of a stay.
2. **Boundary oscillation (drifts in/out at R±ε).** The running-mean anchor + grace window `T` jointly absorb this. We still emit a final `envelope_radius_m` reflecting the max observed reach; a downstream consumer can flag stays where this exceeds R as "loose."
3. **Long stationary segment (device offline / dwelling).** Already aggregated upstream. One segment = one stay if it meets `min_stay_s`. No special case.
4. **Stale-position devices.** Out of scope; assumed handled by step 4b before this layer runs.
5. **Trace gaps (device offline).** A gap ≥ `T` between consecutive `end_ts` and next `start_ts` closes the current episode regardless of envelope. Prevents a multi-day device outage from being absorbed into a stay.
6. **Drive-in-circles-around-depot.** Running centroid anchors at depot; all loops fit within R; one large STAY emitted. Correct.
7. **Pass-through a known destination without stopping.** Stay-duration threshold not met → folded into TRANSIT. The destination cluster (built fleet-wide upstream) still exists; episodes don't pretend to be it.
8. **Two adjacent stays at distinct sub-locations within R of each other.** Resolved by `R` choice. Documented as the calibration trade-off; the helper biases toward smaller R to favor splitting over merging.

## Episode embedding

**WHY.** The point of the layer is to make episodes the unit of similarity search. Pooling segment vectors gives a fixed-width per-episode vector independent of `n_segments`.

**HOW.** Given segment embeddings `V ∈ R^(N,D)`:

- For each episode, take its rows of `V`.
- Pool to fixed width: `concat(mean, std, max-by-magnitude)` → `3D`. Captures central tendency, internal variability, and the most extreme single moment.
- Append episode-level scalars: `[log1p(duration_s), log1p(path_length_m), n_segments, type_one_hot(2)]` → `+5`.
- L2-normalize the concat.

For D=81 segment vectors → 243 + 5 = 248-dim episode vector. Optionally train a single PCA on episode vectors to compress to 64-dim once; not required for v1.

## Episode similarity

**WHY.** Replaces the current "same-route + closest duration" mission-similarity heuristic with real semantic similarity at a meaningful granularity.

**HOW.** FAISS `IndexFlatIP` over L2-normalized episode vectors. Re-uses the existing `search.py` infrastructure. Optional pre-filter: type (STAY-vs-TRANSIT), entity, time window. No same-route restriction — that was the limitation we're trying to lift.

## Calibration helper

**WHY.** `R` is the one parameter the library cannot pick blind; it depends on the physical scale of the user's domain. We can suggest it from the data instead of asking the user to guess.

**HOW.** For each `STOP_DWELL` segment in a sample of the user's data, compute the 95th-percentile radius of its constituent pings around their centroid (approximated by start/end midpoint when ping-level data isn't available). Output the 75th percentile of those radii as suggested `R`, plus the full distribution histogram. Rationale: percentile-based statistics survive long tails; 75th leans toward separation over over-merging.

## Scale presets

Defaults shipped for the obvious classes. User can override or run calibration.

```python
SCALE_PRESETS = {
    "logistics_vehicle": dict(R_m=200,  T_s=300,  min_stay_s=180),
    "maritime_vessel":   dict(R_m=1500, T_s=1800, min_stay_s=1800),
    "wildlife_mammal":   dict(R_m=100,  T_s=600,  min_stay_s=300),
    "mobile_phone":      dict(R_m=100,  T_s=300,  min_stay_s=600),
    "pedestrian":        dict(R_m=30,   T_s=120,  min_stay_s=120),
}
```

## Where this slots in

```
... step 9 build_indices (segment embeddings + FAISS)
   ↓
   step 9b detect_episodes      ← NEW, this layer
   ↓
   step 9c embed_episodes       ← NEW, this layer
   ↓
   step 10 build_missions (now reads episodes, not raw segments, for grouping)
   ↓
   step 11 destination_index
```

Missions become "sequences of episodes bracketed by long stays." Routes become "(start_stay_anchor, end_stay_anchor) pairs." The mission similarity endpoint pivots from `route_id`-filter + duration-rank to episode-vector FAISS search with optional filters.

## What "complete" looks like

A v1 deliverable that ships, not a research project. Checklist:

- [ ] `episode_layer/episode.py` — `detect_episodes(segments_df, R_m, T_s, min_stay_s) -> episodes_df`. Pure function, no I/O.
- [ ] `episode_layer/embed.py` — `embed_episodes(episodes_df, segment_vectors, segments_df) -> (np.ndarray, list[episode_id])`.
- [ ] `episode_layer/calibrate.py` — `suggest_radius(segments_df, percentile=0.75) -> (float, hist)`.
- [ ] `episode_layer/presets.py` — `SCALE_PRESETS` dict.
- [ ] `episode_layer/tests/` — synthetic-trace unit tests covering: simple stay, simple transit, depot-loop stay, boundary oscillation, gap-split, sub-min-stay rejection, two-stays-too-close. ≥ 80% line coverage on `episode.py`.
- [ ] `scripts/run_episodes.sh` — local runner: reads `segments_missions.parquet`, writes `episodes.parquet` + `episode_vectors.npz`.
- [ ] `api/main.py` additions: `GET /episodes/{id}`, `GET /episodes/{id}/similar?k=N&type=STAY|TRANSIT`. No removal of existing endpoints in v1.
- [ ] Validation run on at least one non-fleet dataset (Geolife or open AIS slice). Document: did the calibrated `R` produce sane stays? Yes/no with histogram.
- [ ] `episode_layer/README.md` — 1-page quickstart with the input/output contract above.

Out of scope for v1:
- Learned (contrastive) episode embeddings. The pooled baseline lands first.
- Replacing the mission layer. Episodes are added alongside; missions keep their current contract until the migration step is its own design doc.
- Streaming / incremental episode detection. Batch only in v1.
