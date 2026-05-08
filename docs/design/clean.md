# `trajkit.clean`

## Purpose

Take raw spatial-temporal pings and produce a quality-flagged, normalized
stream the rest of the pipeline can trust. Real-world GPS data carries
duplicates, drift clusters, gap-following speed outliers, and stale-position
devices that ping repeatedly without updating coordinates. This is the only
layer that knows about those pathologies; downstream layers assume their
input is already clean.

## Assumptions

- Input matches `PingsSchema` (see `schemas.md`).
- Single entity per call (L1 invariant). Multi-entity input raises.
- Sorted by `ts` (asserted, never silently fixed).
- Coordinates are WGS84.
- Thresholds are user-decided — defaults shipped via `SCALE_PRESETS` but
  never auto-tuned from the data inside this layer.
- Stale-position handling is opt-in: the user either calls
  `merge_stale_positions` explicitly or asks `clean(..., merge_stale=True)`.

## Architecture

Two pure functions; either may be called without the other.

```python
clean(pings_df: pd.DataFrame, params: CleanParams) -> pd.DataFrame
merge_stale_positions(pings_df: pd.DataFrame, params: StaleMergeParams) -> pd.DataFrame
```

`clean`:
1. Assert single-entity, sorted-by-ts.
2. Compute `dt_seconds`, `displacement_m` from consecutive rows.
3. Derive `speed_ms`, `bearing_deg` if absent.
4. Set `is_duplicate` for identical-coord consecutive pings.
5. Assign `quality_flag` from a fixed precedence:
   `DEVICE_FAULT > SPEED_OUTLIER > DRIFT > GAP_FOLLOWS > VALID`.

`merge_stale_positions`:
1. Detect runs of consecutive identical (lat, lon).
2. Collapse each run to its first ping; record `merge_count`, `run_duration_s`.
3. Re-derive `dt_seconds`, `displacement_m`, `speed_ms`, `bearing_deg`.
4. Re-flag quality on the merged frame.

Both functions are stateless; no learned parameters. `CleanParams` and
`StaleMergeParams` are frozen Pydantic models.

## Efficiency

- Linear in N pings. All operations vectorized over numpy arrays except the
  final flag-precedence pass (vectorized via `np.select`).
- Memory: one entity's frame plus a constant number of derived columns.
  Copy-on-write column appends; no row-wise iteration.
- Target: 1 M pings/sec on the dedup/derive path; 100 K pings/sec when
  `merge_stale_positions` runs (groupby on consecutive runs is the
  bottleneck and is unavoidable).
- No spatial index. POI/road joins are out of scope here — they belong in
  `embed` feature plugins.

## Usage

```python
import trajkit

# L1, single entity
clean_df = trajkit.clean(pings_df, trajkit.CleanParams.from_preset("logistics_vehicle"))

# Optional stale-position pass — user-decided per fleet
if pings_df.attrs.get("provider_known_stale", False):
    clean_df = trajkit.merge_stale_positions(clean_df, trajkit.StaleMergeParams())

# Inside the L3 runner, "clean" is a stage; stale-merge is a parameter on that stage.
```

## Successful deliverable

- [ ] `clean(pings_df, params) -> pd.DataFrame`. Output validates against the
      `Pings (cleaned)` schema in `schemas.md`.
- [ ] `merge_stale_positions(pings_df, params) -> pd.DataFrame`. Output adds
      `merge_count`, `run_duration_s`; downstream behavior identical to
      non-merged cleaned pings otherwise.
- [ ] `CleanParams`, `StaleMergeParams` Pydantic models — frozen, `extra="forbid"`,
      with a `.from_preset(name)` classmethod.
- [ ] Property-based tests asserting: monotonic ts is preserved; coordinate
      ranges preserved; quality flag never `None`; `is_duplicate=True` implies
      `displacement_m == 0`.
- [ ] Synthetic-fixture tests covering: pristine input, mid-trace gap,
      drift cluster, speed outlier, duplicate run, stale-position run,
      naive ts (raises), reversed (lon, lat) (raises at L2).
- [ ] ≥ 80% line coverage.

## Not in this layer

- Trajectory segmentation — `segment`.
- POI / road / spatial enrichment — `embed` feature plugins.
- Anomaly scoring on cleaned data — `compare`.
- Cross-entity baselining — pass-2 (`fit_baselines`).
- Cross-provider format normalization — `iter_entities` `column_map=` parameter.
