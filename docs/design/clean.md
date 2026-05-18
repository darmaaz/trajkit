# `trajkit.clean`

## Purpose

Take raw spatial-temporal pings and produce a quality-flagged, normalized
stream the rest of the pipeline can trust. Real-world GPS data carries
duplicates, drift clusters, gap-following speed outliers, and device
faults that ping repeatedly without updating coordinates. This is the
only layer that knows about those pathologies; downstream layers assume
their input is already clean.

## Assumptions

- Input matches `PingsSchema` (see [`schemas.md`](schemas.md)).
- Single entity per call. Multi-entity input is a user error.
- Sorted by `ts` (asserted, never silently fixed).
- Coordinates are WGS84.
- Thresholds are user-decided. Defaults are documented inline on
  `CleanParams`; users override per call for their domain.

## Architecture

One pure function:

```python
clean(pings_df: pd.DataFrame, params: CleanParams) -> pd.DataFrame
```

Steps:

1. Assert single-entity, sorted-by-ts.
2. Compute `dt_seconds`, `displacement_m` from consecutive rows.
3. Derive `speed_ms`, `bearing_deg`.
4. Set `is_duplicate` for identical-coord consecutive pings.
5. Assign `quality_flag` from a fixed precedence:
   `DEVICE_FAULT > SPEED_OUTLIER > GAP_FOLLOWS > DRIFT > VALID`.

   Rationale for `GAP_FOLLOWS > DRIFT`: a ping with `dt_seconds` larger
   than `gap_threshold_min` has unreliable derived kinematics — the
   `displacement_m` spans an unobserved interval, the apparent
   `speed_ms` is "displacement / a long time" rather than physically
   meaningful. Drift heuristics ("tiny movement at near-zero speed")
   only apply when inter-ping spacing is normal. Without this ordering,
   a multi-hour gap with small implied displacement gets stamped DRIFT,
   the segmenter sees no gap boundary, and segments grow across the
   unobserved interval — concretely surfaced on real Geolife data
   where a 70-minute offline period appeared as a single 4310-s `MOVE`
   segment.

`CleanParams` is a frozen Pydantic v2 model with `extra='forbid'`.

## Efficiency

- Linear in N pings. All operations vectorised over numpy arrays except
  the final flag-precedence pass (vectorised via `np.select`).
- Memory: one entity's frame plus a constant number of derived columns;
  no row-wise iteration.
- No spatial index. POI / road joins are out of scope here — they
  belong in `embed` feature plugins.

## Usage

```python
from trajkit.clean import clean, CleanParams

clean_df = clean(pings_df, CleanParams())
```

## Deliverable

- [x] `clean(pings_df, params) -> pd.DataFrame`. Output validates against
      the `CleanedPings` schema in `schemas.md`.
- [x] `CleanParams` Pydantic v2 model — frozen, `extra='forbid'`.
- [x] Synthetic-fixture tests covering: pristine input, mid-trace gap,
      drift cluster, speed outlier, duplicate run, naive ts (raises),
      reversed (lon, lat) (raises at schema validation).

## Not in this layer

- Trajectory segmentation — [`segment`](segment.md).
- POI / road / spatial enrichment — [`embed`](embed.md) feature plugins.
- Similarity / anomaly scoring on cleaned data — [`compare`](compare.md).
