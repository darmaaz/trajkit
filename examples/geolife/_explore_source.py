# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # trajkit × Geolife
#
# A walk through the end-to-end pipeline on one Geolife user. Each
# section anchors a design decision in the code to something visible on a
# real GPS trace.
#
# The two headline ideas:
#
# 1. **Segmentation that splits on motion state and on sustained
#    direction change.** Hysteresis avoids flicker; the multi-scale
#    circular-R bearing detector splits long `MOVE` segments where the
#    trajectory actually turns.
# 2. **Similarity search over per-segment vectors.** With trips pooled
#    from their constituent segments, "find me trips like this one"
#    becomes a cosine query against a FAISS index — returning hits that
#    match on behaviour shape, not geography.
#
# The flow is **pings → segments → episodes → similarity**.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local Geolife reader
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "examples/geolife")
from reader import read_user  # noqa: E402

from trajkit.clean import CleanParams, clean  # noqa: E402
from trajkit.compare import build_index, search  # noqa: E402
from trajkit.embed import EmbedParams, embed_segments  # noqa: E402
from trajkit.episode import EpisodeParams, detect_episodes  # noqa: E402
from trajkit.segment import SegmentParams, aggregate_segments  # noqa: E402
from trajkit.segment import segment as run_segment  # noqa: E402

GEOLIFE_DATA = Path(os.path.expanduser("~/.cache/trajkit/Geolife Trajectories 1.3/Data"))
USER_ID = "000"
TIME_WINDOW_DAYS = 7

# Pedestrian-scale calibration for Geolife.
SEG_PARAMS = SegmentParams(
    stop_speed_kmh=1.0,
    resume_speed_kmh=3.0,
    max_stop_displacement_m=50.0,
    dwell_threshold_min=3.0,
)
EP_PARAMS = EpisodeParams(R_m=30.0, T_s=120.0, min_stay_s=120.0)
EMBED_PARAMS = EmbedParams(spatial_bounds=(39.5, 40.5, 115.5, 117.5))
CLEAN_PARAMS = CleanParams()

# %% [markdown]
# ## 1. Load Geolife user

# %%
all_pings = read_user(GEOLIFE_DATA / USER_ID, entity_id=USER_ID)
print(f"User {USER_ID}: {len(all_pings):,} pings spanning "
      f"{all_pings['ts'].min().date()} → {all_pings['ts'].max().date()}")

day_counts = all_pings["ts"].dt.floor("D").value_counts().sort_values(ascending=False)
busiest_day = day_counts.index[0]
window_start = busiest_day - pd.Timedelta(days=TIME_WINDOW_DAYS // 2)
window_end = window_start + pd.Timedelta(days=TIME_WINDOW_DAYS)
mask = (all_pings["ts"] >= window_start) & (all_pings["ts"] < window_end)
pings = all_pings.loc[mask].reset_index(drop=True)
print(f"Window: {window_start.date()} → {window_end.date()}  "
      f"({len(pings):,} pings)")

# %% [markdown]
# ## 2. Run the pipeline
#
# The L1 functions compose explicitly — no orchestrator, no on-disk
# persistence. Everything stays in memory for the rest of the notebook.

# %%
cleaned_per_ping = clean(pings, CLEAN_PARAMS)
per_ping_segmented = run_segment(cleaned_per_ping, SEG_PARAMS)
segments = aggregate_segments(per_ping_segmented, SEG_PARAMS)
episodes = detect_episodes(segments, EP_PARAMS)
seg_vectors, seg_ids = embed_segments(segments, EMBED_PARAMS)

# Pack the segment vectors into a VectorsSchema-shaped DataFrame so later
# cells can lookup vectors by id without juggling raw arrays.
seg_vectors_df = pd.DataFrame(
    {
        "id": pd.Series(seg_ids, dtype="string"),
        "entity_id": pd.Series([USER_ID] * len(seg_ids), dtype="string"),
        "vector": [seg_vectors[i].astype(np.float32) for i in range(len(seg_ids))],
    }
)
print(f"Pings (cleaned):       {len(per_ping_segmented):>7,}")
print(f"Segments (aggregated): {len(segments):>7,}")
print(f"Episodes:              {len(episodes):>7,} "
      f"(STAY={(episodes['episode_type']=='STAY').sum()}, "
      f"TRANSIT={(episodes['episode_type']=='TRANSIT').sum()})")

# %% [markdown]
# ## 3. Pings → Segments → Episodes
#
# A three-level hierarchy:
#
# * **Ping** — one GPS observation. Raw input.
# * **Segment** — a contiguous run of pings sharing one of
#   `MOVE`, `MOVE_BRIEF`, `STOP_BRIEF`, `STOP_DWELL`.
# * **Episode** — a contiguous run of segments sharing one of
#   `STAY` or `TRANSIT`, decided by the spatial-envelope rule.
#
# One real example below.

# %%
# Show one episode's nested structure
sample_ep = episodes[episodes["episode_type"] == "TRANSIT"].sort_values(
    "n_segments", ascending=False
).iloc[0]
print(
    f"Episode {sample_ep['episode_id']}  "
    f"({sample_ep['episode_type']}, {sample_ep['duration_s']:.0f}s, "
    f"{int(sample_ep['n_segments'])} segments, "
    f"{float(sample_ep['path_length_m']):.0f}m path)"
)
print()
print("├── segments (one per row, time-ordered):")
for seg_id in sample_ep["segment_ids"]:
    seg = segments.loc[segments["segment_id"] == seg_id].iloc[0]
    n_pings = (per_ping_segmented["segment_id"] == seg_id).sum()
    print(
        f"│   ├── {seg_id}  "
        f"{seg['segment_type']:<11s}  {seg['duration_s']:>5.0f}s  "
        f"path={seg['path_length_m']:>6.0f}m  pings={n_pings}"
    )
print("│")
print("└── (each segment in turn is a contiguous run of pings)")

# %% [markdown]
# One TRANSIT episode wraps several typed segments, each itself a run of
# dozens of pings. Ping-level is too granular to ask "what kind of trip
# is this?"; segment-level is too noisy; episode-level is the right grain.

# %% [markdown]
# ## 4. Segment-type breakdown
#
# Atomic count + total duration share per segment type.

# %%
type_counts = segments["segment_type"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
type_counts.plot.bar(ax=ax[0], color=["#1f77b4", "#ff7f0e", "#ffbb78", "#d62728"])
ax[0].set_title("Segment count by type")
ax[0].set_ylabel("Count")
ax[0].tick_params(axis="x", rotation=0)
fractions = (type_counts / type_counts.sum() * 100).round(1)
for i in range(len(type_counts)):
    ax[0].text(i, type_counts.iloc[i], f" {fractions.iloc[i]}%", ha="center", va="bottom")
type_total_duration = segments.groupby("segment_type")["duration_s"].sum() / 3600
type_total_duration.plot.bar(ax=ax[1], color=["#1f77b4", "#ff7f0e", "#ffbb78", "#d62728"])
ax[1].set_title("Total duration by type (hours)")
ax[1].set_ylabel("Hours")
ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Hysteresis at work
#
# The state machine has a *dead zone* between `stop_speed_kmh` and
# `resume_speed_kmh`. While speed is in the dead zone, the inferred
# state stays put. The plot below zooms into a boundary region scored
# for non-trivial dead-zone activity. The speed line repeatedly dips
# into the dead zone without flipping the state band underneath — that
# stability is the whole point of using two thresholds.

# %%
_per_ping = per_ping_segmented.copy()
_per_ping["_is_stop"] = _per_ping["segment_type"].str.startswith("STOP")
_per_ping["_state_change"] = _per_ping["_is_stop"] != _per_ping["_is_stop"].shift(
    1, fill_value=_per_ping["_is_stop"].iloc[0]
)
stop_ms_ = SEG_PARAMS.stop_speed_kmh * (1000.0 / 3600.0)
resume_ms_ = SEG_PARAMS.resume_speed_kmh * (1000.0 / 3600.0)
ds = _per_ping["speed_ms"].fillna(0.0).to_numpy()
in_dead = (ds > stop_ms_) & (ds < resume_ms_)

_WIN = 120
_csum = np.concatenate([[0], np.cumsum(in_dead.astype(int))])
_best_idx, _best_score = None, -1
for b in _per_ping.index[_per_ping["_state_change"]]:
    lo = max(0, int(b) - _WIN // 2)
    hi = min(len(_per_ping), int(b) + _WIN // 2)
    score = int(in_dead[lo:hi].sum())
    if score > _best_score:
        _best_score, _best_idx = score, int(b)

if _best_idx is None:
    print("no state transitions in this slice — hysteresis demo skipped")
else:
    lo = max(0, _best_idx - _WIN // 2)
    hi = min(len(_per_ping), _best_idx + _WIN // 2)
    view = _per_ping.iloc[lo:hi].reset_index(drop=True)
    speed_kmh = view["speed_ms"].fillna(0.0) * 3.6
    elapsed_s = (view["ts"] - view["ts"].iloc[0]).dt.total_seconds()
    state = view["_is_stop"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(elapsed_s, speed_kmh, color="#1f77b4", lw=1.5, label="speed (km/h)")
    ax.axhspan(SEG_PARAMS.stop_speed_kmh, SEG_PARAMS.resume_speed_kmh,
               color="#9999cc", alpha=0.15, label="dead zone")
    ax.axhline(SEG_PARAMS.stop_speed_kmh, color="#d62728", lw=1, ls="--",
               label=f"stop_speed = {SEG_PARAMS.stop_speed_kmh} km/h")
    ax.axhline(SEG_PARAMS.resume_speed_kmh, color="#2ca02c", lw=1, ls="--",
               label=f"resume_speed = {SEG_PARAMS.resume_speed_kmh} km/h")
    prev_s, band_start = state[0], 0
    for k in range(1, len(state) + 1):
        if k == len(state) or state[k] != prev_s:
            colour = "#fbb4b4" if prev_s else "#bbe1bb"
            ax.axvspan(elapsed_s.iloc[band_start],
                       elapsed_s.iloc[min(k, len(state) - 1)],
                       ymax=0.06, color=colour, alpha=0.7)
            if k < len(state):
                band_start, prev_s = k, state[k]
    ax.set_ylim(0, max(float(speed_kmh.max()) * 1.1, SEG_PARAMS.resume_speed_kmh * 1.5))
    ax.set_xlabel("seconds since window start")
    ax.set_ylabel("speed (km/h)")
    ax.set_title("Hysteresis: speed weaves through the dead zone; bottom band = inferred state")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()
    print(f"window: {_best_score} dead-zone pings out of {hi - lo}; "
          f"state transitions: {int(view['_state_change'].sum())}")

# %% [markdown]
# ## 6. Why `GAP_FOLLOWS` outranks `DRIFT`
#
# The cleaning layer's flag precedence is
# `DEVICE_FAULT > SPEED_OUTLIER > GAP_FOLLOWS > DRIFT > VALID`. The
# non-obvious choice is `GAP_FOLLOWS > DRIFT`. A ping after a long gap
# has a tiny implied displacement and a near-zero effective speed only
# because the position was sampled minutes or hours apart — the drift
# heuristic would mis-claim it as drift, and the segmenter would then
# see no gap boundary and grow segments straight across the missing
# interval.
#
# Below: for every `GAP_FOLLOWS` ping in the trace, what would the
# naive "drift first" rule have said?

# %%
_drift_ms = CLEAN_PARAMS.drift_speed_kmh * (1000.0 / 3600.0)
_gap_mask = (cleaned_per_ping["quality_flag"] == "GAP_FOLLOWS").to_numpy()

# Re-derive raw kinematics that drift would have inspected, ignoring the
# gap-edge nulling that clean() performs.
_lat = cleaned_per_ping["lat"].to_numpy()
_lon = cleaned_per_ping["lon"].to_numpy()
_prev_lat = np.concatenate([[np.nan], _lat[:-1]])
_prev_lon = np.concatenate([[np.nan], _lon[:-1]])
_dt = cleaned_per_ping["dt_seconds"].to_numpy()
_EARTH = 6_371_000.0
_phi1, _phi2 = np.radians(_prev_lat), np.radians(_lat)
_dphi = np.radians(_lat - _prev_lat)
_dlmb = np.radians(_lon - _prev_lon)
_a = np.sin(_dphi / 2) ** 2 + np.cos(_phi1) * np.cos(_phi2) * np.sin(_dlmb / 2) ** 2
_raw_disp = 2 * _EARTH * np.arcsin(np.sqrt(_a))
_raw_disp = np.where(np.isnan(_raw_disp), 0.0, _raw_disp)
with np.errstate(divide="ignore", invalid="ignore"):
    _raw_speed_ms = np.where(_dt > 0, _raw_disp / _dt, np.inf)

_would_be_drift = (
    (_raw_disp > 0)
    & (_raw_disp < CLEAN_PARAMS.drift_radius_m)
    & (_raw_speed_ms < _drift_ms)
)
_caught_by_precedence = int((_gap_mask & _would_be_drift).sum())
_total_gap = int(_gap_mask.sum())
print(
    f"GAP_FOLLOWS pings: {_total_gap}\n"
    f"  would have been DRIFT under naive ordering: {_caught_by_precedence}"
    f"  ({100 * _caught_by_precedence / max(_total_gap, 1):.1f}%)"
)

_sample_rows = []
for idx in np.where(_gap_mask)[0][:6]:
    _sample_rows.append({
        "row": int(idx),
        "dt_h": round(float(_dt[idx]) / 3600.0, 2),
        "raw_disp_m": round(float(_raw_disp[idx]), 1),
        "raw_speed_kmh": round(float(_raw_speed_ms[idx]) * 3.6, 3),
        "naive_DRIFT_would_fire": bool(
            _raw_disp[idx] > 0
            and _raw_disp[idx] < CLEAN_PARAMS.drift_radius_m
            and _raw_speed_ms[idx] < _drift_ms
        ),
    })
print()
print(pd.DataFrame(_sample_rows).to_string(index=False))

# %% [markdown]
# Most gap-edge pings have tiny implied displacement (the device pinged
# from nearly the same place after a multi-hour silence). Without the
# precedence rule those would be labelled `DRIFT`, no gap boundary
# would fire in the segmenter, and a segment would grow straight across
# the unobserved interval. The precedence is what keeps that from
# happening.

# %% [markdown]
# ## 7. Map: segments along the actual ping path
#
# Two views of the same segmentation. The first map traces the **actual
# GPS pings** of each segment — segments respect the natural shape of the
# trajectory because they are runs of consecutive pings. The second map
# collapses each segment to a single **start → end straight line**, the
# representation downstream consumers (embeddings, similarity search) work
# from. Comparing the two shows how much detail the segment abstraction
# discards, and where that abstraction is faithful vs lossy.

# %%
COLOUR = {
    "MOVE": "#1f77b4",
    "MOVE_BRIEF": "#ff7f0e",
    "STOP_BRIEF": "#ffbb78",
    "STOP_DWELL": "#d62728",
}

legend_html = """
<div style='position: fixed; top:10px; right:10px; z-index:9999;
            background:white; padding:8px; border:1px solid #888; font-size:13px;'>
  <b>Segment type</b><br>
  <span style='color:#1f77b4'>━━━</span> MOVE<br>
  <span style='color:#ff7f0e'>━━━</span> MOVE_BRIEF<br>
  <span style='color:#ffbb78'>━━━</span> STOP_BRIEF<br>
  <span style='color:#d62728'>━━━</span> STOP_DWELL<br>
</div>
"""

center_lat = float(pings["lat"].median())
center_lon = float(pings["lon"].median())

# ── Map 6a: full ping path per segment ──────────────────────────────
m_segments = folium.Map(
    location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron"
)

seg_meta = segments.set_index("segment_id")[["segment_type", "duration_s"]]
for seg_id, group in per_ping_segmented.groupby("segment_id", sort=False):
    if len(group) < 2:
        continue
    if seg_id not in seg_meta.index:
        continue
    meta = seg_meta.loc[seg_id]
    seg_type = str(meta["segment_type"])
    coords = list(zip(group["lat"].astype(float), group["lon"].astype(float), strict=True))
    folium.PolyLine(
        locations=coords,
        color=COLOUR.get(seg_type, "#888888"),
        weight=3 if seg_type.startswith("MOVE") else 5,
        opacity=0.6,
        tooltip=(
            f"{seg_id} • {seg_type} • {float(meta['duration_s']):.0f}s • "
            f"pings={len(group)}"
        ),
    ).add_to(m_segments)

m_segments.get_root().html.add_child(folium.Element(legend_html))
m_segments.save("segments_map.html")
m_segments

# %% [markdown]
# ### Map 6b: simplified — each segment as a start → end straight line
#
# Same colour encoding. A straight chord between `(start_lat, start_lon)`
# and `(end_lat, end_lon)` for every segment. Curvy MOVE segments shrink
# to short chords (low straightness); near-stationary STOP segments
# collapse to a point or near-point.

# %%
m_segments_simplified = folium.Map(
    location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron"
)
for _, seg in segments.iterrows():
    seg_type = str(seg["segment_type"])
    folium.PolyLine(
        locations=[
            (float(seg["start_lat"]), float(seg["start_lon"])),
            (float(seg["end_lat"]), float(seg["end_lon"])),
        ],
        color=COLOUR.get(seg_type, "#888888"),
        weight=3 if seg_type.startswith("MOVE") else 5,
        opacity=0.6,
        tooltip=(
            f"{seg['segment_id']} • {seg_type} • "
            f"{float(seg['duration_s']):.0f}s • "
            f"path={float(seg['path_length_m']):.0f}m • "
            f"chord={float(seg['displacement_m']):.0f}m • "
            f"straightness={float(seg['straightness']):.2f}"
        ),
    ).add_to(m_segments_simplified)

m_segments_simplified.get_root().html.add_child(folium.Element(legend_html))
m_segments_simplified.save("segments_simplified_map.html")
m_segments_simplified

# %% [markdown]
# ## 8. Anatomy of one segment
#
# A segment is a sequence of pings sharing one motion type, not a
# straight line. `displacement_m` is the bird's-eye gap;
# `path_length_m` is the sum of inter-ping displacements; their ratio
# gives `straightness ∈ [0, 1]`.

# %%
# Pick a MOVE segment with a representative ping count
move_segments = segments[segments["segment_type"] == "MOVE"].copy()
move_segments["__ping_count"] = move_segments["segment_id"].map(
    per_ping_segmented["segment_id"].value_counts()
)
sample_seg = move_segments.sort_values("__ping_count", ascending=False).iloc[
    len(move_segments) // 4  # pick around the 25th-percentile-from-largest
]
sample_pings = per_ping_segmented[
    per_ping_segmented["segment_id"] == sample_seg["segment_id"]
].sort_values("ts")
print(
    f"Segment: {sample_seg['segment_id']} ({sample_seg['segment_type']})"
)
print(f"  pings:                 {len(sample_pings)}")
print(f"  duration_s:            {sample_seg['duration_s']:.1f}")
print(f"  start → end straight:  {sample_seg['displacement_m']:.1f} m")
print(f"  actual path length:    {sample_seg['path_length_m']:.1f} m")
print(f"  straightness:          {sample_seg['straightness']:.3f}  "
      f"(displacement / path_length)")

# Map zoomed to this segment
seg_center_lat = float(sample_pings["lat"].mean())
seg_center_lon = float(sample_pings["lon"].mean())
m_one_seg = folium.Map(
    location=[seg_center_lat, seg_center_lon], zoom_start=15, tiles="cartodbpositron"
)
# Actual ping path
folium.PolyLine(
    locations=list(zip(sample_pings["lat"].astype(float),
                       sample_pings["lon"].astype(float),
                       strict=True)),
    color=COLOUR[str(sample_seg["segment_type"])],
    weight=4,
    opacity=0.8,
    tooltip=f"actual path: {sample_seg['path_length_m']:.0f}m through "
             f"{len(sample_pings)} pings",
).add_to(m_one_seg)
# Straight-line displacement (dashed grey, for comparison)
folium.PolyLine(
    locations=[(float(sample_seg["start_lat"]), float(sample_seg["start_lon"])),
               (float(sample_seg["end_lat"]), float(sample_seg["end_lon"]))],
    color="#666666", weight=2, opacity=0.6, dash_array="6,6",
    tooltip=f"start → end straight line: {sample_seg['displacement_m']:.0f}m",
).add_to(m_one_seg)
# Markers
folium.CircleMarker(
    location=[float(sample_seg["start_lat"]), float(sample_seg["start_lon"])],
    radius=6, color="#2ca02c", fill=True, fill_opacity=0.9,
    tooltip="segment start",
).add_to(m_one_seg)
folium.CircleMarker(
    location=[float(sample_seg["end_lat"]), float(sample_seg["end_lon"])],
    radius=6, color="#9467bd", fill=True, fill_opacity=0.9,
    tooltip="segment end",
).add_to(m_one_seg)
m_one_seg.save("segment_anatomy.html")
m_one_seg

# %% [markdown]
# The coloured polyline is the actual pings; the grey dashed line is the
# straight-line displacement. A straight commute scores `straightness ≈
# 1.0`; a winding walk lands closer to 0.3.

# %% [markdown]
# ## 9. A bearing-driven boundary
#
# Programmatically pick a pair of consecutive `MOVE` segments joined at
# one ping interval with no `GAP_FOLLOWS` flag. Neither the state-change
# detector nor the gap detector could have fired here — the only thing
# that could have split them is the bearing detector. The map shows the
# transition; the R plot below shows the circular-concentration curves
# (short and long windows) plunging through the entry threshold at the
# boundary.

# %%
from trajkit.segment._segment import _circular_r_over_distance  # noqa: E402

# Find a real MOVE → MOVE boundary: adjacent same-segment-type segments,
# joined at one ping interval, where the join ping is NOT GAP_FOLLOWS.
# That means neither the state-change nor the gap detector fired — the
# only thing that could have split them is the bearing detector.
sorted_segs = segments.sort_values("start_ts").reset_index(drop=True)
ctx_pings_before = 30
ctx_pings_after = 120
best = None
for i in range(len(sorted_segs) - 1):
    a, b = sorted_segs.iloc[i], sorted_segs.iloc[i + 1]
    if a["segment_type"] != "MOVE" or b["segment_type"] != "MOVE":
        continue
    if a["n_pings"] < ctx_pings_before or b["n_pings"] < ctx_pings_after:
        continue
    b_first_idx = int(per_ping_segmented.index[
        per_ping_segmented["segment_id"] == b["segment_id"]
    ].min())
    if per_ping_segmented["quality_flag"].iloc[b_first_idx] == "GAP_FOLLOWS":
        continue
    score = float(a["path_length_m"]) + float(b["path_length_m"])
    if best is None or score > best["score"]:
        best = {"prev_id": a["segment_id"], "next_id": b["segment_id"],
                "b_first_idx": b_first_idx, "score": score}
if best is None:
    raise RuntimeError("no MOVE → MOVE boundary found in this slice")

TARGET_SEG = best["next_id"]
prev_seg_id = best["prev_id"]
prev_seg_type = "MOVE"
target_mask = per_ping_segmented["segment_id"] == TARGET_SEG
target_idx0 = best["b_first_idx"]
target_idx_last = int(per_ping_segmented.index[target_mask].max())
target_meta = segments.loc[segments["segment_id"] == TARGET_SEG].iloc[0]

# Two scopes:
#   map_slice — full target segment + context tail of the previous segment,
#               so the visual extent matches what's drawn on the section 6 map.
#   r_slice   — narrow window around the boundary, so the R curve isn't
#               diluted by averaging over the whole segment.
map_slice = per_ping_segmented.iloc[
    max(0, target_idx0 - ctx_pings_before) : target_idx_last + 1
].reset_index(drop=True)
r_slice = per_ping_segmented.iloc[
    max(0, target_idx0 - ctx_pings_before) : target_idx0 + ctx_pings_after
].reset_index(drop=True)

print(
    f"Boundary: end of {prev_seg_id} ({prev_seg_type}) → "
    f"start of {TARGET_SEG} ({target_meta['segment_type']})"
)
print(
    f"  target segment: {int(target_meta['n_pings'])} pings, "
    f"{float(target_meta['duration_s']):.0f}s, "
    f"path={float(target_meta['path_length_m']):.0f}m"
)

# Map: the full target segment + tail of the previous segment.
seg_center_lat = float(map_slice["lat"].astype(float).mean())
seg_center_lon = float(map_slice["lon"].astype(float).mean())
m_seg_bearing = folium.Map(
    location=[seg_center_lat, seg_center_lon], zoom_start=12, tiles="cartodbpositron"
)
for sid, group in map_slice.groupby("segment_id", sort=False):
    if len(group) < 2:
        continue
    seg_type = str(group["segment_type"].iloc[0])
    folium.PolyLine(
        locations=list(zip(group["lat"].astype(float), group["lon"].astype(float), strict=True)),
        color=COLOUR.get(seg_type, "#888888"),
        weight=4 if seg_type.startswith("MOVE") else 6,
        opacity=0.8,
        tooltip=f"{sid} • {seg_type} • {len(group)} pings",
    ).add_to(m_seg_bearing)
boundary_lat = float(map_slice.loc[map_slice["segment_id"] == TARGET_SEG, "lat"].iloc[0])
boundary_lon = float(map_slice.loc[map_slice["segment_id"] == TARGET_SEG, "lon"].iloc[0])
folium.CircleMarker(
    location=[boundary_lat, boundary_lon],
    radius=7, color="black", fill=True, fill_opacity=0.9,
    tooltip=f"boundary: first ping of {TARGET_SEG}",
).add_to(m_seg_bearing)
m_seg_bearing.get_root().html.add_child(folium.Element(legend_html))
m_seg_bearing.save("segment_bearing_anatomy.html")
m_seg_bearing

# %%
# R curves: narrow window around the boundary only.
moving = r_slice["segment_type"].str.startswith("MOVE").to_numpy()
bearing_arr = r_slice["bearing_deg"].to_numpy(dtype=float)
valid_b = moving & ~np.isnan(bearing_arr)
disp = r_slice["displacement_m"].fillna(0.0).to_numpy(dtype=float)
disp_motion = np.where(moving, disp, 0.0)
cum_dist_ctx = np.cumsum(disp_motion)
r_short = _circular_r_over_distance(
    cum_dist_ctx, bearing_arr, valid_b,
    SEG_PARAMS.bearing_window_short_m, SEG_PARAMS.bearing_window_min_pings,
)
r_long = _circular_r_over_distance(
    cum_dist_ctx, bearing_arr, valid_b,
    SEG_PARAMS.bearing_window_long_m, SEG_PARAMS.bearing_window_min_pings,
)
boundary_local = int(r_slice.index[r_slice["segment_id"] == TARGET_SEG].min())
boundary_dist = float(cum_dist_ctx[boundary_local])

fig, ax_r = plt.subplots(1, 1, figsize=(11, 4))
ax_r.plot(cum_dist_ctx, r_short, label=f"R (short {SEG_PARAMS.bearing_window_short_m:.0f} m)", color="#1f77b4")
ax_r.plot(cum_dist_ctx, r_long, label=f"R (long {SEG_PARAMS.bearing_window_long_m:.0f} m)", color="#ff7f0e")
ax_r.axhline(SEG_PARAMS.bearing_r_enter, ls="--", color="red", alpha=0.6, lw=1)
ax_r.axhline(SEG_PARAMS.bearing_r_exit, ls="--", color="green", alpha=0.6, lw=1)
ax_r.text(cum_dist_ctx[-1], SEG_PARAMS.bearing_r_enter, " r_enter", color="red", va="center", fontsize=9)
ax_r.text(cum_dist_ctx[-1], SEG_PARAMS.bearing_r_exit, " r_exit", color="green", va="center", fontsize=9)
ax_r.axvline(boundary_dist, color="black", ls=":", alpha=0.7, label=f"boundary @ {boundary_dist:.0f} m")
ax_r.set_xlabel("cumulative motion distance (m)")
ax_r.set_ylabel("R (circular concentration)")
ax_r.set_ylim(0, 1.05)
ax_r.set_title(f"R curves around the start of {TARGET_SEG}")
ax_r.legend(fontsize=9, loc="lower right")
plt.tight_layout(); plt.show()

# %% [markdown]
# R drops sharply at the boundary because the windows straddle a heading
# change: bearings on either side point in different directions, and the
# vector-mean registers them as spread out. A purely-bearing boundary
# (MOVE → MOVE with no stop in between) looks the same in R, just
# without the coincident state-change vote.

# %% [markdown]
# ## 10. Stay duration distribution

# %%
stay_minutes = episodes.loc[episodes["episode_type"] == "STAY", "duration_s"] / 60.0
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(stay_minutes, bins=np.logspace(np.log10(2), np.log10(stay_minutes.max() + 1), 30))
ax.set_xscale("log")
ax.set_xlabel("Stay duration (minutes, log scale)")
ax.set_ylabel("Count")
ax.set_title(f"STAY episode duration distribution (n={len(stay_minutes)})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. STAY anchors

# %%
stays = episodes[episodes["episode_type"] == "STAY"].copy()
m_stays = folium.Map(
    location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron"
)
for _, ep in stays.iterrows():
    duration_min = float(ep["duration_s"]) / 60.0
    radius = float(np.clip(np.log1p(duration_min) * 2.5, 4, 25))
    folium.CircleMarker(
        location=[float(ep["anchor_lat"]), float(ep["anchor_lon"])],
        radius=radius,
        color="#d62728",
        fill=True,
        fill_opacity=0.5,
        tooltip=(
            f"{duration_min:.1f} min • envelope_radius={ep['envelope_radius_m']:.1f} m"
        ),
    ).add_to(m_stays)
m_stays.save("stays_map.html")
m_stays

# %% [markdown]
# ## 12. Episode dual gate — what the radius check catches
#
# The episode detector applies two qualification gates: a time gate
# (`duration ≥ min_stay_s`) and a space gate (`max observed radius from
# anchor ≤ R_m`). A centroid-only check would let a spatially extended
# single-segment `MOVE` qualify as its own "stay" — its centroid is one
# point ≤ R from itself. The space gate rejects those.
#
# Below: how many single segments in this slice would have qualified
# under a centroid-only rule but get rejected by the endpoint-reach
# check.

# %%
def _endpoint_reach(row: pd.Series) -> float:
    cl = (float(row["start_lat"]) + float(row["end_lat"])) / 2.0
    cn = (float(row["start_lon"]) + float(row["end_lon"])) / 2.0
    R = 6_371_000.0
    p1, p2 = np.radians(cl), np.radians([row["start_lat"], row["end_lat"]])
    dphi = np.radians([row["start_lat"] - cl, row["end_lat"] - cl])
    dlmb = np.radians([row["start_lon"] - cn, row["end_lon"] - cn])
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(np.max(2 * R * np.arcsin(np.sqrt(a))))


_rejected = segments[
    (segments["duration_s"] >= EP_PARAMS.min_stay_s)
    & segments.apply(lambda r: _endpoint_reach(r) > EP_PARAMS.R_m, axis=1)
].copy()
print(
    f"Single segments that pass duration_s ≥ {EP_PARAMS.min_stay_s:.0f} but fail "
    f"endpoint reach ≤ {EP_PARAMS.R_m:.0f} m: {len(_rejected)}\n"
)
if len(_rejected) > 0:
    _rejected = _rejected.sort_values("duration_s", ascending=False)
    _cols = ["segment_id", "segment_type", "duration_s",
             "path_length_m", "displacement_m", "n_pings"]
    print(_rejected[_cols].head(8).round(1).to_string(index=False))
    print(
        "\nEach of these would have anchored a spurious single-segment stay "
        "under a centroid-only rule. The endpoint-reach gate rejects them."
    )
else:
    print("None found in this slice — the gate didn't reject anything here. "
          "Still useful: tells us this user's traces don't produce endpoint-"
          "stretched candidates at these thresholds.")

# %% [markdown]
# ## 13. Episode-type breakdown

# %%
ep_type_counts = episodes["episode_type"].value_counts()
ep_total_duration_h = episodes.groupby("episode_type")["duration_s"].sum() / 3600
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ep_type_counts.plot.bar(ax=ax[0], color=["#d62728", "#1f77b4"])
ax[0].set_title("Episode count by type")
ax[0].set_ylabel("Count")
ax[0].tick_params(axis="x", rotation=0)
for i, count in enumerate(ep_type_counts):
    ax[0].text(i, count, f" {count}", ha="center", va="bottom")
ep_total_duration_h.plot.bar(ax=ax[1], color=["#d62728", "#1f77b4"])
ax[1].set_title("Total duration by type (hours)")
ax[1].set_ylabel("Hours")
ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Anatomy of one episode
#
# One TRANSIT with its constituent segments coloured individually —
# the segment boundaries inside a single journey become visible.

# %%
def _episode_path_polyline(
    fmap: folium.Map,
    episode: pd.Series,
    rank: int = 0,
    is_query: bool = False,
) -> None:
    """Draw an episode's constituent segments through actual pings."""
    seg_ids = list(episode["segment_ids"])
    for sid in seg_ids:
        seg_pings = per_ping_segmented[per_ping_segmented["segment_id"] == sid]
        if len(seg_pings) < 2:
            continue
        seg_meta_row = segments.loc[segments["segment_id"] == sid]
        if len(seg_meta_row) == 0:
            continue
        seg_type = str(seg_meta_row.iloc[0]["segment_type"])
        if is_query:
            colour = "#000000"
            weight = 5.0
            opacity = 0.95
        else:
            colour = COLOUR.get(seg_type, "#1f77b4")
            weight = 3.0
            opacity = float(max(0.35, 0.9 - rank * 0.12))
        folium.PolyLine(
            locations=list(zip(seg_pings["lat"].astype(float),
                               seg_pings["lon"].astype(float),
                               strict=True)),
            color=colour,
            weight=weight,
            opacity=opacity,
            tooltip=(
                f"ep={episode['episode_id']} rank={rank} "
                f"seg={sid} {seg_type}"
            ),
        ).add_to(fmap)


sample_transit = episodes[episodes["episode_type"] == "TRANSIT"].sort_values(
    "n_segments", ascending=False
).iloc[0]
print(
    f"Episode {sample_transit['episode_id']}  "
    f"({sample_transit['episode_type']}, {sample_transit['duration_s']:.0f}s, "
    f"{int(sample_transit['n_segments'])} segments, "
    f"{float(sample_transit['path_length_m']):.0f}m path)"
)
seg_type_in_ep = []
for sid in sample_transit["segment_ids"]:
    s = segments.loc[segments["segment_id"] == sid].iloc[0]
    seg_type_in_ep.append(s["segment_type"])
print("  Segment-type composition:",
      pd.Series(seg_type_in_ep).value_counts().to_dict())

# Centre on this episode
ep_pings_subset = per_ping_segmented[
    per_ping_segmented["segment_id"].isin(sample_transit["segment_ids"])
]
ep_center_lat = float(ep_pings_subset["lat"].mean())
ep_center_lon = float(ep_pings_subset["lon"].mean())

m_one_ep = folium.Map(
    location=[ep_center_lat, ep_center_lon], zoom_start=14, tiles="cartodbpositron"
)
_episode_path_polyline(m_one_ep, sample_transit, rank=0, is_query=False)
folium.CircleMarker(
    location=[float(sample_transit["start_lat"]), float(sample_transit["start_lon"])],
    radius=7, color="#2ca02c", fill=True, fill_opacity=0.9,
    tooltip="episode start",
).add_to(m_one_ep)
folium.CircleMarker(
    location=[float(sample_transit["end_lat"]), float(sample_transit["end_lon"])],
    radius=7, color="#9467bd", fill=True, fill_opacity=0.9,
    tooltip="episode end",
).add_to(m_one_ep)
m_one_ep.get_root().html.add_child(folium.Element(legend_html))
m_one_ep.save("episode_anatomy.html")
m_one_ep

# %% [markdown]
# Each colour change along the journey is a segment boundary —
# stretches of `MOVE` interrupted by short `STOP_BRIEF` blocks (traffic
# lights) or `MOVE_BRIEF` chunks (parking-lot manoeuvring).

# %% [markdown]
# ## 15. TRANSIT map
#
# All TRANSIT episodes drawn as their constituent segments' actual paths
# — i.e., just the journeys.

# %%
m_transits = folium.Map(
    location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron"
)
transits = episodes[episodes["episode_type"] == "TRANSIT"]
for _, ep in transits.iterrows():
    _episode_path_polyline(m_transits, ep, rank=0, is_query=False)
m_transits.get_root().html.add_child(folium.Element(legend_html))
m_transits.save("transits_map.html")
m_transits

# %% [markdown]
# ## 16. Episode similarity — *"find me trips like this trip"*
#
# `embed_segments` produces one vector per segment. Episode-level
# pooling is a user-side choice; here we concat
# `[mean, std, max-by-magnitude]` across an episode's segments and
# append two episode-level scalars. Cosine similarity over the result
# matches on behaviour shape, not geography.
#
# The query is rendered in **black**, the top-5 hits in colour with
# decreasing opacity by rank.

# %%
def _pool_episode(seg_vec_subset: np.ndarray, ep_row: pd.Series) -> np.ndarray:
    """3·D + 2 episode-level scalars; L2-normalised.

    A deliberately minimal pool. Richer schemes (per-type one-hots,
    path-length scalars, learned projections) are user-side decisions.
    """
    if len(seg_vec_subset) == 0:
        return np.zeros(seg_vec_subset.shape[1] * 3 + 2, dtype=np.float32)
    mean = seg_vec_subset.mean(axis=0)
    std = seg_vec_subset.std(axis=0)
    abs_argmax = np.abs(seg_vec_subset).argmax(axis=0)
    cols = np.arange(seg_vec_subset.shape[1])
    max_by_mag = seg_vec_subset[abs_argmax, cols]
    scalars = np.array(
        [np.log1p(max(float(ep_row["duration_s"]), 0.0)),
         1.0 if ep_row["episode_type"] == "TRANSIT" else 0.0],
        dtype=np.float32,
    )
    v = np.concatenate([mean, std, max_by_mag, scalars]).astype(np.float32)
    norm = float(np.linalg.norm(v))
    return v / max(norm, 1e-8)

_seg_id_to_row = {sid: i for i, sid in enumerate(seg_ids)}
_pooled_rows = []
_pooled_ids = []
for _, ep_row in episodes.iterrows():
    member_rows = [
        _seg_id_to_row[sid] for sid in ep_row["segment_ids"] if sid in _seg_id_to_row
    ]
    if not member_rows:
        continue
    _pooled_rows.append(_pool_episode(seg_vectors[member_rows], ep_row))
    _pooled_ids.append(str(ep_row["episode_id"]))

ep_vectors = np.vstack(_pooled_rows).astype(np.float32)
ep_ids = _pooled_ids

# Pick a query programmatically: the median-duration TRANSIT with at least 3
# constituent segments. Hard-coded IDs are brittle across parameter changes.
m_ep_sim = None
transit_episodes = episodes[
    (episodes["episode_type"] == "TRANSIT") & (episodes["n_segments"] >= 3)
].sort_values("duration_s").reset_index(drop=True)
if len(transit_episodes) > 0:
    query_ep_id = str(transit_episodes.iloc[len(transit_episodes) // 2]["episode_id"])
    query_idx = ep_ids.index(query_ep_id)
    ep_index = build_index(ep_vectors, ep_ids, metric="cosine")
    ep_hits = search(ep_index, ep_vectors[query_idx], k=6)
    print(f"Query: {query_ep_id}")
    hits_table = pd.DataFrame(
        [{"rank": h.rank, "episode_id": h.id, "score": h.score} for h in ep_hits]
    )
    hits_table = hits_table.merge(
        episodes[
            ["episode_id", "episode_type", "duration_s", "n_segments",
             "path_length_m", "displacement_m"]
        ],
        on="episode_id",
    )
    print(
        hits_table[
            ["rank", "episode_id", "score", "episode_type", "duration_s",
             "n_segments", "path_length_m"]
        ].round(3).to_string(index=False)
    )

    m_ep_sim = folium.Map(
        location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron"
    )
    for _, row in hits_table.iterrows():
        ep_row = episodes.loc[episodes["episode_id"] == row["episode_id"]].iloc[0]
        is_query = row["rank"] == 0
        if ep_row["episode_type"] == "STAY":
            folium.CircleMarker(
                location=[float(ep_row["anchor_lat"]), float(ep_row["anchor_lon"])],
                radius=10 if is_query else 6,
                color="#000000" if is_query else "#1f77b4",
                fill=True,
                fill_opacity=0.7 if is_query else 0.5,
                tooltip=(
                    f"rank={row['rank']} score={row['score']:.4f} STAY "
                    f"({float(ep_row['duration_s']) / 60:.0f} min)"
                ),
            ).add_to(m_ep_sim)
        else:
            _episode_path_polyline(
                m_ep_sim, ep_row, rank=int(row["rank"]), is_query=is_query
            )
    m_ep_sim.save("episode_similarity_map.html")
else:
    print("No multi-segment TRANSIT episodes in this slice — similarity demo skipped.")

m_ep_sim

# %% [markdown]
# Behaviourally similar hits (similar duration + path length + n_segments)
# that are geographically scattered are the signal we want — the embedding
# captured *what* not *where*.

# %% [markdown]
# ## 17. Embedding sanity — PCA of segment vectors
#
# If the 32-d segment vectors carry meaningful type information, the
# four classes should at least partially separate in 2-D. The PCA below
# is a quick sanity check; clean separation would say the recipe is
# doing real work, total overlap would say it's mostly noise.

# %%
from sklearn.decomposition import PCA  # noqa: E402

_pca = PCA(n_components=2)
_xy = _pca.fit_transform(seg_vectors)
_types_arr = segments.set_index("segment_id").loc[seg_ids]["segment_type"].to_numpy()

fig, ax = plt.subplots(figsize=(8, 6))
for _t, _c in COLOUR.items():
    _m = _types_arr == _t
    if not _m.any():
        continue
    ax.scatter(_xy[_m, 0], _xy[_m, 1], s=18, alpha=0.55, c=_c,
               label=f"{_t} (n={int(_m.sum())})", edgecolors="none")
ax.set_xlabel(f"PC1 ({_pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({_pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax.set_title("Segment vectors in 2-D PCA — coloured by segment_type")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.show()
print(f"PC1+PC2 explained variance: "
      f"{_pca.explained_variance_ratio_.sum() * 100:.1f}%")

# %% [markdown]
# ## 18. Honest failure modes
#
# Where this run looks weak. Each line below is a concrete calibration
# lead, not a flaw in the algorithm.

# %%
LONG_MOVE_S = 30 * 60  # 30 minutes
_long_moves = segments[
    (segments["segment_type"] == "MOVE") & (segments["duration_s"] > LONG_MOVE_S)
]
print(f"[a] MOVE segments longer than {LONG_MOVE_S // 60:.0f} min: {len(_long_moves)}")
if len(_long_moves) > 0:
    print(_long_moves.sort_values("duration_s", ascending=False)[
        ["segment_id", "duration_s", "path_length_m", "n_pings"]
    ].head(5).round(1).to_string(index=False))
    print("  (calibration lead: bearing windows may be too short for sustained-turn scale)")

_stays_in_eps = episodes[episodes["episode_type"] == "STAY"]
_loose = _stays_in_eps[_stays_in_eps["envelope_radius_m"] >= 0.9 * EP_PARAMS.R_m]
print(f"\n[b] STAYs with envelope_radius ≥ 0.9 × R_m: {len(_loose)} of {len(_stays_in_eps)}")
if len(_loose) > 0:
    print(_loose[["episode_id", "duration_s", "envelope_radius_m", "n_segments"]]
          .head(5).round(1).to_string(index=False))
    print("  (calibration lead: stays at the envelope edge — bigger R merges places, smaller splits them)")

_n_dwell = int((segments["segment_type"] == "STOP_DWELL").sum())
_n_brief = int((segments["segment_type"] == "STOP_BRIEF").sum())
_dwell_share = _n_dwell / max(_n_dwell + _n_brief, 1)
print(
    f"\n[c] STOP_DWELL count = {_n_dwell}; STOP_BRIEF count = {_n_brief}; "
    f"STOP_DWELL share of STOP_* = {100 * _dwell_share:.1f}%"
)
if _dwell_share < 0.20:
    print(
        f"  (calibration lead: dwell_threshold_min = {SEG_PARAMS.dwell_threshold_min} "
        "may be too high for walking-cadence dwell durations.)"
    )

_drift_rate = (cleaned_per_ping["quality_flag"] == "DRIFT").mean()
print(f"\n[d] DRIFT flag rate: {100 * _drift_rate:.1f}% of all pings")
if _drift_rate > 0.10:
    print("  (calibration lead: thresholds at 50 m / 1 km/h may flag genuine GPS jitter; retune for this cadence)")

# %% [markdown]
# ## Closing
#
# Each section above ties one design decision to its visible effect on a
# real trace. For the full design rationale — why circular-R over rolling
# bearing deltas, why distance-based windows, why a dual qualification
# gate on episodes — see `docs/design/{segment,episode,embed,compare}.md`.
