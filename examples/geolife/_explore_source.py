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
    stop_speed_kmh=1.0, resume_speed_kmh=3.0, max_stop_displacement_m=50.0
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
# ## 4. Pings → Segments → Episodes
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
# ## 5. Segment-type breakdown
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
# ## 6. Map: segments along the actual ping path
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
# ## 7. Anatomy of one segment
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
# ## 7b. A bearing-driven boundary
#
# A pair of consecutive `MOVE` segments with no stop or gap between them
# can only have been split by the bearing detector. The map shows the
# transition; the R plot below shows the circular concentration in both
# distance windows plunging through the entry threshold at the boundary
# — that drop is why the split fired.

# %%
from trajkit.segment._segment import _circular_r_over_distance  # noqa: E402

TARGET_SEG = "000_seg_00876"
ctx_pings_before = 30   # last N pings of preceding segment, for R-curve context
ctx_pings_after  = 120  # first N pings of the target segment, for R-curve context

target_mask = per_ping_segmented["segment_id"] == TARGET_SEG
if not target_mask.any():
    raise RuntimeError(f"{TARGET_SEG} not found in per-ping frame")
target_idx0 = int(per_ping_segmented.index[target_mask].min())
target_idx_last = int(per_ping_segmented.index[target_mask].max())
prev_seg_id = per_ping_segmented["segment_id"].iloc[target_idx0 - 1]
prev_seg_type = per_ping_segmented["segment_type"].iloc[target_idx0 - 1]
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
# ## 8. Stay duration distribution

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
# ## 9. STAY anchors

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
# ## 10. Episode-type breakdown

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
# ## 11. Anatomy of one episode
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
# ## 12. TRANSIT map
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
# ## 13. Episode similarity — *"find me trips like this trip"*
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
    """3·D + 2 episode-level scalars; L2-normalised."""
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

transit_episodes = episodes[episodes["episode_type"] == "TRANSIT"].sort_values(
    "duration_s"
)
if len(transit_episodes) > 0:
    query_ep_id = "ep_000_00036"
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
    print("No TRANSIT episodes — episode similarity demo skipped.")

m_ep_sim if len(transit_episodes) > 0 else None

# %% [markdown]
# Behaviourally similar hits (similar duration + path length + n_segments)
# that are geographically scattered are the signal we want — the embedding
# captured *what* not *where*.

# %% [markdown]
# ## Closing
#
# Each section above ties one design decision to its visible effect on a
# real trace. For the full design rationale — why circular-R over rolling
# bearing deltas, why distance-based windows, why a dual qualification
# gate on episodes — see `docs/design/{segment,episode,embed,compare}.md`.
