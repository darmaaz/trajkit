"""End-to-end pipeline run on Microsoft Geolife trajectories.

Reads N users, runs the pipeline per user with pedestrian-tuned parameters,
builds a FAISS index over all segment vectors, and runs a similarity query
for sanity. No multiprocessing, no on-disk persistence — everything happens
in memory.

Usage::

    python examples/geolife/run.py /path/to/Geolife/Data --users 5

Layout (from Microsoft's distribution)::

    /path/to/Geolife/Data/
    ├── 000/Trajectory/*.plt
    ├── 001/Trajectory/*.plt
    └── ...

See ``examples/geolife/README.md`` for the download URL.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trajkit.clean import CleanParams, clean
from trajkit.compare import build_index, search
from trajkit.embed import EmbedParams, embed_segments
from trajkit.episode import EpisodeParams, detect_episodes
from trajkit.segment import SegmentParams, aggregate_segments, segment

# Make the local reader importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent))
from reader import discover_users, read_user  # noqa: E402

WIDTH = 72

# Pedestrian-shape calibration for Geolife.
PEDESTRIAN_SEGMENT = SegmentParams(
    stop_speed_kmh=1.0, resume_speed_kmh=3.0, max_stop_displacement_m=50.0
)
PEDESTRIAN_EPISODE = EpisodeParams(R_m=30.0, T_s=120.0, min_stay_s=120.0)
PEDESTRIAN_EMBED = EmbedParams(spatial_bounds=(39.5, 40.5, 115.5, 117.5))


def _banner(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {text}")
    print("=" * WIDTH)


def _step(n: int, total: int, label: str) -> None:
    print()
    print(f"[{n}/{total}] {label}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end trajkit pipeline run on Microsoft Geolife "
        "trajectories with pedestrian-tuned parameters.",
        epilog=(
            "Expected data layout (from Microsoft's distribution):\n"
            "  data_dir/\n"
            "  ├── 000/Trajectory/*.plt\n"
            "  ├── 001/Trajectory/*.plt\n"
            "  └── ...\n\n"
            "See examples/geolife/README.md for the download URL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to Geolife Data directory (contains 000/, 001/, ...)",
    )
    parser.add_argument(
        "--users", type=int, default=5, help="Number of users to process"
    )
    parser.add_argument(
        "--user-list",
        type=str,
        default=None,
        help="Comma-separated user IDs (overrides --users)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"error: data_dir does not exist: {data_dir}", file=sys.stderr)
        return 1

    _banner(f"trajkit × Geolife — end-to-end pipeline ({data_dir})")

    # ── Step 1: discover + load users ───────────────────────────────
    _step(1, 4, "Discovering and loading users")
    available = discover_users(data_dir)
    print(f"  Found {len(available)} users with Trajectory/ subdirs")

    if args.user_list:
        chosen = [u.strip() for u in args.user_list.split(",")]
    else:
        chosen = available[: args.users]
    print(f"  Selected: {chosen}")

    per_user_pings: dict[str, pd.DataFrame] = {}
    for uid in chosen:
        t0 = time.monotonic()
        sub = read_user(data_dir / uid, entity_id=uid)
        elapsed = time.monotonic() - t0
        print(f"    user={uid}  pings={len(sub):>8d}  read={elapsed:.2f}s")
        if not sub.empty:
            per_user_pings[uid] = sub

    if not per_user_pings:
        print("error: no pings loaded; aborting", file=sys.stderr)
        return 2

    # ── Step 2: pipeline per user ───────────────────────────────────
    _step(2, 4, "Running pipeline per user")
    all_segments: list[pd.DataFrame] = []
    all_episodes: list[pd.DataFrame] = []
    all_vectors: list[np.ndarray] = []
    all_ids: list[str] = []

    t0 = time.monotonic()
    for uid, pings in per_user_pings.items():
        cleaned = clean(pings, CleanParams())
        segs = aggregate_segments(segment(cleaned, PEDESTRIAN_SEGMENT), PEDESTRIAN_SEGMENT)
        eps = detect_episodes(segs, PEDESTRIAN_EPISODE)
        vectors, ids = embed_segments(segs, PEDESTRIAN_EMBED)
        all_segments.append(segs)
        all_episodes.append(eps)
        all_vectors.append(vectors)
        all_ids.extend(ids)
        print(f"    user={uid}  segments={len(segs):>5d}  episodes={len(eps):>4d}")
    elapsed = time.monotonic() - t0
    print(f"  elapsed: {elapsed:.1f}s")

    segments_df = pd.concat(all_segments, ignore_index=True)
    episodes_df = pd.concat(all_episodes, ignore_index=True)
    matrix = np.vstack(all_vectors).astype(np.float32) if all_vectors else np.zeros((0, 0))

    # ── Step 3: stage summary ───────────────────────────────────────
    _step(3, 4, "Stage summary")
    print("  segments by type:")
    print(
        segments_df["segment_type"]
        .value_counts()
        .to_frame("count")
        .to_string()
        .replace("\n", "\n    ")
    )
    print()
    print("  episodes by type:")
    print(
        episodes_df["episode_type"]
        .value_counts()
        .to_frame("count")
        .to_string()
        .replace("\n", "\n    ")
    )

    # ── Step 4: similarity search ───────────────────────────────────
    _step(4, 4, "Similarity search over segment vectors")
    if matrix.size == 0:
        print("  no segments — skipping")
    else:
        index = build_index(matrix, all_ids, metric="cosine")
        query_idx = len(all_ids) // 2
        hits = search(index, matrix[query_idx], k=5)
        print(f"  Corpus: {len(all_ids)} segments × {matrix.shape[1]} dims")
        print(f"  Query : '{all_ids[query_idx]}' (top 5):")
        for h in hits:
            mark = "  ← query" if h.id == all_ids[query_idx] else ""
            print(f"    rank={h.rank}  score={h.score:.4f}  id={h.id}{mark}")

    _banner("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
