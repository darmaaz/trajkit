"""End-to-end Geolife integration: cross-domain validation gate.

Reads N Geolife users' trajectories, runs the trajkit pipeline with the
pedestrian preset, and prints a summary at each stage.

Usage::

    python examples/geolife/run.py /path/to/Geolife/Data --users 5

The expected directory layout (from Microsoft's distribution) is::

    /path/to/Geolife/Data/
    ├── 000/Trajectory/*.plt
    ├── 001/Trajectory/*.plt
    └── ...

See ``examples/geolife/README.md`` for the download URL and instructions.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trajkit.baselines import BaselineParams, fit_baselines
from trajkit.compare import build_index, search
from trajkit.embed import baseline_zscores
from trajkit.runner import RunParams, process

# Make the local reader importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent))
from reader import discover_users, read_user  # noqa: E402

WIDTH = 72


def _banner(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {text}")
    print("=" * WIDTH)


def _step(n: int, total: int, label: str) -> None:
    print()
    print(f"[{n}/{total}] {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--sink",
        type=Path,
        default=None,
        help="Output directory (defaults to a temp dir cleaned on exit)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"error: data_dir does not exist: {data_dir}", file=sys.stderr)
        return 1

    _banner(f"trajkit × Geolife — cross-domain integration ({data_dir})")

    # ── Step 1: discover users + load trajectories ──────────────────
    _step(1, 6, "Discovering and loading Geolife users")
    available = discover_users(data_dir)
    print(f"  Found {len(available)} users with Trajectory/ subdirs")

    if args.user_list:
        chosen = [u.strip() for u in args.user_list.split(",")]
    else:
        chosen = available[: args.users]
    print(f"  Selected: {chosen}")

    frames: list[pd.DataFrame] = []
    for uid in chosen:
        t0 = time.monotonic()
        sub = read_user(data_dir / uid, entity_id=uid)
        elapsed = time.monotonic() - t0
        print(f"    user={uid}  pings={len(sub):>8d}  read={elapsed:.2f}s")
        if not sub.empty:
            frames.append(sub)

    if not frames:
        print("error: no pings loaded; aborting", file=sys.stderr)
        return 2
    pings = pd.concat(frames, ignore_index=True)
    print(f"  TOTAL: {len(pings)} pings across {len(frames)} entities")

    # ── Step 2: run the pipeline ────────────────────────────────────
    _step(2, 6, "Running pipeline (pedestrian preset)")
    sink = args.sink if args.sink else Path(tempfile.mkdtemp(prefix="geolife_run_"))
    sink.mkdir(parents=True, exist_ok=True)
    params = RunParams.from_preset("pedestrian")
    t0 = time.monotonic()
    report = process(pings, sink, params, n_workers=1)
    elapsed = time.monotonic() - t0
    status = "✓" if report.succeeded else "✗"
    print(f"  {status} elapsed={elapsed:.1f}s  completed={report.n_completed}  "
          f"skipped={report.n_skipped_existing}")
    if not report.succeeded:
        print(
            f"  failed: {report.failed_entity}/{report.failed_stage}: "
            f"{report.failed_reason}",
            file=sys.stderr,
        )
        return 3
    print(f"  sink={sink}")

    # ── Step 3: stage summary ───────────────────────────────────────
    _step(3, 6, "Stage outputs")
    summary = _summarize_stages(sink, chosen)
    print(summary.to_string(index=False))

    # ── Step 4: similarity search ───────────────────────────────────
    _step(4, 6, "Similarity search over segment vectors")
    seg_vec = _read_all_partitions(sink / "embed_segments")
    if len(seg_vec) == 0:
        print("  no segments produced — skipping similarity search")
    else:
        vectors = np.vstack(
            [np.asarray(v, dtype=np.float32) for v in seg_vec["vector"]]
        )
        ids = seg_vec["id"].astype(str).tolist()
        index = build_index(vectors, ids, metric="cosine")
        query_idx = len(ids) // 2
        hits = search(index, vectors[query_idx], k=5)
        print(f"  Corpus: {len(ids)} segments × {vectors.shape[1]} dims")
        print(f"  Query: '{ids[query_idx]}' (top 5):")
        for h in hits:
            mark = "  ← query" if h.id == ids[query_idx] else ""
            print(f"    rank={h.rank}  score={h.score:.4f}  id={h.id}{mark}")

    # ── Step 5: cohort baselines + z-scores ─────────────────────────
    _step(5, 6, "Cohort baselines + z-scores")
    try:
        baselines = fit_baselines(
            sink / "segment",
            cohort_keys=["entity_id"],
            metrics=["duration_s", "path_length_m", "mean_speed_ms"],
            params=BaselineParams(min_cohort_n=5, min_global_n=5),
        )
        print(f"  Computed {len(baselines)} baseline rows")
        if len(baselines) > 0:
            print(baselines.head(8).to_string(index=False).replace("\n", "\n    "))
            segments_df = _read_all_partitions(sink / "segment")
            zscored = baseline_zscores(
                segments_df, baselines, cohort_keys=["entity_id"]
            )
            z_cols = [c for c in zscored.columns if c.endswith("_z")]
            if z_cols:
                summary = zscored[z_cols].describe().loc[["mean", "std", "min", "max"]]
                print()
                print(summary.round(2).to_string().replace("\n", "\n    "))
    except Exception as exc:
        print(f"  (baselines stage hit an issue: {type(exc).__name__}: {exc})")

    # ── Step 6: stay locations sanity check ─────────────────────────
    _step(6, 6, "STAY anchor locations (top recurring places)")
    eps = _read_all_partitions(sink / "episode")
    stays = eps[eps["episode_type"] == "STAY"]
    print(f"  {len(stays)} STAY episodes detected across {len(chosen)} users")
    if len(stays) > 0:
        # Per-user count of distinct anchor h3 cells (a proxy for distinct places)
        per_user = stays.groupby("entity_id")["anchor_h3"].nunique()
        print("  Distinct STAY anchors per user (h3 res 9):")
        for eid, n in per_user.items():
            user_stays = (stays["entity_id"] == eid).sum()
            print(f"    user={eid}  stays={user_stays:>4d}  distinct_anchors={n}")

    _banner("✅  Geolife pipeline ran end-to-end on real cross-domain data.")
    return 0


# ── Helpers ─────────────────────────────────────────────────────────


def _summarize_stages(sink: Path, entity_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in ("clean", "segment", "episode", "embed_segments", "embed_episodes"):
        for eid in entity_ids:
            path = sink / stage / f"entity_id={eid}" / "data.parquet"
            if not path.exists():
                rows.append(
                    {"stage": stage, "entity_id": eid, "rows": 0, "summary": "(missing)"}
                )
                continue
            df = pd.read_parquet(path)
            rows.append(
                {
                    "stage": stage,
                    "entity_id": eid,
                    "rows": len(df),
                    "summary": _summary_for(stage, df),
                }
            )
    return pd.DataFrame(rows)


def _summary_for(stage: str, df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "(empty)"
    if stage == "clean":
        flags = df["quality_flag"].value_counts().to_dict()
        return f"flags={flags}"
    if stage == "segment":
        types = df["segment_type"].value_counts().to_dict()
        return f"types={types}"
    if stage == "episode":
        types = df["episode_type"].value_counts().to_dict()
        return f"types={types}"
    if stage in ("embed_segments", "embed_episodes"):
        first = np.asarray(df["vector"].iloc[0], dtype=np.float32)
        return f"vector_dim={first.shape[0]}"
    return ""


def _read_all_partitions(stage_dir: Path) -> pd.DataFrame:
    parts = sorted(stage_dir.glob("entity_id=*/data.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


if __name__ == "__main__":
    sys.exit(main())
