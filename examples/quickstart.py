"""trajkit quickstart demo.

Runs the full pipeline end-to-end on synthetic data for three entities
with different motion patterns, then exercises every public surface
(similarity search, cohort baselines, anomaly scoring). Prints a
human-readable summary at each stage.

Usage::

    python examples/quickstart.py

No external data required. Runs in under 10 seconds.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trajkit.baselines import BaselineParams, fit_baselines
from trajkit.compare import anomaly_score, build_index, search
from trajkit.embed import baseline_zscores
from trajkit.runner import RunParams, process
from trajkit.testing import make_pings

WIDTH = 64


def _banner(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {text}")
    print("=" * WIDTH)


def _step(n: int, total: int, label: str) -> None:
    print()
    print(f"[{n}/{total}] {label}")


def main() -> None:
    _banner("trajkit quickstart demo")

    workdir = Path(tempfile.mkdtemp(prefix="trajkit_quickstart_"))
    sink = workdir / "out"

    try:
        # ── Step 1: synthesize a multi-entity batch ────────────────
        _step(1, 7, "Generating synthetic data")
        traces = {
            "v1_stationary": make_pings(
                n=600, entity_id="v1_stationary",
                start_ts="2026-01-01 08:00:00", motion="stationary",
                lat=19.40, lon=-99.20,
            ),
            "v2_walker": make_pings(
                n=600, entity_id="v2_walker",
                start_ts="2026-01-01 08:00:00", motion="linear",
                lat=19.42, lon=-99.20, lat_step=0.0001,  # ~6 km/h
            ),
            "v3_mixed": make_pings(
                n=600, entity_id="v3_mixed",
                start_ts="2026-01-01 08:00:00", motion="stop_then_move",
                lat=19.44, lon=-99.20, lat_step=0.0001,
            ),
        }
        pings = pd.concat(traces.values(), ignore_index=True)
        for eid, df in traces.items():
            print(f"  {eid:18s}  {len(df):4d} pings  motion={_motion_of(eid)}")
        print(f"  {'TOTAL':18s}  {len(pings):4d} pings across {len(traces)} entities")

        # ── Step 2: run the L3 pipeline ────────────────────────────
        _step(2, 7, "Running pipeline (trajkit.runner.process)")
        params = RunParams.from_preset("pedestrian")  # tighter R for synthetic scale
        t0 = time.monotonic()
        report = process(pings, sink, params, n_workers=1)
        elapsed = time.monotonic() - t0
        status = "✓" if report.succeeded else "✗"
        print(f"  {status} succeeded={report.succeeded} elapsed={elapsed:.2f}s")
        print(f"    completed={report.n_completed} skipped={report.n_skipped_existing}")
        print(f"    sink={report.sink_dir}")
        if not report.succeeded:
            raise RuntimeError(
                f"pipeline failed: {report.failed_entity}/{report.failed_stage}: "
                f"{report.failed_reason}"
            )

        # ── Step 3: inspect per-stage outputs ──────────────────────
        _step(3, 7, "Stage outputs (one row per stage × entity)")
        stages_df = _summarize_stages(sink, list(traces.keys()))
        print(stages_df.to_string(index=False))

        # ── Step 4: similarity search over segment vectors ─────────
        _step(4, 7, "Similarity search over segment vectors")
        seg_vec_df = _read_all_partitions(sink / "embed_segments")
        vectors = np.vstack([np.asarray(v, dtype=np.float32) for v in seg_vec_df["vector"]])
        ids = seg_vec_df["id"].astype(str).tolist()
        print(f"  Built corpus: {len(ids)} segments × {vectors.shape[1]} dims")

        index = build_index(vectors, ids, metric="cosine")
        query = vectors[0]
        hits = search(index, query, k=5)
        print(f"  Query: '{ids[0]}' (top 5 hits, cosine):")
        for h in hits:
            self_marker = "  ← query" if h.id == ids[0] else ""
            print(f"    rank={h.rank}  score={h.score:.4f}  id={h.id}{self_marker}")

        # ── Step 5: cohort baselines (pass-2) ──────────────────────
        _step(5, 7, "Cohort baselines (trajkit.baselines.fit_baselines)")
        # Demo uses tiny thresholds because we only have 3 entities × 1-2 segments;
        # production defaults are min_cohort_n=30, min_global_n=10.
        baselines = fit_baselines(
            sink / "segment",
            cohort_keys=["entity_id"],
            metrics=["duration_s", "path_length_m", "mean_speed_ms"],
            params=BaselineParams(min_cohort_n=1, min_global_n=2),
        )
        print(f"  Computed {len(baselines)} baseline rows")
        print("  Sample:")
        print(baselines.to_string(index=False).replace("\n", "\n    "))

        # ── Step 6: apply baselines as z-scores ─────────────────────
        _step(6, 7, "Cohort-aware z-scoring (trajkit.embed.baseline_zscores)")
        segments_df = _read_all_partitions(sink / "segment")
        zscored = baseline_zscores(segments_df, baselines, cohort_keys=["entity_id"])
        z_cols = [c for c in zscored.columns if c.endswith("_z")]
        print(f"  Added z-score columns: {z_cols}")
        if z_cols:
            z_summary = zscored[z_cols].describe().loc[["mean", "std", "min", "max"]]
            print(z_summary.round(2).to_string().replace("\n", "\n    "))
        else:
            print("  (no metrics qualified — increase synthetic data or lower thresholds)")

        # ── Step 7: anomaly scoring ────────────────────────────────
        _step(7, 7, "Anomaly scoring (trajkit.compare.anomaly_score)")
        scores = anomaly_score(vectors, contamination=0.1)
        score_df = pd.DataFrame({"id": ids, "score": scores})
        top = score_df.nlargest(3, "score")
        print("  Top 3 anomalies (more positive = more anomalous):")
        for row in top.itertuples():
            print(f"    score={row.score:+.4f}  id={row.id}")

        # ── Final ──────────────────────────────────────────────────
        _banner("✅  All stages produced sensible output. trajkit v0.1.0 alive.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── Helpers ─────────────────────────────────────────────────────────


def _motion_of(entity_id: str) -> str:
    if "stationary" in entity_id:
        return "stationary"
    if "walker" in entity_id:
        return "linear"
    return "stop_then_move"


def _summarize_stages(sink: Path, entity_ids: list[str]) -> pd.DataFrame:
    """One row per (stage, entity) with row count + a key descriptor."""
    rows: list[dict[str, object]] = []
    for stage in ("clean", "segment", "episode", "embed_segments", "embed_episodes"):
        for eid in entity_ids:
            path = sink / stage / f"entity_id={eid}" / "data.parquet"
            if not path.exists():
                rows.append({"stage": stage, "entity_id": eid, "rows": 0, "summary": "(missing)"})
                continue
            df = pd.read_parquet(path)
            rows.append({
                "stage": stage,
                "entity_id": eid,
                "rows": len(df),
                "summary": _summary_for(stage, df),
            })
    return pd.DataFrame(rows)


def _summary_for(stage: str, df: pd.DataFrame) -> str:
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
        if len(df) == 0:
            return "(empty)"
        first = np.asarray(df["vector"].iloc[0], dtype=np.float32)
        return f"vector_dim={first.shape[0]}"
    return ""


def _read_all_partitions(stage_dir: Path) -> pd.DataFrame:
    """Concatenate every Hive partition under a stage directory."""
    parts = sorted(stage_dir.glob("entity_id=*/data.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


if __name__ == "__main__":
    main()
