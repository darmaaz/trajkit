"""L3 runner: ``process`` orchestrator + per-entity stage worker.

Implements the v0.1.0 contract from LIBRARY.md §5/§7:

* Iterates entities via ``trajkit.io.iter_entities``.
* Applies stages in fixed order, writing Hive-partitioned parquet per
  stage with atomic file-level renames.
* Per-entity multiprocessing pool when the source is a path. In-memory
  DataFrame / Arrow Table sources force ``n_workers=1`` (workers can't
  cheaply re-read in-memory data).
* Resume by skipping entities whose final stage output already exists.
* Per-entity exceptions abort the whole run (no skip-on-failure mode).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from trajkit.clean import clean, merge_stale_positions
from trajkit.embed import embed_episodes, embed_segments
from trajkit.episode import detect_episodes
from trajkit.io import iter_entities
from trajkit.runner._params import RunParams
from trajkit.runner._report import RunReport
from trajkit.segment import aggregate_segments, segment

_logger = logging.getLogger(__name__)

DEFAULT_STAGES: tuple[str, ...] = (
    "clean",
    "segment",
    "episode",
    "embed_segments",
    "embed_episodes",
)
_VALID_STAGES: frozenset[str] = frozenset(DEFAULT_STAGES)


# ── Public entry point ──────────────────────────────────────────────


def process(
    source: str | Path | pa.Table | pd.DataFrame,
    sink_dir: str | Path,
    params: RunParams | None = None,
    *,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    n_workers: int = 1,
) -> RunReport:
    """Run the configured pipeline stages for every entity in ``source``.

    Parameters
    ----------
    source
        Anything ``iter_entities`` accepts: parquet path, Arrow table,
        DataFrame, CSV path.
    sink_dir
        Output root. Each stage writes into ``sink_dir/<stage>/`` as a
        Hive-partitioned ``entity_id=<X>/data.parquet`` layout.
    params
        ``RunParams``; defaults are scale-class agnostic.
    stages
        Subset of ``DEFAULT_STAGES`` in the canonical order. Earlier
        stages whose outputs already exist on disk are read instead of
        re-computed.
    n_workers
        Number of multiprocessing workers. Forced to ``1`` when the
        source is in-memory (DataFrame, Arrow Table) since workers can't
        cheaply re-read in-memory data without pickling.
    """
    p = params if params is not None else RunParams()
    sink = Path(sink_dir)
    sink.mkdir(parents=True, exist_ok=True)

    _validate_stages(stages)

    is_path_source = isinstance(source, str | Path)
    if n_workers > 1 and not is_path_source:
        _logger.warning(
            "process: n_workers=%d ignored — in-memory sources require single-"
            "process iteration. Falling back to n_workers=1.",
            n_workers,
        )
        n_workers = 1

    started = time.monotonic()
    completed: list[str] = []
    skipped_count = 0

    try:
        if n_workers == 1:
            for entity_id, pings_df in iter_entities(source):
                summary = _process_one_entity(
                    entity_id, pings_df, p, stages, sink
                )
                completed.append(entity_id)
                skipped_count += summary.skipped_stages
        else:
            assert isinstance(source, str | Path)  # narrowed above
            entity_ids = _collect_entity_ids(Path(source))
            with mp.get_context("spawn").Pool(processes=n_workers) as pool:
                args = [
                    (eid, str(Path(source)), p, stages, str(sink))
                    for eid in entity_ids
                ]
                for res in pool.imap_unordered(_worker_entrypoint, args):
                    completed.append(res.entity_id)
                    skipped_count += res.skipped_stages
    except _StageError as exc:
        elapsed = time.monotonic() - started
        return RunReport(
            sink_dir=sink,
            stages=tuple(stages),
            n_entities=len(completed) + 1,
            n_completed=len(completed),
            n_skipped_existing=skipped_count,
            elapsed_seconds=elapsed,
            failed_entity=exc.entity_id,
            failed_stage=exc.stage,
            failed_reason=str(exc.__cause__) if exc.__cause__ else exc.reason,
            completed_entity_ids=tuple(completed),
        )

    elapsed = time.monotonic() - started
    return RunReport(
        sink_dir=sink,
        stages=tuple(stages),
        n_entities=len(completed),
        n_completed=len(completed),
        n_skipped_existing=skipped_count,
        elapsed_seconds=elapsed,
        completed_entity_ids=tuple(completed),
    )


# ── Per-entity processor ────────────────────────────────────────────


class _StageError(Exception):
    """Internal error wrapper carrying entity / stage attribution."""

    def __init__(self, entity_id: str, stage: str, reason: str) -> None:
        super().__init__(f"{entity_id}/{stage}: {reason}")
        self.entity_id = entity_id
        self.stage = stage
        self.reason = reason


class _EntitySummary:
    __slots__ = ("entity_id", "skipped_stages")

    def __init__(self, entity_id: str, skipped_stages: int) -> None:
        self.entity_id = entity_id
        self.skipped_stages = skipped_stages


def _process_one_entity(
    entity_id: str,
    pings_df: pd.DataFrame,
    params: RunParams,
    stages: tuple[str, ...],
    sink_dir: Path,
) -> _EntitySummary:
    """Run all requested stages for one entity, persisting each output."""
    outputs: dict[str, object] = {}
    skipped = 0

    for stage in stages:
        try:
            out_path = _stage_path(sink_dir, stage, entity_id)
            if out_path.exists():
                outputs[stage] = _read_stage(stage, out_path)
                skipped += 1
                continue
            outputs[stage] = _run_stage(
                stage, entity_id, pings_df, outputs, params, sink_dir
            )
            _write_stage_atomic(stage, outputs[stage], entity_id, out_path)
        except _StageError:
            raise
        except Exception as exc:
            raise _StageError(entity_id, stage, type(exc).__name__) from exc

    return _EntitySummary(entity_id, skipped)


def _run_stage(
    stage: str,
    entity_id: str,
    pings_df: pd.DataFrame,
    outputs: dict[str, object],
    params: RunParams,
    sink_dir: Path,
) -> object:
    """Dispatch a single stage's L1 computation and return its output."""
    if stage == "clean":
        cleaned = clean(pings_df, params.clean)
        if params.run_stale_merge:
            cleaned = merge_stale_positions(cleaned, params.stale_merge, params.clean)
        return cleaned

    if stage == "segment":
        cleaned = _ensure_loaded(outputs, "clean", sink_dir, entity_id, stage)
        per_ping = segment(cleaned, params.segment)
        return aggregate_segments(per_ping, params.segment)

    if stage == "episode":
        segments_df = _ensure_loaded(outputs, "segment", sink_dir, entity_id, stage)
        return detect_episodes(segments_df, params.episode)

    if stage == "embed_segments":
        segments_df = _ensure_loaded(outputs, "segment", sink_dir, entity_id, stage)
        vectors, ids = embed_segments(segments_df, params.embed)
        return _vectors_to_df(entity_id, ids, vectors)

    if stage == "embed_episodes":
        episodes_df = _ensure_loaded(outputs, "episode", sink_dir, entity_id, stage)
        seg_vec_df = _ensure_loaded(
            outputs, "embed_segments", sink_dir, entity_id, stage
        )
        seg_vectors, seg_ids = _df_to_vectors(seg_vec_df)
        ep_vectors, ep_ids = embed_episodes(
            episodes_df, seg_vectors, seg_ids, params.embed
        )
        return _vectors_to_df(entity_id, ep_ids, ep_vectors)

    msg = f"unknown stage {stage!r}"
    raise ValueError(msg)


def _ensure_loaded(
    outputs: dict[str, object],
    needed: str,
    sink_dir: Path,
    entity_id: str,
    requesting_stage: str,
) -> pd.DataFrame:
    """Return ``outputs[needed]`` if present, otherwise read it from disk."""
    if needed in outputs:
        cached = outputs[needed]
        assert isinstance(cached, pd.DataFrame)
        return cached
    path = _stage_path(sink_dir, needed, entity_id)
    if not path.exists():
        msg = (
            f"stage {requesting_stage!r} requires {needed!r} output "
            f"but it is neither in memory nor at {path}. Either include "
            f"{needed!r} in the stages tuple or pre-populate the sink."
        )
        raise FileNotFoundError(msg)
    return _read_stage(needed, path)


# ── Multiprocessing worker entry ────────────────────────────────────


def _collect_entity_ids(source_path: Path) -> list[str]:
    """List entity_ids present at a source path. Used to fan out workers.

    Uses ``iter_entities`` which already validates each per-entity frame;
    that's a one-time cost in the parent process. Workers then re-read
    only their own slice.
    """
    return [eid for eid, _ in iter_entities(source_path)]


def _worker_entrypoint(
    args: tuple[str, str, RunParams, tuple[str, ...], str],
) -> _EntitySummary:
    """Multiprocessing worker: read this entity's slice and process it."""
    entity_id, source_path_str, params, stages, sink_dir_str = args
    source_path = Path(source_path_str)
    sink_dir = Path(sink_dir_str)
    for eid, pings_df in iter_entities(source_path):
        if eid == entity_id:
            return _process_one_entity(eid, pings_df, params, stages, sink_dir)
    msg = f"worker: entity_id={entity_id!r} not found in {source_path}"
    raise RuntimeError(msg)


# ── Persistence helpers ─────────────────────────────────────────────


def _stage_path(sink_dir: Path, stage: str, entity_id: str) -> Path:
    """Hive-partitioned final path for one entity's stage output."""
    safe = _safe_partition_value(entity_id)
    return sink_dir / stage / f"entity_id={safe}" / "data.parquet"


def _safe_partition_value(value: str) -> str:
    """Hive partition values can't contain ``/``; reject for now."""
    if "/" in value:
        msg = f"entity_id contains '/' which is invalid for Hive partitioning: {value!r}"
        raise ValueError(msg)
    return value


def _validate_stages(stages: Iterable[str]) -> None:
    seen: set[str] = set()
    last_index = -1
    for s in stages:
        if s not in _VALID_STAGES:
            msg = f"unknown stage {s!r}; valid: {sorted(_VALID_STAGES)}"
            raise ValueError(msg)
        if s in seen:
            msg = f"stage {s!r} appears twice"
            raise ValueError(msg)
        seen.add(s)
        idx = DEFAULT_STAGES.index(s)
        if idx <= last_index:
            msg = (
                f"stages must be a subset of DEFAULT_STAGES in canonical order; "
                f"got {tuple(stages)}"
            )
            raise ValueError(msg)
        last_index = idx


def _write_stage_atomic(
    stage: str, payload: object, entity_id: str, final_path: Path
) -> None:
    """Atomic per-entity write: tmp file + ``os.replace``."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(".parquet.tmp")
    if not isinstance(payload, pd.DataFrame):
        msg = f"stage {stage!r} produced non-DataFrame output of type {type(payload).__name__}"
        raise TypeError(msg)
    payload.to_parquet(tmp_path, index=False, compression="snappy")
    os.replace(tmp_path, final_path)
    _ = entity_id  # currently unused, reserved for future per-entity diagnostics


def _read_stage(stage: str, path: Path) -> pd.DataFrame:
    """Read a stage's persisted output from parquet."""
    df = pd.read_parquet(path)
    return _restore_dtypes(stage, df)


def _restore_dtypes(stage: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce parquet round-trip dtypes back to canonical schema dtypes.

    Parquet preserves Arrow types; pandas applies ``object`` for some
    string-typed columns when reading without a types_mapper. Force the
    canonical pandas StringDtype on identifier columns so downstream
    L1 functions trust the contract.
    """
    if "entity_id" in df.columns:
        df["entity_id"] = df["entity_id"].astype("string")
    if stage in ("clean", "segment") and "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype("string")
    if stage == "segment":
        if "segment_type" in df.columns:
            df["segment_type"] = df["segment_type"].astype("string")
        for col in ("start_h3", "end_h3"):
            if col in df.columns:
                df[col] = df[col].astype("string")
    if stage == "episode":
        for col in ("episode_id", "episode_type", "anchor_h3"):
            if col in df.columns:
                df[col] = df[col].astype("string")
    if stage in ("embed_segments", "embed_episodes"):
        df["id"] = df["id"].astype("string")
    return df


# ── Vector ↔ DataFrame conversion ───────────────────────────────────


def _vectors_to_df(
    entity_id: str, ids: list[str], vectors: np.ndarray
) -> pd.DataFrame:
    """Build a VectorsSchema-shaped frame for parquet persistence."""
    if vectors.shape[0] != len(ids):
        msg = f"vectors rows {vectors.shape[0]} ≠ ids length {len(ids)}"
        raise ValueError(msg)
    rows = [vectors[i].astype(np.float32) for i in range(vectors.shape[0])]
    return pd.DataFrame(
        {
            "id": pd.Series(ids, dtype="string"),
            "entity_id": pd.Series([entity_id] * len(ids), dtype="string"),
            "vector": rows,
        }
    )


def _df_to_vectors(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Inverse of ``_vectors_to_df``: reconstruct ``(vectors, ids)``."""
    if "vector" not in df.columns or "id" not in df.columns:
        msg = "df must have 'id' and 'vector' columns"
        raise ValueError(msg)
    if len(df) == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    rows = [np.asarray(v, dtype=np.float32) for v in df["vector"]]
    matrix = np.vstack(rows).astype(np.float32)
    return matrix, df["id"].astype(str).tolist()


# Module-level export used in __init__
__all__ = ["DEFAULT_STAGES", "process"]


# Avoid unused-import warning for Iterator (kept for type-hint clarity)
_ = Iterator
