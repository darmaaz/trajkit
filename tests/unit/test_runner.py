"""Unit tests for ``trajkit.runner``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trajkit.runner import DEFAULT_STAGES, RunParams, RunReport, process

# ── Fixture builders ────────────────────────────────────────────────


def _multi_entity_pings(
    n_per: int = 200, entities: tuple[str, ...] = ("v1", "v2")
) -> pd.DataFrame:
    parts = []
    for i, eid in enumerate(entities):
        n = n_per
        ts_start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i)
        parts.append(
            pd.DataFrame(
                {
                    "entity_id": pd.Series([eid] * n, dtype="string"),
                    "ts": pd.date_range(ts_start, periods=n, freq="1s").astype(
                        "datetime64[ns, UTC]"
                    ),
                    "lat": 19.4 + np.arange(n) * 0.0001 + i * 0.01,
                    "lon": np.full(n, -99.2, dtype=np.float64),
                    "speed_ms": np.full(n, np.nan, dtype=np.float32),
                    "bearing_deg": np.full(n, np.nan, dtype=np.float32),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


# ── End-to-end happy path ───────────────────────────────────────────


def test_process_writes_all_stages_for_each_entity(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    rep = process(df, tmp_path)
    assert rep.succeeded
    assert rep.n_completed == 2
    for eid in ("v1", "v2"):
        for stage in DEFAULT_STAGES:
            assert (tmp_path / stage / f"entity_id={eid}" / "data.parquet").exists()


def test_process_returns_runreport_with_metadata(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    rep = process(df, tmp_path)
    assert isinstance(rep, RunReport)
    assert rep.sink_dir == tmp_path
    assert rep.stages == DEFAULT_STAGES
    assert sorted(rep.completed_entity_ids) == ["v1", "v2"]
    assert rep.elapsed_seconds >= 0.0


def test_process_default_params_used_when_none(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    rep = process(df, tmp_path)
    assert rep.succeeded


# ── Resume / skip-existing ──────────────────────────────────────────


def test_process_skips_existing_outputs_on_re_run(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    process(df, tmp_path)  # first run writes everything

    rep = process(df, tmp_path)  # second run sees all stages present
    assert rep.succeeded
    # Each entity has 5 stages; both entities skip all → 10 skipped stages
    assert rep.n_skipped_existing == len(DEFAULT_STAGES) * 2


def test_process_completes_partial_resume(tmp_path: Path) -> None:
    """A run that only requested ``clean`` first; subsequent run completes."""
    df = _multi_entity_pings()
    process(df, tmp_path, stages=("clean",))
    # Only clean exists; segment/etc don't
    assert (tmp_path / "clean" / "entity_id=v1" / "data.parquet").exists()
    assert not (tmp_path / "segment" / "entity_id=v1" / "data.parquet").exists()

    rep = process(df, tmp_path)  # full pipeline
    assert rep.succeeded
    # The clean stage skipped (already exists), others computed
    assert rep.n_skipped_existing == 2  # 2 entities × 1 skipped clean stage


# ── Stage subset ────────────────────────────────────────────────────


def test_process_runs_only_requested_stages(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    rep = process(df, tmp_path, stages=("clean", "segment"))
    assert rep.succeeded
    for eid in ("v1", "v2"):
        assert (tmp_path / "clean" / f"entity_id={eid}" / "data.parquet").exists()
        assert (tmp_path / "segment" / f"entity_id={eid}" / "data.parquet").exists()
        assert not (tmp_path / "episode" / f"entity_id={eid}" / "data.parquet").exists()


def test_process_rejects_stages_out_of_canonical_order(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    with pytest.raises(ValueError, match="canonical order"):
        process(df, tmp_path, stages=("segment", "clean"))


def test_process_rejects_unknown_stage(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    with pytest.raises(ValueError, match="unknown stage"):
        process(df, tmp_path, stages=("nope",))  # type: ignore[arg-type]


def test_process_rejects_duplicate_stage(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    with pytest.raises(ValueError, match="appears twice"):
        process(df, tmp_path, stages=("clean", "clean"))


# ── Atomic writes & layout ──────────────────────────────────────────


def test_process_uses_hive_layout(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    process(df, tmp_path, stages=("clean",))
    children = sorted(p.name for p in (tmp_path / "clean").iterdir())
    assert children == ["entity_id=v1", "entity_id=v2"]


def test_process_leaves_no_tmp_files_on_success(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    process(df, tmp_path)
    tmp_files = list(tmp_path.rglob("*.parquet.tmp"))
    assert tmp_files == []


def test_process_rejects_entity_id_with_path_separator(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "entity_id": pd.Series(["bad/id"], dtype="string"),
            "ts": pd.Series(
                pd.date_range("2026-01-01", periods=1, freq="1s", tz="UTC")
            ).astype("datetime64[ns, UTC]"),
            "lat": [19.4],
            "lon": [-99.2],
            "speed_ms": pd.Series([np.nan], dtype=np.float32),
            "bearing_deg": pd.Series([np.nan], dtype=np.float32),
        }
    )
    rep = process(df, tmp_path)
    assert not rep.succeeded
    assert rep.failed_entity == "bad/id"


# ── Failure semantics ───────────────────────────────────────────────


def test_process_aborts_on_per_entity_exception(tmp_path: Path) -> None:
    """A malformed entity (zero pings produces an empty trace) abruptly aborts."""
    # entity v1 OK, v2 has a single ping which still works through the pipeline
    # so we contrive failure via a stage subset that requires a missing prior.
    df = _multi_entity_pings()
    # Request "segment" only — clean output is required but not built
    rep = process(df, tmp_path, stages=("segment",))
    assert not rep.succeeded
    assert rep.failed_stage == "segment"


# ── Path-source dispatch ────────────────────────────────────────────


def test_process_accepts_parquet_path(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    src = tmp_path / "pings.parquet"
    df.to_parquet(src, index=False)
    out = tmp_path / "out"
    rep = process(src, out, stages=("clean",))
    assert rep.succeeded
    assert (out / "clean" / "entity_id=v1" / "data.parquet").exists()


def test_process_forces_single_process_for_in_memory_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    df = _multi_entity_pings()
    with caplog.at_level("WARNING"):
        rep = process(df, tmp_path, n_workers=4, stages=("clean",))
    assert rep.succeeded
    assert any("Falling back to n_workers=1" in rec.message for rec in caplog.records)


# ── RunParams plumbing ──────────────────────────────────────────────


def test_run_params_are_frozen() -> None:
    p = RunParams()
    with pytest.raises(ValidationError):
        p.run_stale_merge = True  # type: ignore[misc]


def test_run_params_reject_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RunParams(junk="oops")  # type: ignore[call-arg]


def test_run_params_run_stale_merge_default_off() -> None:
    p = RunParams()
    assert p.run_stale_merge is False


def test_process_with_stale_merge_enabled(tmp_path: Path) -> None:
    """Sanity check: enabling stale-merge in the runner doesn't break the pipeline."""
    df = _multi_entity_pings(n_per=300)
    rep = process(df, tmp_path, RunParams(run_stale_merge=True), stages=("clean",))
    assert rep.succeeded


# ── Vectors round-trip ──────────────────────────────────────────────


def test_process_vector_outputs_have_expected_columns(tmp_path: Path) -> None:
    df = _multi_entity_pings()
    process(df, tmp_path, stages=("clean", "segment", "embed_segments"))
    vec_path = tmp_path / "embed_segments" / "entity_id=v1" / "data.parquet"
    vec_df = pd.read_parquet(vec_path)
    assert set(vec_df.columns) == {"id", "entity_id", "vector"}
    # Vector cells contain numpy arrays
    first = vec_df["vector"].iloc[0]
    assert hasattr(first, "__len__")
