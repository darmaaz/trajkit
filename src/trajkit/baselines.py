"""Pass-2 baseline fitting: cohort statistics from a segments stage output.

Implements ``fit_baselines``. Reads pass-1 segments output (parquet
directory or DataFrame), computes ``mean`` / ``std`` / ``n_samples`` per
``(cohort_keys, metric)``, applies a v0.1.0 single-tier global fallback
when cohort sample count falls below ``min_cohort_n``, and persists the
result.

Output conforms to the Pandera schema produced by
``trajkit.types.make_baselines_schema(cohort_keys)``.

Future work (v1.1+): parent-cohort hierarchy for richer fallback (e.g.,
fall back from ``(entity_id, segment_type)`` to ``(segment_type,)``
before going global).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from trajkit.types import make_baselines_arrow, make_baselines_schema

DEFAULT_METRICS: tuple[str, ...] = (
    "duration_s",
    "path_length_m",
    "displacement_m",
    "straightness",
    "mean_speed_ms",
    "max_speed_ms",
    "bearing_variance",
    "n_pings",
)


class BaselineParams(BaseModel):
    """Frozen parameters for ``fit_baselines``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_cohort_n: PositiveInt = Field(
        default=30,
        description=(
            "Minimum non-null sample count for a cohort to be trusted. "
            "Below this, the global mean/std for the metric is substituted "
            "and ``is_fallback`` is True."
        ),
    )
    min_global_n: NonNegativeInt = Field(
        default=10,
        description=(
            "Minimum non-null sample count across all rows for a metric to "
            "be included at all. Metrics below this are dropped entirely "
            "(no global statistic is meaningful enough to fall back to)."
        ),
    )


def fit_baselines(
    source: str | Path | pd.DataFrame,
    cohort_keys: list[str],
    *,
    metrics: list[str] | None = None,
    params: BaselineParams | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute per-cohort baselines from a segments table.

    Parameters
    ----------
    source
        Either a path (parquet file or Hive-partitioned directory of
        ``trajkit.segment.aggregate_segments`` outputs) or a DataFrame
        already in memory.
    cohort_keys
        Column names defining the cohort group (e.g. ``["entity_id"]`` or
        ``["entity_id", "segment_type"]``). Must exist on the segments
        table.
    metrics
        Optional list of metric column names. Defaults to the standard
        kinematic set in ``DEFAULT_METRICS``. Metrics that don't exist on
        the input are silently skipped.
    params
        Frozen ``BaselineParams``. Defaults: ``min_cohort_n=30``,
        ``min_global_n=10``.
    out_path
        Optional parquet path. When set, the output is written there in
        addition to being returned.

    Returns
    -------
    pd.DataFrame
        Baselines frame validated against
        ``make_baselines_schema(cohort_keys)``.
    """
    p = params if params is not None else BaselineParams()
    df = _load_source(source)

    missing_keys = [k for k in cohort_keys if k not in df.columns]
    if missing_keys:
        msg = f"cohort_keys not in source: {missing_keys}"
        raise ValueError(msg)
    if not cohort_keys:
        msg = "cohort_keys must contain at least one column"
        raise ValueError(msg)

    metric_set = list(metrics) if metrics else list(DEFAULT_METRICS)
    available = [m for m in metric_set if m in df.columns]
    if not available:
        msg = f"none of the requested metrics are in source: {metric_set}"
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    for metric in available:
        rows.extend(_fit_one_metric(df, cohort_keys, metric, p))

    if not rows:
        return _empty_baselines_frame(cohort_keys)

    out = _build_output_frame(rows, cohort_keys)
    schema = make_baselines_schema(cohort_keys)
    schema.validate(out)

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False, compression="snappy")

    return out


# ── Per-metric fit ──────────────────────────────────────────────────


def _fit_one_metric(
    df: pd.DataFrame,
    cohort_keys: list[str],
    metric: str,
    p: BaselineParams,
) -> list[dict[str, object]]:
    """Compute baseline rows for a single metric."""
    series = df[metric]
    valid_mask = series.notna()
    valid = series[valid_mask].astype(np.float64)
    n_global = int(valid.shape[0])
    if n_global < p.min_global_n:
        return []

    global_mean = float(valid.mean())
    global_std = float(valid.std()) if n_global > 1 else 0.0

    cohort_df = df.loc[valid_mask, [*cohort_keys, metric]].copy()
    cohort_df[metric] = cohort_df[metric].astype(np.float64)
    grouped = cohort_df.groupby(cohort_keys, sort=False, dropna=False)

    rows: list[dict[str, object]] = []
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        n = int(len(group))
        if n >= p.min_cohort_n:
            mean = float(group[metric].mean())
            std = float(group[metric].std()) if n > 1 else 0.0
            is_fallback = False
        else:
            mean = global_mean
            std = global_std
            is_fallback = True

        row: dict[str, object] = {key_col: key_tuple[i] for i, key_col in enumerate(cohort_keys)}
        row["metric"] = metric
        row["mean"] = mean
        row["std"] = std
        row["n_samples"] = n
        row["is_fallback"] = is_fallback
        rows.append(row)

    return rows


# ── I/O helpers ─────────────────────────────────────────────────────


def _load_source(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a segments source — DataFrame, parquet file, or parquet directory.

    Directory inputs are read by globbing each per-entity parquet file
    individually (rather than via ``pd.read_parquet(dir)`` which delegates
    to ``pyarrow.dataset``). The dataset-level read trips on a dict-vs-
    string type mismatch when ``entity_id`` is encoded both as a Hive
    partition (dictionary) and as an in-file column (string), which is
    how the runner writes Hive output. Per-file reads bypass that.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        msg = f"source path does not exist: {path}"
        raise FileNotFoundError(msg)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            msg = f"no parquet files under {path}"
            raise FileNotFoundError(msg)
        parts: list[pd.DataFrame] = []
        for f in files:
            sub = pd.read_parquet(f)
            # Restore Hive partition columns (path-encoded `key=value`)
            # when the per-file parquet doesn't carry them. Handles both
            # Hive-pure writers (pandas' ``partition_cols`` strips the col)
            # and redundant writers (col both in file and path).
            for parent in f.parents:
                key, sep, value = parent.name.partition("=")
                if not sep:
                    continue
                if key not in sub.columns:
                    sub[key] = pd.Series([value] * len(sub), dtype="string")
            parts.append(sub)
        return pd.concat(parts, ignore_index=True)
    return pd.read_parquet(path)


def _build_output_frame(
    rows: list[dict[str, object]], cohort_keys: list[str]
) -> pd.DataFrame:
    """Coerce a list-of-dicts to canonical Baselines dtypes."""
    out = pd.DataFrame(rows)
    for key in cohort_keys:
        out[key] = out[key].astype("string")
    out["metric"] = out["metric"].astype("string")
    out["mean"] = out["mean"].astype(np.float32)
    out["std"] = out["std"].astype(np.float32)
    out["n_samples"] = out["n_samples"].astype(np.int32)
    out["is_fallback"] = out["is_fallback"].astype(bool)
    column_order = [*cohort_keys, "metric", "mean", "std", "n_samples", "is_fallback"]
    return out[column_order]


def _empty_baselines_frame(cohort_keys: list[str]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        key: pd.Series([], dtype="string") for key in cohort_keys
    }
    columns["metric"] = pd.Series([], dtype="string")
    columns["mean"] = pd.Series([], dtype=np.float32)
    columns["std"] = pd.Series([], dtype=np.float32)
    columns["n_samples"] = pd.Series([], dtype=np.int32)
    columns["is_fallback"] = pd.Series([], dtype=bool)
    return pd.DataFrame(columns)


# Re-export the Arrow schema helper for users who want it for parquet writes
# from outside this module.
__all__ = [
    "DEFAULT_METRICS",
    "BaselineParams",
    "fit_baselines",
    "make_baselines_arrow",
]
