"""Cohort-relative z-score helper.

Implements ``baseline_zscores``. Given a precomputed ``BaselinesSchema``
table (output of pass-2 ``fit_baselines``), this function looks up each
segment's cohort and emits a ``<metric>_z`` column for every metric in
the baseline table. The expensive computation — fitting the means and
standard deviations across the cohort — lives in pass-2; this is a pure
per-entity application stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_zscores(
    segments_df: pd.DataFrame,
    baselines: pd.DataFrame,
    cohort_keys: list[str],
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Add ``<metric>_z`` columns to ``segments_df`` from cohort baselines.

    Parameters
    ----------
    segments_df
        Output of ``trajkit.segment.aggregate_segments`` for one entity.
    baselines
        Output of pass-2 ``fit_baselines`` validated against the schema
        produced by ``trajkit.types.make_baselines_schema(cohort_keys)``.
    cohort_keys
        Column names that compose the cohort key (must exist on both
        frames). Same value used to fit the baseline.
    epsilon
        Floor on the baseline standard deviation to avoid division by zero
        for collapsed cohorts.

    Returns
    -------
    pd.DataFrame
        ``segments_df`` with one ``<metric>_z`` column per metric in the
        baseline table. Cohorts not present in the baseline table emit
        ``NaN`` for the affected segments — callers can downstream-fill if
        desired.
    """
    if len(segments_df) == 0:
        return segments_df.copy()

    missing = [k for k in cohort_keys if k not in segments_df.columns]
    if missing:
        msg = f"cohort_keys not in segments_df: {missing}"
        raise ValueError(msg)
    missing_b = [k for k in cohort_keys if k not in baselines.columns]
    if missing_b:
        msg = f"cohort_keys not in baselines: {missing_b}"
        raise ValueError(msg)

    df = segments_df.copy()

    metrics = baselines["metric"].unique().tolist()
    for metric in metrics:
        if metric not in df.columns:
            # Baseline mentions a metric the segments frame doesn't have;
            # skip silently since the user opted in to this baseline set.
            continue
        baseline_metric = baselines[baselines["metric"] == metric][
            [*cohort_keys, "mean", "std"]
        ]
        merged = df.merge(baseline_metric, on=cohort_keys, how="left")
        std = merged["std"].astype(np.float64).to_numpy()
        mean = merged["mean"].astype(np.float64).to_numpy()
        value = df[metric].astype(np.float64).to_numpy()
        z = (value - mean) / np.maximum(std, epsilon)
        df[f"{metric}_z"] = z.astype(np.float32)

    return df
