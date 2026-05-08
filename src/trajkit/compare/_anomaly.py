"""Per-call anomaly scoring via IsolationForest.

Implements ``anomaly_score``. v0.1.0 ships this convenience helper only —
cohort-stable scoring (``fit_anomaly_model`` + persisted models) is
deferred to v1.1 per LIBRARY.md §17.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

_F32 = np.float32


def anomaly_score(
    vectors: np.ndarray,
    contamination: float = 0.01,
    *,
    n_estimators: int = 200,
    random_state: int = 42,
) -> np.ndarray:
    """Return anomaly scores for each row of ``vectors``.

    Fits an ``IsolationForest`` to ``vectors`` and returns the negative of
    its decision function so that **more positive = more anomalous**. The
    fit is per-call; this is a one-shot convenience, not a cohort-stable
    scorer. Use cases that need stable scoring across runs should wait
    for ``fit_anomaly_model`` (v1.1) which persists the trained model.

    Parameters
    ----------
    vectors
        ``(N, D)`` float-typed array.
    contamination
        Expected fraction of anomalies, in ``(0, 0.5]``. Defaults to 0.01.
    n_estimators
        Forest size; 200 trees is a sensible default for ≤ 1M vectors.
    random_state
        Seeded for determinism by default.

    Returns
    -------
    np.ndarray
        ``(N,)`` float32 array of scores aligned with ``vectors`` rows.
    """
    if vectors.ndim != 2:
        msg = f"vectors must be 2-D, got shape {vectors.shape}"
        raise ValueError(msg)
    if not (0.0 < contamination <= 0.5):
        msg = f"contamination must be in (0, 0.5], got {contamination}"
        raise ValueError(msg)

    if vectors.shape[0] == 0:
        return np.zeros((0,), dtype=_F32)

    forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    forest.fit(vectors)
    scores = -forest.decision_function(vectors)
    result: np.ndarray = scores.astype(_F32)
    return result
