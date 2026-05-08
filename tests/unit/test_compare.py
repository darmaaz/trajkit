"""Unit tests for ``trajkit.compare``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from trajkit.compare import (
    Hit,
    Index,
    anomaly_score,
    build_index,
    load_index,
    save_index,
    search,
)

# ── Fixture helpers ─────────────────────────────────────────────────


def _random_vectors(n: int = 100, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _ids(n: int) -> list[str]:
    return [f"id_{i:05d}" for i in range(n)]


# ── build_index: contract ───────────────────────────────────────────


def test_build_index_returns_index_with_expected_metadata() -> None:
    vectors = _random_vectors(100, 8)
    idx = build_index(vectors, _ids(100))
    assert isinstance(idx, Index)
    assert len(idx) == 100
    assert idx.dim == 8
    assert idx.metric == "cosine"


def test_build_index_supports_l2_metric() -> None:
    vectors = _random_vectors(50, 4)
    idx = build_index(vectors, _ids(50), metric="l2")
    assert idx.metric == "l2"
    assert len(idx) == 50


def test_build_index_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        build_index(_random_vectors(10, 4), _ids(10), metric="manhattan")  # type: ignore[arg-type]


def test_build_index_rejects_misaligned_ids() -> None:
    with pytest.raises(ValueError, match="disagree"):
        build_index(_random_vectors(10, 4), _ids(9))


def test_build_index_rejects_non_2d_vectors() -> None:
    with pytest.raises(ValueError, match="2-D"):
        build_index(np.zeros(10, dtype=np.float32), _ids(10))


def test_build_index_coerces_float64_to_float32() -> None:
    vectors = np.random.standard_normal((20, 4))  # float64
    idx = build_index(vectors, _ids(20))
    assert len(idx) == 20  # didn't error during faiss add


# ── search: top-k correctness ───────────────────────────────────────


def test_search_self_match_scores_unity_for_cosine() -> None:
    vectors = _random_vectors(50, 8)
    idx = build_index(vectors, _ids(50))
    hits = search(idx, vectors[7], k=3)
    assert hits[0].id == "id_00007"
    assert abs(hits[0].score - 1.0) < 1e-4


def test_search_returns_k_results() -> None:
    vectors = _random_vectors(100, 8)
    idx = build_index(vectors, _ids(100))
    hits = search(idx, vectors[0], k=10)
    assert len(hits) == 10


def test_search_ranks_descending_by_score_for_cosine() -> None:
    vectors = _random_vectors(50, 8)
    idx = build_index(vectors, _ids(50))
    hits = search(idx, vectors[0], k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_supports_2d_query_with_one_row() -> None:
    vectors = _random_vectors(50, 8)
    idx = build_index(vectors, _ids(50))
    hits_1d = search(idx, vectors[0], k=3)
    hits_2d = search(idx, vectors[0:1], k=3)
    assert [h.id for h in hits_1d] == [h.id for h in hits_2d]


def test_search_rejects_dim_mismatch() -> None:
    vectors = _random_vectors(20, 8)
    idx = build_index(vectors, _ids(20))
    with pytest.raises(ValueError, match="dim"):
        search(idx, np.zeros(4, dtype=np.float32), k=3)


def test_search_handles_empty_index_gracefully() -> None:
    idx = build_index(np.zeros((0, 8), dtype=np.float32), [])
    hits = search(idx, np.zeros(8, dtype=np.float32), k=5)
    assert hits == []


def test_search_l2_returns_negative_distances_as_score() -> None:
    """For L2, score = -distance so higher is better, matching cosine semantics."""
    vectors = _random_vectors(20, 4)
    idx = build_index(vectors, _ids(20), metric="l2")
    hits = search(idx, vectors[0], k=3)
    # Self-match: distance 0 → score 0
    assert hits[0].id == "id_00000"
    assert abs(hits[0].score) < 1e-3
    # Subsequent hits have negative scores (positive distance)
    assert hits[1].score <= 0


# ── search: filter_ids ──────────────────────────────────────────────


def test_search_filter_restricts_results_to_allowed_ids() -> None:
    vectors = _random_vectors(50, 8)
    idx = build_index(vectors, _ids(50))
    allowed = frozenset({"id_00010", "id_00020", "id_00030"})
    hits = search(idx, vectors[0], k=5, filter_ids=allowed)
    assert all(h.id in allowed for h in hits)


def test_search_filter_can_yield_fewer_than_k() -> None:
    vectors = _random_vectors(50, 8)
    idx = build_index(vectors, _ids(50))
    allowed = frozenset({"id_00007"})
    hits = search(idx, vectors[0], k=10, filter_ids=allowed)
    assert len(hits) == 1
    assert hits[0].id == "id_00007"


# ── Persistence ─────────────────────────────────────────────────────


def test_save_and_load_index_round_trip(tmp_path: Path) -> None:
    vectors = _random_vectors(40, 8)
    ids = _ids(40)
    idx = build_index(vectors, ids)
    save_index(idx, tmp_path / "idx")
    loaded = load_index(tmp_path / "idx")

    assert len(loaded) == len(idx)
    assert loaded.dim == idx.dim
    assert loaded.metric == idx.metric

    # Top-k results identical
    hits_a = search(idx, vectors[3], k=5)
    hits_b = search(loaded, vectors[3], k=5)
    assert [h.id for h in hits_a] == [h.id for h in hits_b]


def test_load_index_raises_on_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "doesnt_exist")


# ── anomaly_score ───────────────────────────────────────────────────


def test_anomaly_score_returns_one_score_per_row() -> None:
    vectors = _random_vectors(100, 8)
    scores = anomaly_score(vectors, contamination=0.05)
    assert scores.shape == (100,)
    assert scores.dtype == np.float32


def test_anomaly_score_handles_empty_input() -> None:
    vectors = np.zeros((0, 8), dtype=np.float32)
    scores = anomaly_score(vectors)
    assert scores.shape == (0,)


def test_anomaly_score_ranks_planted_outliers_high() -> None:
    """Synthetic mixture: 200 normal points + 5 planted outliers."""
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((200, 4)).astype(np.float32) * 0.5
    outliers = (rng.standard_normal((5, 4)).astype(np.float32) * 0.1) + 100.0
    vectors = np.vstack([normal, outliers])
    scores = anomaly_score(vectors, contamination=0.025)

    # The 5 highest-scoring rows should mostly be the planted outliers (last 5)
    top_indices = np.argsort(-scores)[:5]
    n_outliers_in_top = sum(1 for i in top_indices if i >= 200)
    assert n_outliers_in_top >= 4  # allow one false positive for noise


def test_anomaly_score_rejects_invalid_contamination() -> None:
    vectors = _random_vectors(50, 4)
    with pytest.raises(ValueError, match="contamination"):
        anomaly_score(vectors, contamination=0.0)
    with pytest.raises(ValueError, match="contamination"):
        anomaly_score(vectors, contamination=0.6)


def test_anomaly_score_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        anomaly_score(np.zeros(10, dtype=np.float32))


def test_anomaly_score_is_deterministic_with_default_seed() -> None:
    vectors = _random_vectors(50, 4)
    a = anomaly_score(vectors)
    b = anomaly_score(vectors)
    np.testing.assert_array_equal(a, b)


# ── Hit dataclass ───────────────────────────────────────────────────


def test_hit_is_frozen_and_carries_three_fields() -> None:
    h = Hit(id="x", score=0.5, rank=2)
    assert h.id == "x"
    assert h.score == 0.5
    assert h.rank == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.score = 0.9  # type: ignore[misc]
