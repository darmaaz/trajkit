"""Result type returned by ``process``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunReport:
    """Summary of a ``process`` invocation.

    Per LIBRARY.md §17 v1: per-entity exception aborts the run. The report
    therefore lists at most one failed entity (the first to fail). v1.1+
    adds a resilient mode that catches and continues.
    """

    sink_dir: Path
    stages: tuple[str, ...]
    n_entities: int
    n_completed: int
    n_skipped_existing: int
    elapsed_seconds: float
    failed_entity: str | None = None
    failed_stage: str | None = None
    failed_reason: str | None = None
    completed_entity_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> bool:
        return self.failed_entity is None
