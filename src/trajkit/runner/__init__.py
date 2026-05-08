"""``trajkit.runner`` — L3 process orchestrator.

Public API:

* ``process(source, sink_dir, params, *, stages, n_workers)`` —
  end-to-end pipeline run with Hive-partitioned outputs and atomic
  per-entity writes.
* ``RunParams`` — bundles per-stage Params.
* ``RunReport`` — frozen dataclass summary returned by ``process``.
* ``DEFAULT_STAGES`` — canonical stage order for the runner.

See ``docs/design/LIBRARY.md`` §5 (L3) and §7 (I/O & storage).
"""

from __future__ import annotations

from trajkit.runner._params import RunParams
from trajkit.runner._process import DEFAULT_STAGES, process
from trajkit.runner._report import RunReport

__all__ = ["DEFAULT_STAGES", "RunParams", "RunReport", "process"]
