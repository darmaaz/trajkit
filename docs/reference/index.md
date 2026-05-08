# API Reference

Auto-generated from docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/).

| Module | Layer | What |
|---|---|---|
| [`trajkit.clean`](clean.md) | L1 | quality flags, dedup, stale-position merge |
| [`trajkit.segment`](segment.md) | L1 | hysteresis state machine + aggregation |
| [`trajkit.episode`](episode.md) | L1 | spatial-envelope STAY/TRANSIT detection |
| [`trajkit.embed`](embed.md) | L1 | per-segment + per-episode vectorisation |
| [`trajkit.compare`](compare.md) | L1 | FAISS similarity + per-call anomaly |
| [`trajkit.io`](io.md) | L2 | entity iterator (parquet / Arrow / DataFrame / CSV) |
| [`trajkit.runner`](runner.md) | L3 | end-to-end pipeline orchestrator |
| [`trajkit.baselines`](baselines.md) | pass-2 | cohort statistics |
| [`trajkit.testing`](testing.md) | helpers | minimal synthetic builders |
| [`trajkit.presets`](presets.md) | helpers | domain `RunParams` bundles |
| [`trajkit.types`](types.md) | schemas | Pandera + Arrow declarations |
