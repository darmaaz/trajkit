# Concepts

trajkit turns a continuous, noisy GPS trace into a searchable space of
trajectory primitives. The pipeline runs in five stages:

| Stage | Module | What it produces |
|---|---|---|
| Clean | `clean` | per-ping frame with quality flags and derived kinematics |
| Segment | `segment` | typed motion intervals: `MOVE`, `MOVE_BRIEF`, `STOP_BRIEF`, `STOP_DWELL` |
| Episode | `episode` | grouped `STAY` / `TRANSIT` episodes spanning multiple segments |
| Embed | `embed` | fixed-width float32 vector per segment |
| Compare | `compare` | FAISS index + similarity search over segment vectors |

Each stage is a small module with explicit parameters and a single-entity
contract — the L1 functions take one entity's frame in and return one frame
out. Composition across multiple entities is left to user code.

## Reading order

- **[Pipeline](pipeline.md)** — walkthrough of the stages and what they
  produce.
- **[Parameters](parameters.md)** — the parameter model per stage.

For the *why* behind each design choice, see the per-module design notes:
[`clean`](../design/clean.md) · [`segment`](../design/segment.md) ·
[`episode`](../design/episode.md) · [`embed`](../design/embed.md) ·
[`compare`](../design/compare.md).
