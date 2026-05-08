# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-05-08

### Added

- Initial repository scaffold with module skeletons (`clean`, `segment`, `episode`, `embed`, `compare`, `io`, `runner`, `testing`).
- Design documents under `docs/design/`:
  - `LIBRARY.md` — cross-cutting plan (shape, scope, extraction).
  - `schemas.md` — canonical column schemas (single source of truth).
  - `clean.md`, `segment.md`, `episode.md`, `embed.md`, `compare.md` — per-module designs.
- `pyproject.toml` with core dependencies and `[search]`, `[viz]`, `[fast]`, `[dev]` extras.
- GitHub Actions workflows: `ci.yml` (lint, typecheck, test) and `docs.yml` (mkdocs build).
- MkDocs configuration with material theme and `mkdocstrings`.
- Smoke test asserting package import + version exposure.
