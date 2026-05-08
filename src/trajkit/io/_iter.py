"""L2 entity iterator: dispatch sources, coerce, sort, validate, yield.

Implements ``iter_entities``. The contract from LIBRARY.md §5:

* Yields ``(entity_id, pings_df)`` tuples.
* Each yielded frame is single-entity, sorted by ``ts``.
* Each frame validates against ``PingsSchema`` (raises on invalid).
* Light coercion happens once at the boundary — string entity_id,
  tz-aware UTC timestamps, optional ``speed_ms`` / ``bearing_deg``
  added as null if absent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from trajkit.types import PingsSchema

_logger = logging.getLogger(__name__)
_csv_warned = False


def iter_entities(
    source: str | Path | pa.Table | pd.DataFrame,
    *,
    format: Literal["auto", "parquet", "csv"] = "auto",  # noqa: A002
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(entity_id, pings_df)`` tuples from a heterogeneous source.

    Parameters
    ----------
    source
        One of:

        * Path or string pointing to a parquet directory / file (the
          canonical layout) or a CSV file.
        * ``pyarrow.Table`` already in memory.
        * ``pandas.DataFrame`` already in memory.

    format
        ``"auto"`` (default) infers from the path extension. ``"parquet"``
        and ``"csv"`` force explicitly.

    Yields
    ------
    tuple[str, pd.DataFrame]
        ``entity_id`` and a single-entity frame validated against
        ``PingsSchema``. Frames are sorted by ``ts`` and reset to a fresh
        ``RangeIndex``.

    Raises
    ------
    pandera.errors.SchemaError | SchemaErrors
        When a per-entity frame fails ``PingsSchema``.
    ValueError
        For unrecognised ``format`` or sources missing required columns.
    """
    df = _load_source(source, format)
    df = _coerce_canonical_dtypes(df)
    df = df.sort_values(["entity_id", "ts"]).reset_index(drop=True)

    for entity_id, group in df.groupby("entity_id", sort=False):
        sub = group.reset_index(drop=True)
        PingsSchema.validate(sub)
        yield str(entity_id), sub


# ── Source dispatch ─────────────────────────────────────────────────


def _load_source(
    source: str | Path | pa.Table | pd.DataFrame,
    format: Literal["auto", "parquet", "csv"],  # noqa: A002
) -> pd.DataFrame:
    """Resolve any supported source type to a single pandas DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, pa.Table):
        # Use pandas-native dtypes (not ``pd.ArrowDtype``) so downstream
        # tz-aware-datetime handling and string coercion match the DataFrame
        # path; ArrowDtype timestamps require different localisation handling.
        return source.to_pandas()
    if isinstance(source, str | Path):
        return _load_path(Path(source), format)
    msg = f"unsupported source type: {type(source).__name__}"
    raise TypeError(msg)


def _load_path(
    path: Path,
    format: Literal["auto", "parquet", "csv"],  # noqa: A002
) -> pd.DataFrame:
    """Load a parquet file/directory or CSV file from disk."""
    resolved = format
    if resolved == "auto":
        resolved = (
            "csv" if path.is_file() and path.suffix.lower() == ".csv" else "parquet"
        )

    if resolved not in ("parquet", "csv"):
        msg = f"unknown format {resolved!r}"
        raise ValueError(msg)

    if not path.exists():
        msg = f"source path does not exist: {path}"
        raise FileNotFoundError(msg)

    if resolved == "parquet":
        return pd.read_parquet(path)
    if resolved == "csv":
        global _csv_warned
        if not _csv_warned:
            _logger.warning(
                "iter_entities: reading CSV is supported for tutorials but "
                "not recommended for production data — type inference, mixed "
                "encodings, and null sentinels are common pitfalls. Convert "
                "to parquet for production pipelines."
            )
            _csv_warned = True
        return pd.read_csv(path)

    # Unreachable: format set was validated above.
    raise AssertionError(f"unhandled format {resolved!r}")  # pragma: no cover


# ── Boundary coercion ───────────────────────────────────────────────


_REQUIRED_COLUMNS: tuple[str, ...] = ("entity_id", "ts", "lat", "lon")
_OPTIONAL_NULLABLE_F32: tuple[str, ...] = ("speed_ms", "bearing_deg")


def _coerce_canonical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Light coercion to the canonical PingsSchema dtypes.

    Done once at the L2 boundary so users with cosmetic dtype differences
    (object entity_id from CSV, naive datetimes) succeed without manual
    pre-processing. Beyond this, ``PingsSchema.validate`` is strict.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"source missing required columns: {missing}"
        raise ValueError(msg)

    out = df.copy()
    out["entity_id"] = out["entity_id"].astype("string")
    out["ts"] = _coerce_ts_utc_ns(out["ts"])
    out["lat"] = out["lat"].astype(np.float64)
    out["lon"] = out["lon"].astype(np.float64)

    for col in _OPTIONAL_NULLABLE_F32:
        if col not in out.columns:
            out[col] = pd.array([np.nan] * len(out), dtype="float32")
        else:
            out[col] = out[col].astype(np.float32)

    return out


def _coerce_ts_utc_ns(ts: pd.Series) -> pd.Series:
    """Force a timestamp column to ``datetime64[ns, UTC]``.

    Naive datetimes are treated as already-UTC and localised; tz-aware
    datetimes in another zone are converted to UTC. Non-datetime input
    triggers ``pd.to_datetime`` parsing.
    """
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, utc=True, errors="raise")
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    return ts.astype("datetime64[ns, UTC]")
