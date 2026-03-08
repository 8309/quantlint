"""I/O and normalization utilities for quantlint."""

from __future__ import annotations

from pathlib import Path

import polars as pl


class UnsupportedFormatError(ValueError):
    """Raised when the input table format is unsupported."""


class TableValidationError(ValueError):
    """Raised when required columns are missing or invalid."""


def read_table(path: str) -> pl.DataFrame:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(src, try_parse_dates=True)
    if suffix == ".parquet":
        return pl.read_parquet(src)
    raise UnsupportedFormatError(f"Unsupported file type: {suffix}")


def ensure_required_columns(df: pl.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise TableValidationError(f"{table_name} missing required columns: {missing}")


def normalize_id_time(
    df: pl.DataFrame, id_col: str, time_col: str, table_name: str
) -> pl.DataFrame:
    ensure_required_columns(df, [id_col, time_col], table_name)

    ts_expr = (
        pl.when(pl.col(time_col).is_null())
        .then(None)
        .otherwise(
            pl.coalesce(
                [
                    pl.col(time_col).cast(pl.Datetime, strict=False),
                    pl.col(time_col)
                    .cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Datetime, strict=False),
                ]
            )
        )
    )

    normalized = df.with_columns(
        [
            pl.col(id_col).cast(pl.Utf8, strict=False).alias(id_col),
            ts_expr.alias(time_col),
        ]
    )

    null_ts = normalized.filter(pl.col(time_col).is_null()).height
    if null_ts > 0:
        raise TableValidationError(
            f"{table_name}.{time_col} has {null_ts} null/unparseable timestamps"
        )

    return normalized


def normalize_labels(
    labels: pl.DataFrame,
    id_col: str,
    time_col: str,
    label_col: str,
) -> pl.DataFrame:
    ensure_required_columns(labels, [id_col, time_col, label_col], "labels")
    normalized = normalize_id_time(labels, id_col=id_col, time_col=time_col, table_name="labels")
    return normalized.with_columns(
        pl.col(label_col).cast(pl.Float64, strict=False).alias(label_col)
    )


def normalize_features(features: pl.DataFrame, id_col: str, time_col: str) -> pl.DataFrame:
    ensure_required_columns(features, [id_col, time_col], "features")
    return normalize_id_time(features, id_col=id_col, time_col=time_col, table_name="features")
