"""Monotonicity and duplicate timestamp checks."""

from __future__ import annotations

import polars as pl

from quantlint.checks.common import build_evidence
from quantlint.schema import Issue, Severity


def check_monotonic_and_duplicates(
    features: pl.DataFrame,
    id_col: str,
    time_col: str,
    max_evidence: int,
) -> list[Issue]:
    issues: list[Issue] = []
    indexed = features.with_row_index("__row_nr")

    dup_keys = (
        indexed.group_by([id_col, time_col])
        .agg(pl.len().alias("dup_count"))
        .filter(pl.col("dup_count") > 1)
    )

    if dup_keys.height > 0:
        dup_rows_all = indexed.join(dup_keys, on=[id_col, time_col], how="inner").sort(
            [id_col, time_col, "__row_nr"]
        )
        dup_rows = dup_rows_all.head(max_evidence)
        issues.append(
            Issue(
                name="duplicate_asset_timestamp",
                severity=Severity.HIGH,
                description=(
                    f"Found {dup_keys.height} duplicate (asset, ts) keys in features. "
                    "Duplicate keys can corrupt joins and training label alignment."
                ),
                suggestion="Deduplicate by (asset, ts) or define deterministic aggregation before training.",
                metrics={
                    "duplicate_keys_count": dup_keys.height,
                    "duplicate_rows_count": dup_rows_all.height,
                },
                evidence=build_evidence(
                    dup_rows,
                    id_col,
                    time_col,
                    ["dup_count"],
                    row_index_col="__row_nr",
                ),
            )
        )

    monotonic_df = indexed.with_columns(
        [
            pl.col(time_col).shift(1).over(id_col).alias("__prev_ts"),
            pl.col("__row_nr").shift(1).over(id_col).alias("__prev_row_nr"),
        ]
    )

    non_mono = monotonic_df.filter(
        pl.col("__prev_ts").is_not_null() & (pl.col(time_col) <= pl.col("__prev_ts"))
    )

    if non_mono.height > 0:
        sample = non_mono.select([id_col, time_col, "__prev_ts", "__prev_row_nr", "__row_nr"]).head(
            max_evidence
        )
        issues.append(
            Issue(
                name="non_monotonic_timestamp",
                severity=Severity.HIGH,
                description=(
                    f"Found {non_mono.height} rows where timestamp is not strictly increasing within asset "
                    "in input order."
                ),
                suggestion="Sort each asset by timestamp and remove out-of-order records before feature/label joins.",
                metrics={
                    "violations_count": non_mono.height,
                },
                evidence=build_evidence(
                    sample,
                    id_col=id_col,
                    time_col=time_col,
                    columns=["__prev_ts", "__prev_row_nr", "__row_nr"],
                    row_index_col="__row_nr",
                ),
            )
        )

    return issues
