"""Imputation leakage checks (bfill/ffill hints)."""

from __future__ import annotations

import polars as pl

from quantlint.checks.common import build_evidence
from quantlint.schema import Issue, Severity


def _find_pairs(feature_cols: list[str], suffix: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for col in feature_cols:
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in feature_cols:
                pairs.append((base, col))
    return pairs


def check_imputation(
    features: pl.DataFrame,
    id_col: str,
    time_col: str,
    max_evidence: int,
    *,
    sorted_features: pl.DataFrame | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    feature_cols = [c for c in features.columns if c not in {id_col, time_col}]

    bfill_pairs = _find_pairs(feature_cols, "_bfill")
    generic_pairs = _find_pairs(feature_cols, "_filled")
    ffill_pairs = _find_pairs(feature_cols, "_ffill")

    all_bfill_pairs = list(dict.fromkeys(bfill_pairs + generic_pairs))
    all_ffill_pairs = list(dict.fromkeys(ffill_pairs + generic_pairs))

    base = sorted_features if sorted_features is not None else features.sort([id_col, time_col])
    sorted_df = base.with_row_index("__row_nr")

    if all_bfill_pairs:
        bfill_evidence_frames: list[pl.DataFrame] = []
        total_hits = 0
        total_nulls = 0

        for base_col, filled_col in all_bfill_pairs:
            probe = sorted_df.with_columns(
                [
                    pl.col(base_col).backward_fill().over(id_col).alias("__base_bfill"),
                ]
            )
            hits = probe.filter(
                pl.col(base_col).is_null()
                & pl.col(filled_col).is_not_null()
                & (pl.col(filled_col) == pl.col("__base_bfill"))
            ).with_columns(
                [pl.lit(base_col).alias("__base_col"), pl.lit(filled_col).alias("__filled_col")]
            )

            null_count = probe.filter(pl.col(base_col).is_null()).height
            total_nulls += null_count
            total_hits += hits.height
            if hits.height > 0:
                bfill_evidence_frames.append(hits)

        if total_hits > 0:
            evidence_df = (
                pl.concat(bfill_evidence_frames)
                .select(
                    [
                        id_col,
                        time_col,
                        "__base_col",
                        "__filled_col",
                        "__base_bfill",
                        "__row_nr",
                    ]
                )
                .head(max_evidence)
            )
            ratio = (total_hits / total_nulls) if total_nulls else 0.0
            severity = Severity.CRITICAL if (total_hits >= 3 and ratio >= 0.5) else Severity.HIGH
            issues.append(
                Issue(
                    name="imputation_bfill_future_fill",
                    severity=severity,
                    description=(
                        "Detected rows where base feature is null but filled feature equals backward-filled future value "
                        f"within asset timeline (hits={total_hits}, null_base_rows={total_nulls}, ratio={ratio:.3f})."
                    ),
                    suggestion=(
                        "Avoid backward filling across time for training features; prefer forward-only transforms "
                        "or drop rows requiring future values."
                    ),
                    metrics={
                        "hits_count": total_hits,
                        "null_base_rows": total_nulls,
                        "bfill_suspect_rate": ratio,
                        "column_pairs_checked": len(all_bfill_pairs),
                    },
                    evidence=build_evidence(
                        evidence_df,
                        id_col=id_col,
                        time_col=time_col,
                        columns=["__base_col", "__filled_col", "__base_bfill"],
                        row_index_col="__row_nr",
                    ),
                )
            )

    if all_ffill_pairs:
        ffill_evidence_frames: list[pl.DataFrame] = []
        total_hits = 0
        total_nulls = 0

        for base_col, filled_col in all_ffill_pairs:
            probe = sorted_df.with_columns(
                [
                    pl.col(base_col).forward_fill().over(id_col).alias("__base_ffill"),
                ]
            )
            hits = probe.filter(
                pl.col(base_col).is_null()
                & pl.col(filled_col).is_not_null()
                & (pl.col(filled_col) == pl.col("__base_ffill"))
            ).with_columns(
                [pl.lit(base_col).alias("__base_col"), pl.lit(filled_col).alias("__filled_col")]
            )
            null_count = probe.filter(pl.col(base_col).is_null()).height
            total_nulls += null_count
            total_hits += hits.height
            if hits.height > 0:
                ffill_evidence_frames.append(hits)

        if total_hits > 0:
            evidence_df = (
                pl.concat(ffill_evidence_frames)
                .select(
                    [
                        id_col,
                        time_col,
                        "__base_col",
                        "__filled_col",
                        "__base_ffill",
                        "__row_nr",
                    ]
                )
                .head(max_evidence)
            )
            ratio = (total_hits / total_nulls) if total_nulls else 0.0
            severity = Severity.MEDIUM if (total_hits >= 3 and ratio >= 0.5) else Severity.LOW
            issues.append(
                Issue(
                    name="imputation_ffill_hint",
                    severity=severity,
                    description=(
                        "Detected forward-fill style rows where null base values are replaced by historical values "
                        f"(hits={total_hits}, null_base_rows={total_nulls}, ratio={ratio:.3f})."
                    ),
                    suggestion=(
                        "Validate that forward fill does not cross regime boundaries and is only applied where justified."
                    ),
                    metrics={
                        "hits_count": total_hits,
                        "null_base_rows": total_nulls,
                        "ffill_ratio": ratio,
                        "column_pairs_checked": len(all_ffill_pairs),
                    },
                    evidence=build_evidence(
                        evidence_df,
                        id_col=id_col,
                        time_col=time_col,
                        columns=["__base_col", "__filled_col", "__base_ffill"],
                        row_index_col="__row_nr",
                    ),
                )
            )

    return issues
