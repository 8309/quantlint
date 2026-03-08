"""Split-boundary leakage checks."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from quantlint.checks.alignment import _with_group_index
from quantlint.checks.common import build_evidence
from quantlint.schema import Issue, LabelTimeMode, Severity


def check_split_boundary_bleed(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    id_col: str,
    time_col: str,
    label_time_mode: LabelTimeMode,
    train_end: datetime | None,
    val_end: datetime | None,
    max_evidence: int,
    *,
    sorted_features: pl.DataFrame | None = None,
    sorted_labels: pl.DataFrame | None = None,
) -> list[Issue]:
    if train_end is None and val_end is None:
        return []

    issues: list[Issue] = []

    if label_time_mode == LabelTimeMode.LABEL_TS and train_end is not None:
        f_base = sorted_features if sorted_features is not None else features
        l_base = sorted_labels if sorted_labels is not None else labels
        presorted = sorted_features is not None
        f = _with_group_index(f_base, id_col, time_col, presorted=presorted).select(
            [id_col, pl.col(time_col).alias("feature_ts"), "__idx"]
        )
        labels_indexed = _with_group_index(
            l_base, id_col, time_col, presorted=sorted_labels is not None
        ).select([id_col, pl.col(time_col).alias("label_ts"), "__idx"])
        aligned = f.join(labels_indexed, on=[id_col, "__idx"], how="inner")

        train_bleed = aligned.filter(
            (pl.col("feature_ts") <= pl.lit(train_end)) & (pl.col("label_ts") > pl.lit(train_end))
        )
        if train_bleed.height > 0:
            sample = train_bleed.select(
                [id_col, pl.col("feature_ts").alias(time_col), "label_ts", "__idx"]
            ).head(max_evidence)
            issues.append(
                Issue(
                    name="split_train_boundary_label_bleed",
                    severity=Severity.HIGH,
                    description=(
                        f"Found {train_bleed.height} rows where feature_ts <= train_end ({train_end.isoformat()}) "
                        "but aligned label_ts is after train_end."
                    ),
                    suggestion=(
                        "Build train split by feature timestamp and enforce label horizon windows to avoid leakage "
                        "across split boundaries."
                    ),
                    evidence=build_evidence(
                        sample,
                        id_col=id_col,
                        time_col=time_col,
                        columns=["label_ts", "__idx"],
                    ),
                )
            )

        if val_end is not None:
            val_bleed = aligned.filter(
                (pl.col("feature_ts") <= pl.lit(val_end)) & (pl.col("label_ts") > pl.lit(val_end))
            )
            if val_bleed.height > 0:
                sample = val_bleed.select(
                    [id_col, pl.col("feature_ts").alias(time_col), "label_ts", "__idx"]
                ).head(max_evidence)
                issues.append(
                    Issue(
                        name="split_val_boundary_label_bleed",
                        severity=Severity.HIGH,
                        description=(
                            f"Found {val_bleed.height} rows where feature_ts <= val_end ({val_end.isoformat()}) "
                            "but aligned label_ts is after val_end."
                        ),
                        suggestion="Apply strict split masks after label construction and verify by timestamp joins.",
                        evidence=build_evidence(
                            sample,
                            id_col=id_col,
                            time_col=time_col,
                            columns=["label_ts", "__idx"],
                        ),
                    )
                )

    boundary_ref = val_end if val_end is not None else train_end
    assert boundary_ref is not None

    start = boundary_ref - timedelta(days=1)
    end = boundary_ref + timedelta(days=1)

    feature_win = features.filter(
        (pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) <= pl.lit(end))
    ).select([id_col, time_col])
    label_win = labels.filter(
        (pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) <= pl.lit(end))
    ).select([id_col, time_col])

    feature_only = feature_win.join(label_win, on=[id_col, time_col], how="anti")
    label_only = label_win.join(feature_win, on=[id_col, time_col], how="anti")

    mismatch_count = feature_only.height + label_only.height
    if mismatch_count > 0:
        sample = pl.concat(
            [
                feature_only.with_columns(pl.lit("feature_only").alias("side")),
                label_only.with_columns(pl.lit("label_only").alias("side")),
            ],
            how="vertical_relaxed",
        ).head(max_evidence)

        issues.append(
            Issue(
                name="split_boundary_key_mismatch_risk",
                severity=Severity.MEDIUM,
                description=(
                    "Detected feature/label key mismatches near split boundary window "
                    f"[{start.isoformat()}, {end.isoformat()}], total mismatches={mismatch_count}."
                ),
                suggestion=(
                    "Reconcile feature/label keys around split cutoffs and confirm deterministic join logic "
                    "before training/validation separation."
                ),
                evidence=build_evidence(sample, id_col=id_col, time_col=time_col, columns=["side"]),
            )
        )

    return issues
