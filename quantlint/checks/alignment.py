"""Label/feature alignment checks."""

from __future__ import annotations

import re

import polars as pl

from quantlint.checks.common import build_evidence
from quantlint.schema import Issue, LabelTimeMode, Severity

_HORIZON_RE = re.compile(r"^(\d+)([mMhHdDwW])$")


def _parse_horizon(horizon: str) -> tuple[int, str]:
    match = _HORIZON_RE.match(horizon)
    if not match:
        raise ValueError(f"Invalid horizon: {horizon}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    return value, unit


def _horizon_duration_expr(horizon: str) -> pl.Expr:
    value, unit = _parse_horizon(horizon)
    if unit == "m":
        return pl.duration(minutes=value)
    if unit == "h":
        return pl.duration(hours=value)
    if unit == "d":
        return pl.duration(days=value)
    if unit == "w":
        return pl.duration(weeks=value)
    raise ValueError(f"Unsupported horizon unit: {unit}")


def _with_group_index(
    df: pl.DataFrame, id_col: str, time_col: str, *, presorted: bool = False
) -> pl.DataFrame:
    base = df if presorted else df.sort([id_col, time_col])
    return base.with_columns((pl.col(time_col).cum_count().over(id_col) - 1).alias("__idx"))


def _check_label_ts_mode(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    id_col: str,
    time_col: str,
    horizon: str,
    max_evidence: int,
    *,
    sorted_features: pl.DataFrame | None = None,
    sorted_labels: pl.DataFrame | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    f_base = sorted_features if sorted_features is not None else features
    l_base = sorted_labels if sorted_labels is not None else labels
    presorted = sorted_features is not None
    f = _with_group_index(f_base, id_col=id_col, time_col=time_col, presorted=presorted).select(
        [id_col, pl.col(time_col).alias("feature_ts"), "__idx"]
    )
    labels_indexed = _with_group_index(
        l_base, id_col=id_col, time_col=time_col, presorted=sorted_labels is not None
    ).select([id_col, pl.col(time_col).alias("label_ts"), "__idx"])

    aligned = f.join(labels_indexed, on=[id_col, "__idx"], how="inner")
    if aligned.height == 0:
        sample = (
            f.select([id_col, pl.col("feature_ts").alias(time_col), "__idx"]).head(max_evidence)
            if f.height > 0
            else labels_indexed.select([id_col, pl.col("label_ts").alias(time_col), "__idx"]).head(
                max_evidence
            )
        )
        issues.append(
            Issue(
                name="alignment_no_pairings_label_ts_mode",
                severity=Severity.HIGH,
                description=(
                    "Could not align any feature/label rows by per-asset temporal index in label_ts mode; "
                    "alignment quality cannot be guaranteed."
                ),
                suggestion="Check feature/label sampling cadence and per-asset row counts before training.",
                evidence=build_evidence(
                    sample, id_col=id_col, time_col=time_col, columns=["__idx"]
                ),
            )
        )
        return issues

    violations = aligned.with_columns(
        (pl.col("feature_ts") + _horizon_duration_expr(horizon)).alias("min_label_ts")
    ).filter(pl.col("label_ts") < pl.col("min_label_ts"))

    if violations.height > 0:
        sample = violations.select(
            [
                id_col,
                pl.col("feature_ts").alias(time_col),
                "label_ts",
                "min_label_ts",
                "__idx",
            ]
        ).head(max_evidence)
        issues.append(
            Issue(
                name="alignment_lookahead_label_too_early",
                severity=Severity.CRITICAL,
                description=(
                    f"Found {violations.height} aligned rows where label_ts < feature_ts + horizon ({horizon}). "
                    "This is a direct look-ahead violation."
                ),
                suggestion=(
                    "Rebuild labels so each feature row only observes outcomes at or after feature_ts + horizon."
                ),
                evidence=build_evidence(
                    sample,
                    id_col=id_col,
                    time_col=time_col,
                    columns=["label_ts", "min_label_ts", "__idx"],
                ),
            )
        )

    return issues


def _check_feature_ts_mode(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    id_col: str,
    time_col: str,
    max_evidence: int,
    *,
    sorted_features: pl.DataFrame | None = None,
    sorted_labels: pl.DataFrame | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    f_base = sorted_features if sorted_features is not None else features
    l_base = sorted_labels if sorted_labels is not None else labels
    presorted = sorted_features is not None
    f = _with_group_index(f_base, id_col=id_col, time_col=time_col, presorted=presorted).select(
        [id_col, pl.col(time_col).alias("feature_ts"), "__idx"]
    )
    labels_indexed = _with_group_index(
        l_base, id_col=id_col, time_col=time_col, presorted=sorted_labels is not None
    ).select([id_col, pl.col(time_col).alias("label_ts"), "__idx"])

    aligned = f.join(labels_indexed, on=[id_col, "__idx"], how="inner").sort([id_col, "__idx"])
    if aligned.height == 0:
        sample = (
            f.select([id_col, pl.col("feature_ts").alias(time_col), "__idx"]).head(max_evidence)
            if f.height > 0
            else labels_indexed.select([id_col, pl.col("label_ts").alias(time_col), "__idx"]).head(
                max_evidence
            )
        )
        issues.append(
            Issue(
                name="alignment_no_pairings_feature_ts_mode",
                severity=Severity.HIGH,
                description=(
                    "Could not align any feature/label rows by per-asset temporal index in feature_ts mode; "
                    "future label contamination risk cannot be ruled out."
                ),
                suggestion="Ensure labels are generated on the same (asset, ts) grid as features.",
                evidence=build_evidence(
                    sample, id_col=id_col, time_col=time_col, columns=["__idx"]
                ),
            )
        )
        return issues

    aligned = aligned.with_columns(
        [
            pl.col("feature_ts").shift(-1).over(id_col).alias("__feature_next"),
            pl.col("feature_ts").shift(1).over(id_col).alias("__feature_prev"),
        ]
    ).with_columns(
        [
            (pl.col("label_ts") == pl.col("feature_ts")).alias("eq0"),
            (pl.col("label_ts") == pl.col("__feature_next")).alias("eq_next"),
            (pl.col("label_ts") == pl.col("__feature_prev")).alias("eq_prev"),
        ]
    )

    stats = aligned.select(
        [
            pl.len().alias("n"),
            pl.col("eq0").mean().alias("eq0_ratio"),
            pl.col("eq_next").mean().alias("eq_next_ratio"),
            pl.col("eq_prev").mean().alias("eq_prev_ratio"),
        ]
    ).to_dicts()[0]

    feature_keys = features.select([id_col, time_col]).unique()
    label_keys = labels.select([id_col, time_col]).unique()
    matched = feature_keys.join(label_keys, on=[id_col, time_col], how="inner").height
    unmatched_features = feature_keys.height - matched
    unmatched_labels = label_keys.height - matched

    eq0_ratio = float(stats["eq0_ratio"] or 0.0)
    eq_next_ratio = float(stats["eq_next_ratio"] or 0.0)
    eq_prev_ratio = float(stats["eq_prev_ratio"] or 0.0)
    shifted_ratio = max(eq_next_ratio, eq_prev_ratio)

    if eq0_ratio < 0.8 and shifted_ratio >= 0.95:
        sample = (
            aligned.filter(pl.col("eq_next") | pl.col("eq_prev"))
            .select(
                [
                    id_col,
                    pl.col("feature_ts").alias(time_col),
                    "label_ts",
                    "eq0",
                    "eq_next",
                    "eq_prev",
                    "__idx",
                ]
            )
            .head(max_evidence)
        )
        issues.append(
            Issue(
                name="alignment_feature_ts_one_step_shift_risk",
                severity=Severity.HIGH,
                description=(
                    "In feature_ts mode, exact ts match ratio is low while one-step shifted ts match ratio is very high; "
                    "this strongly suggests label shift/misalignment leakage risk."
                ),
                suggestion="Verify label generation shift direction and align labels on the exact feature timestamp grid.",
                evidence=build_evidence(
                    sample,
                    id_col=id_col,
                    time_col=time_col,
                    columns=["label_ts", "eq0", "eq_next", "eq_prev", "__idx"],
                ),
            )
        )
    elif eq0_ratio < 0.95 or unmatched_features > 0 or unmatched_labels > 0:
        sample = (
            aligned.filter(~pl.col("eq0"))
            .select([id_col, pl.col("feature_ts").alias(time_col), "label_ts", "eq0", "__idx"])
            .head(max_evidence)
        )
        issues.append(
            Issue(
                name="alignment_feature_ts_mismatch_risk",
                severity=Severity.MEDIUM,
                description=(
                    "Feature/label timestamps do not consistently match in feature_ts mode. "
                    f"eq0_ratio={eq0_ratio:.3f}, unmatched_features={unmatched_features}, "
                    f"unmatched_labels={unmatched_labels}."
                ),
                suggestion="Audit timestamp joins and verify labels are keyed by the same (asset, ts) as features.",
                evidence=build_evidence(
                    sample,
                    id_col=id_col,
                    time_col=time_col,
                    columns=["label_ts", "eq0", "__idx"],
                ),
            )
        )

    return issues


def check_alignment(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    id_col: str,
    time_col: str,
    label_time_mode: LabelTimeMode,
    horizon: str,
    max_evidence: int,
    *,
    sorted_features: pl.DataFrame | None = None,
    sorted_labels: pl.DataFrame | None = None,
) -> list[Issue]:
    if label_time_mode == LabelTimeMode.LABEL_TS:
        return _check_label_ts_mode(
            features=features,
            labels=labels,
            id_col=id_col,
            time_col=time_col,
            horizon=horizon,
            max_evidence=max_evidence,
            sorted_features=sorted_features,
            sorted_labels=sorted_labels,
        )

    return _check_feature_ts_mode(
        features=features,
        labels=labels,
        id_col=id_col,
        time_col=time_col,
        max_evidence=max_evidence,
        sorted_features=sorted_features,
        sorted_labels=sorted_labels,
    )
