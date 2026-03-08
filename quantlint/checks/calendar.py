"""Calendar/timezone hint checks."""

from __future__ import annotations

import polars as pl

from quantlint.checks.common import build_evidence
from quantlint.schema import Issue, Severity


def check_calendar_hints(
    features: pl.DataFrame,
    id_col: str,
    time_col: str,
    max_evidence: int,
) -> list[Issue]:
    issues: list[Issue] = []
    if features.height == 0:
        return issues

    enriched = features.with_columns(
        [
            pl.col(time_col).dt.weekday().alias("__weekday"),
            pl.col(time_col).dt.hour().alias("__hour"),
        ]
    )

    # Polars uses ISO weekday numbering (Mon=1 ... Sun=7).
    weekend_mask = pl.col("__weekday") >= 6
    weekend_rows = enriched.filter(weekend_mask)
    weekend_ratio = weekend_rows.height / features.height
    if weekend_ratio > 0:
        sample = weekend_rows.select([id_col, time_col, "__weekday"]).head(max_evidence)
        issues.append(
            Issue(
                name="calendar_weekend_ratio_hint",
                severity=Severity.LOW,
                description=(
                    f"Weekend timestamp ratio in features is {weekend_ratio:.3%} "
                    f"({weekend_rows.height}/{features.height})."
                ),
                suggestion="Confirm this market/instrument should trade on weekends; otherwise verify calendar filters.",
                evidence=build_evidence(
                    sample, id_col=id_col, time_col=time_col, columns=["__weekday"]
                ),
            )
        )

    # Simple non-equity-hours heuristic (outside 09:00-16:59 local timestamp hour)
    offhour_mask = (pl.col("__hour") < 9) | (pl.col("__hour") > 16)
    offhour_rows = enriched.filter(offhour_mask)
    offhour_ratio = offhour_rows.height / features.height
    if offhour_ratio > 0.1:
        sample = offhour_rows.select([id_col, time_col, "__hour"]).head(max_evidence)
        issues.append(
            Issue(
                name="calendar_offhours_ratio_hint",
                severity=Severity.LOW,
                description=(
                    f"Off-hours timestamp ratio in features is {offhour_ratio:.3%} "
                    f"({offhour_rows.height}/{features.height})."
                ),
                suggestion=(
                    "If this is intraday equity data, verify timezone alignment and exchange session filters."
                ),
                evidence=build_evidence(
                    sample, id_col=id_col, time_col=time_col, columns=["__hour"]
                ),
            )
        )

    return issues
