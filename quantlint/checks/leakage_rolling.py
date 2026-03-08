"""Rolling leakage hint checks."""

from __future__ import annotations

import polars as pl

from quantlint.checks.common import build_evidence
from quantlint.schema import Issue, Severity


def _guess_base_column(rolling_col: str, columns: set[str]) -> str | None:
    candidates = []
    if rolling_col.startswith("rolling_"):
        candidates.append(rolling_col.replace("rolling_", "", 1))
    if rolling_col.startswith("roll_"):
        candidates.append(rolling_col.replace("roll_", "", 1))
    if rolling_col.endswith("_rolling"):
        candidates.append(rolling_col[: -len("_rolling")])
    if rolling_col.endswith("_roll"):
        candidates.append(rolling_col[: -len("_roll")])

    for c in candidates:
        if c in columns:
            return c
    return None


def check_rolling_leakage_hint(
    features: pl.DataFrame,
    id_col: str,
    time_col: str,
    rolling_cols: list[str],
    max_evidence: int,
) -> list[Issue]:
    if not rolling_cols:
        return []

    issues: list[Issue] = []
    cols = set(features.columns)

    missing = [c for c in rolling_cols if c not in cols]
    if missing:
        issues.append(
            Issue(
                name="rolling_cols_missing_hint",
                severity=Severity.LOW,
                description=f"rolling-cols contains missing columns: {missing}",
                suggestion="Fix --rolling-cols list or ensure these columns exist in features.",
                evidence=[],
            )
        )

    for rolling_col in rolling_cols:
        if rolling_col not in cols:
            continue
        base_col = _guess_base_column(rolling_col, cols)
        if base_col is None:
            continue

        comparable = features.filter(
            pl.col(base_col).is_not_null() & pl.col(rolling_col).is_not_null()
        )
        if comparable.height == 0:
            continue

        eq_rows = comparable.filter(pl.col(base_col) == pl.col(rolling_col))
        eq_ratio = eq_rows.height / comparable.height
        if eq_ratio >= 0.98 and comparable.height >= 5:
            sample = eq_rows.select([id_col, time_col, base_col, rolling_col]).head(max_evidence)
            issues.append(
                Issue(
                    name="rolling_not_shifted_hint",
                    severity=Severity.LOW,
                    description=(
                        f"Rolling column `{rolling_col}` is almost identical to `{base_col}` "
                        f"(equality ratio={eq_ratio:.3f}, rows={comparable.height}). "
                        "This can indicate rolling features were not shifted."
                    ),
                    suggestion=(
                        "Verify rolling feature pipeline applies a one-step lag before model training."
                    ),
                    evidence=build_evidence(
                        sample,
                        id_col=id_col,
                        time_col=time_col,
                        columns=[base_col, rolling_col],
                    ),
                )
            )

    return issues
