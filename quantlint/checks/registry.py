"""Check registry: ordered list of checks with config-gated execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from quantlint.checks.alignment import check_alignment
from quantlint.checks.calendar import check_calendar_hints
from quantlint.checks.imputation import check_imputation
from quantlint.checks.leakage_rolling import check_rolling_leakage_hint
from quantlint.checks.monotonic import check_monotonic_and_duplicates
from quantlint.checks.splits import check_split_boundary_bleed
from quantlint.schema import Issue, ScanConfig


@dataclass(frozen=True)
class CheckContext:
    features: pl.DataFrame
    labels: pl.DataFrame
    config: ScanConfig
    sorted_features: pl.DataFrame | None = None
    sorted_labels: pl.DataFrame | None = None


@dataclass(frozen=True)
class CheckEntry:
    name: str
    run: Callable[[CheckContext], list[Issue]]
    enabled: Callable[[ScanConfig], bool] = lambda _cfg: True


def _run_monotonic(ctx: CheckContext) -> list[Issue]:
    return check_monotonic_and_duplicates(
        features=ctx.features,
        id_col=ctx.config.id_col,
        time_col=ctx.config.time_col,
        max_evidence=ctx.config.max_evidence,
    )


def _run_alignment(ctx: CheckContext) -> list[Issue]:
    return check_alignment(
        features=ctx.features,
        labels=ctx.labels,
        id_col=ctx.config.id_col,
        time_col=ctx.config.time_col,
        label_time_mode=ctx.config.label_time_mode,
        horizon=ctx.config.horizon,
        max_evidence=ctx.config.max_evidence,
        sorted_features=ctx.sorted_features,
        sorted_labels=ctx.sorted_labels,
    )


def _run_imputation(ctx: CheckContext) -> list[Issue]:
    return check_imputation(
        features=ctx.features,
        id_col=ctx.config.id_col,
        time_col=ctx.config.time_col,
        max_evidence=ctx.config.max_evidence,
        sorted_features=ctx.sorted_features,
    )


def _run_splits(ctx: CheckContext) -> list[Issue]:
    return check_split_boundary_bleed(
        features=ctx.features,
        labels=ctx.labels,
        id_col=ctx.config.id_col,
        time_col=ctx.config.time_col,
        label_time_mode=ctx.config.label_time_mode,
        train_end=ctx.config.train_end,
        val_end=ctx.config.val_end,
        max_evidence=ctx.config.max_evidence,
        sorted_features=ctx.sorted_features,
        sorted_labels=ctx.sorted_labels,
    )


def _run_rolling(ctx: CheckContext) -> list[Issue]:
    return check_rolling_leakage_hint(
        features=ctx.features,
        id_col=ctx.config.id_col,
        time_col=ctx.config.time_col,
        rolling_cols=ctx.config.rolling_cols,
        max_evidence=ctx.config.max_evidence,
    )


def _run_calendar(ctx: CheckContext) -> list[Issue]:
    return check_calendar_hints(
        features=ctx.features,
        id_col=ctx.config.id_col,
        time_col=ctx.config.time_col,
        max_evidence=ctx.config.max_evidence,
    )


DEFAULT_CHECKS: list[CheckEntry] = [
    CheckEntry(name="monotonic", run=_run_monotonic),
    CheckEntry(name="alignment", run=_run_alignment),
    CheckEntry(name="imputation", run=_run_imputation),
    CheckEntry(
        name="splits",
        run=_run_splits,
        enabled=lambda cfg: cfg.train_end is not None or cfg.val_end is not None,
    ),
    CheckEntry(
        name="rolling",
        run=_run_rolling,
        enabled=lambda cfg: len(cfg.rolling_cols) > 0,
    ),
    CheckEntry(name="calendar", run=_run_calendar),
]
