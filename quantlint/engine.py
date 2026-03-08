from __future__ import annotations

import polars as pl

from quantlint.checks.registry import DEFAULT_CHECKS, CheckContext
from quantlint.io import normalize_features, normalize_labels, read_table
from quantlint.schema import ScanConfig, ScanResult, Severity, Summary, meets_fail_threshold


def _severity_counts(issues) -> dict[str, int]:
    counts = {s.value: 0 for s in Severity}
    for issue in issues:
        counts[issue.severity.value] += 1
    return counts


def _config_snapshot(config: ScanConfig) -> dict[str, object]:
    return {
        "features": config.features,
        "labels": config.labels,
        "id_col": config.id_col,
        "time_col": config.time_col,
        "label_col": config.label_col,
        "label_time_mode": config.label_time_mode.value,
        "horizon": config.horizon,
        "rolling_cols": config.rolling_cols,
        "train_end": str(config.train_end) if config.train_end else None,
        "val_end": str(config.val_end) if config.val_end else None,
        "out_dir": config.out_dir,
        "max_evidence": config.max_evidence,
        "fail_on": config.fail_on.value if config.fail_on else None,
        "format": config.output_format.value,
        "stdout": config.stdout,
    }


def run_scan(config: ScanConfig) -> tuple[ScanResult, int]:
    features_raw = read_table(config.features)
    labels_raw = read_table(config.labels)

    features = normalize_features(features_raw, id_col=config.id_col, time_col=config.time_col)
    labels = normalize_labels(
        labels_raw,
        id_col=config.id_col,
        time_col=config.time_col,
        label_col=config.label_col,
    )

    sort_keys = [config.id_col, config.time_col]
    sorted_features = features.sort(sort_keys)
    sorted_labels = labels.sort(sort_keys)

    ctx = CheckContext(
        features=features,
        labels=labels,
        config=config,
        sorted_features=sorted_features,
        sorted_labels=sorted_labels,
    )

    issues = []
    for check in DEFAULT_CHECKS:
        if check.enabled(config):
            issues.extend(check.run(ctx))

    all_assets = pl.concat(
        [features.select(config.id_col), labels.select(config.id_col)],
        how="vertical_relaxed",
    ).unique()

    summary = Summary(
        total_issues=len(issues),
        by_severity=_severity_counts(issues),
        scanned_assets=all_assets.height,
        features_rows=features.height,
        labels_rows=labels.height,
        config=_config_snapshot(config),
    )
    result = ScanResult(summary=summary, issues=issues)

    fail_hit = any(meets_fail_threshold(issue.severity, config.fail_on) for issue in issues)
    return result, (2 if fail_hit else 0)
