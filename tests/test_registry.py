"""Tests for the check registry."""

from quantlint.checks.registry import DEFAULT_CHECKS
from quantlint.schema import ScanConfig


def test_registry_order():
    expected = ["monotonic", "alignment", "imputation", "splits", "rolling", "calendar"]
    actual = [c.name for c in DEFAULT_CHECKS]
    assert actual == expected


def test_splits_disabled_without_train_end():
    cfg = ScanConfig(
        features="f.csv",
        labels="l.csv",
        label_time_mode="label_ts",
        horizon="1m",
    )
    splits_entry = next(c for c in DEFAULT_CHECKS if c.name == "splits")
    assert splits_entry.enabled(cfg) is False


def test_splits_enabled_with_train_end():
    cfg = ScanConfig(
        features="f.csv",
        labels="l.csv",
        label_time_mode="label_ts",
        horizon="1m",
        train_end="2024-01-15T00:00:00",
    )
    splits_entry = next(c for c in DEFAULT_CHECKS if c.name == "splits")
    assert splits_entry.enabled(cfg) is True


def test_rolling_disabled_without_cols():
    cfg = ScanConfig(
        features="f.csv",
        labels="l.csv",
        label_time_mode="label_ts",
        horizon="1m",
    )
    rolling_entry = next(c for c in DEFAULT_CHECKS if c.name == "rolling")
    assert rolling_entry.enabled(cfg) is False


def test_rolling_enabled_with_cols():
    cfg = ScanConfig(
        features="f.csv",
        labels="l.csv",
        label_time_mode="label_ts",
        horizon="1m",
        rolling_cols=["rolling_px"],
    )
    rolling_entry = next(c for c in DEFAULT_CHECKS if c.name == "rolling")
    assert rolling_entry.enabled(cfg) is True
