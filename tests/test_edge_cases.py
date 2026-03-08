"""Edge-case reliability tests for quantlint checks."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from quantlint.checks.alignment import check_alignment
from quantlint.checks.calendar import check_calendar_hints
from quantlint.checks.imputation import check_imputation
from quantlint.checks.leakage_rolling import check_rolling_leakage_hint
from quantlint.checks.monotonic import check_monotonic_and_duplicates
from quantlint.checks.splits import check_split_boundary_bleed
from quantlint.schema import LabelTimeMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime),
            "f": pl.Series([], dtype=pl.Float64),
        }
    )


def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )


def _single_row_features() -> pl.DataFrame:
    return pl.DataFrame({"asset": ["A"], "ts": [datetime(2024, 1, 1, 10, 0)], "f": [1.0]})


def _single_row_labels() -> pl.DataFrame:
    return pl.DataFrame({"asset": ["A"], "ts": [datetime(2024, 1, 1, 10, 0, 30)], "y": [0.5]})


# ---------------------------------------------------------------------------
# Empty features/labels
# ---------------------------------------------------------------------------


class TestEmptyData:
    def test_monotonic_empty(self):
        issues = check_monotonic_and_duplicates(
            _empty_features(), id_col="asset", time_col="ts", max_evidence=20
        )
        assert issues == []

    def test_alignment_empty_features(self):
        issues = check_alignment(
            _empty_features(),
            _single_row_labels(),
            id_col="asset",
            time_col="ts",
            label_time_mode=LabelTimeMode.LABEL_TS,
            horizon="1m",
            max_evidence=20,
        )
        # Should produce a no-pairings issue or empty list, not crash
        for issue in issues:
            assert issue.name in {
                "alignment_no_pairings_label_ts_mode",
                "alignment_no_pairings_feature_ts_mode",
            }

    def test_alignment_empty_labels(self):
        issues = check_alignment(
            _single_row_features(),
            _empty_labels(),
            id_col="asset",
            time_col="ts",
            label_time_mode=LabelTimeMode.LABEL_TS,
            horizon="1m",
            max_evidence=20,
        )
        for issue in issues:
            assert issue.name in {
                "alignment_no_pairings_label_ts_mode",
                "alignment_no_pairings_feature_ts_mode",
            }

    def test_imputation_empty(self):
        issues = check_imputation(_empty_features(), id_col="asset", time_col="ts", max_evidence=20)
        assert issues == []

    def test_calendar_empty(self):
        issues = check_calendar_hints(
            _empty_features(), id_col="asset", time_col="ts", max_evidence=20
        )
        assert issues == []

    def test_splits_empty(self):
        issues = check_split_boundary_bleed(
            _empty_features(),
            _empty_labels(),
            id_col="asset",
            time_col="ts",
            label_time_mode=LabelTimeMode.LABEL_TS,
            train_end=datetime(2024, 1, 15),
            val_end=None,
            max_evidence=20,
        )
        assert issues == []


# ---------------------------------------------------------------------------
# Single-asset / single-row
# ---------------------------------------------------------------------------


class TestSingleRow:
    def test_monotonic_single_row(self):
        issues = check_monotonic_and_duplicates(
            _single_row_features(), id_col="asset", time_col="ts", max_evidence=20
        )
        assert issues == []

    def test_alignment_single_row(self):
        issues = check_alignment(
            _single_row_features(),
            _single_row_labels(),
            id_col="asset",
            time_col="ts",
            label_time_mode=LabelTimeMode.LABEL_TS,
            horizon="1m",
            max_evidence=20,
        )
        # Single row should detect look-ahead (label_ts 10:00:30 < feature_ts + 1m = 10:01)
        assert any(i.name == "alignment_lookahead_label_too_early" for i in issues)

    def test_imputation_single_row(self):
        issues = check_imputation(
            _single_row_features(), id_col="asset", time_col="ts", max_evidence=20
        )
        assert issues == []

    def test_calendar_single_row(self):
        issues = check_calendar_hints(
            _single_row_features(), id_col="asset", time_col="ts", max_evidence=20
        )
        # Should not crash; may or may not emit hints
        assert isinstance(issues, list)

    def test_calendar_friday_is_not_weekend(self):
        features = pl.DataFrame(
            {
                "asset": ["A", "A"],
                "ts": [datetime(2024, 1, 5, 10, 0), datetime(2024, 1, 8, 10, 0)],
                "f": [1.0, 2.0],
            }
        )

        issues = check_calendar_hints(features, id_col="asset", time_col="ts", max_evidence=20)
        assert not any(issue.name == "calendar_weekend_ratio_hint" for issue in issues)


# ---------------------------------------------------------------------------
# Massive duplicate keys
# ---------------------------------------------------------------------------


class TestMassiveDuplicates:
    def test_many_duplicate_keys(self):
        n = 200
        features = pl.DataFrame(
            {
                "asset": ["A"] * n,
                "ts": [datetime(2024, 1, 1, 10, 0)] * n,
                "f": list(range(n)),
            }
        )
        issues = check_monotonic_and_duplicates(
            features, id_col="asset", time_col="ts", max_evidence=5
        )
        dup_issues = [i for i in issues if i.name == "duplicate_asset_timestamp"]
        assert len(dup_issues) == 1
        # Evidence should be capped at max_evidence
        assert len(dup_issues[0].evidence) <= 5
        assert dup_issues[0].metrics["duplicate_rows_count"] == n

    def test_many_duplicates_non_monotonic(self):
        n = 200
        features = pl.DataFrame(
            {
                "asset": ["A"] * n,
                "ts": [datetime(2024, 1, 1, 10, 0)] * n,
                "f": list(range(n)),
            }
        )
        issues = check_monotonic_and_duplicates(
            features, id_col="asset", time_col="ts", max_evidence=5
        )
        non_mono = [i for i in issues if i.name == "non_monotonic_timestamp"]
        assert len(non_mono) == 1
        assert len(non_mono[0].evidence) <= 5


# ---------------------------------------------------------------------------
# Missing optional columns for hint checks
# ---------------------------------------------------------------------------


class TestMissingOptionalColumns:
    def test_rolling_missing_cols(self):
        features = pl.DataFrame({"asset": ["A"], "ts": [datetime(2024, 1, 1, 10, 0)], "f": [1.0]})
        issues = check_rolling_leakage_hint(
            features,
            id_col="asset",
            time_col="ts",
            rolling_cols=["nonexistent_rolling_col"],
            max_evidence=20,
        )
        assert any(i.name == "rolling_cols_missing_hint" for i in issues)

    def test_imputation_no_fill_columns(self):
        """Features with no bfill/ffill/filled columns should produce no issues."""
        features = pl.DataFrame(
            {
                "asset": ["A", "A"],
                "ts": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
                "price": [100.0, 101.0],
            }
        )
        issues = check_imputation(features, id_col="asset", time_col="ts", max_evidence=20)
        assert issues == []


# ---------------------------------------------------------------------------
# Multi-asset correctness
# ---------------------------------------------------------------------------


class TestMultiAsset:
    def test_duplicate_only_in_one_asset(self):
        features = pl.DataFrame(
            {
                "asset": ["A", "A", "B"],
                "ts": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "f": [1.0, 2.0, 3.0],
            }
        )
        issues = check_monotonic_and_duplicates(
            features, id_col="asset", time_col="ts", max_evidence=20
        )
        dup_issues = [i for i in issues if i.name == "duplicate_asset_timestamp"]
        assert len(dup_issues) == 1
        # Only asset A should appear in evidence
        assets_in_evidence = {e.asset for e in dup_issues[0].evidence}
        assert assets_in_evidence == {"A"}
