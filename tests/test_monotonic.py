from datetime import datetime

import polars as pl

from quantlint.checks.monotonic import check_monotonic_and_duplicates
from quantlint.schema import Severity


def test_duplicate_timestamp_issue_detected() -> None:
    features = pl.DataFrame(
        {
            "asset": ["A", "A", "A"],
            "ts": [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 9, 31),
                datetime(2024, 1, 1, 9, 31),
            ],
            "f": [1.0, 2.0, 3.0],
        }
    )

    issues = check_monotonic_and_duplicates(
        features=features,
        id_col="asset",
        time_col="ts",
        max_evidence=20,
    )

    issue = next((i for i in issues if i.name == "duplicate_asset_timestamp"), None)
    assert issue is not None
    assert issue.severity == Severity.HIGH
    assert len(issue.evidence) > 0
    assert issue.metrics is not None
    assert issue.metrics["duplicate_keys_count"] == 1
    assert issue.metrics["duplicate_rows_count"] == 2
    assert issue.evidence[0].row_index is not None
