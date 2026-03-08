from datetime import datetime

import polars as pl

from quantlint.checks.imputation import check_imputation
from quantlint.schema import Severity


def test_bfill_issue_detected() -> None:
    features = pl.DataFrame(
        {
            "asset": ["A", "A", "A"],
            "ts": [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 9, 31),
                datetime(2024, 1, 1, 9, 32),
            ],
            "px": [None, 10.0, 11.0],
            "px_bfill": [10.0, 10.0, 11.0],
        }
    )

    issues = check_imputation(features=features, id_col="asset", time_col="ts", max_evidence=20)

    issue = next((i for i in issues if i.name == "imputation_bfill_future_fill"), None)
    assert issue is not None
    assert issue.severity == Severity.HIGH
    assert len(issue.evidence) > 0
    assert issue.metrics is not None
    assert issue.metrics["hits_count"] == 1
    assert issue.metrics["null_base_rows"] == 1
    assert issue.metrics["bfill_suspect_rate"] == 1.0
    assert issue.evidence[0].row_index is not None
