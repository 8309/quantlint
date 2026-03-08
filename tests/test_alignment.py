from datetime import datetime

import polars as pl

from quantlint.checks.alignment import check_alignment
from quantlint.schema import LabelTimeMode, Severity


def test_label_ts_lookahead_detected_as_critical() -> None:
    features = pl.DataFrame(
        {
            "asset": ["A", "A"],
            "ts": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 1),
            ],
            "f": [1.0, 2.0],
        }
    )
    labels = pl.DataFrame(
        {
            "asset": ["A", "A"],
            "ts": [
                datetime(2024, 1, 1, 10, 0, 20),
                datetime(2024, 1, 1, 10, 1, 20),
            ],
            "y": [0.1, 0.2],
        }
    )

    issues = check_alignment(
        features=features,
        labels=labels,
        id_col="asset",
        time_col="ts",
        label_time_mode=LabelTimeMode.LABEL_TS,
        horizon="1m",
        max_evidence=20,
    )

    issue = next((i for i in issues if i.name == "alignment_lookahead_label_too_early"), None)
    assert issue is not None
    assert issue.severity == Severity.CRITICAL
    assert len(issue.evidence) > 0
