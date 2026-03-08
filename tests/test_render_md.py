"""Tests for markdown report rendering."""

from quantlint.report.render_md import render_markdown_report
from quantlint.schema import EvidenceItem, Issue, ScanResult, Severity, Summary


def _make_result() -> ScanResult:
    return ScanResult(
        summary=Summary(
            total_issues=2,
            by_severity={"CRITICAL": 1, "HIGH": 1, "MEDIUM": 0, "LOW": 0},
            scanned_assets=2,
            features_rows=10,
            labels_rows=10,
            config={},
        ),
        issues=[
            Issue(
                name="alignment_lookahead_label_too_early",
                severity=Severity.CRITICAL,
                description="Found 3 aligned rows with look-ahead.",
                suggestion="Rebuild labels.",
                evidence=[
                    EvidenceItem(asset="A", ts="2024-01-01 10:00:00", columns=["x"], values=[1]),
                ],
                metrics=None,
            ),
            Issue(
                name="duplicate_asset_timestamp",
                severity=Severity.HIGH,
                description="Found 1 duplicate key.",
                suggestion="Deduplicate.",
                evidence=[
                    EvidenceItem(
                        asset="A",
                        ts="2024-01-01 10:01:00",
                        columns=["dup_count"],
                        values=[2],
                        row_index=1,
                    ),
                ],
                metrics={"duplicate_keys_count": 1, "duplicate_rows_count": 2},
            ),
        ],
    )


def test_render_contains_key_sections():
    md = render_markdown_report(_make_result(), max_evidence=20)
    assert "# Quantlint Report" in md
    assert "## Summary" in md
    assert "## CRITICAL (1)" in md
    assert "## HIGH (1)" in md
    assert "## MEDIUM (0)" in md
    assert "## LOW (0)" in md


def test_render_contains_issue_headings():
    md = render_markdown_report(_make_result(), max_evidence=20)
    assert "### " in md
    assert "alignment_lookahead_label_too_early" in md
    assert "duplicate_asset_timestamp" in md


def test_render_contains_evidence_table():
    md = render_markdown_report(_make_result(), max_evidence=20)
    assert "| row_index | asset | ts | columns | values |" in md


def test_render_contains_metrics_table():
    md = render_markdown_report(_make_result(), max_evidence=20)
    assert "| duplicate_keys_count | 1 |" in md


def test_render_contains_anchors():
    md = render_markdown_report(_make_result(), max_evidence=20)
    assert '<a id="alignment-lookahead-label-too-early">' in md
    assert '<a id="duplicate-asset-timestamp">' in md


def test_render_contains_per_issue_summary():
    md = render_markdown_report(_make_result(), max_evidence=20)
    assert "**Severity:** CRITICAL" in md
    assert "**Severity:** HIGH" in md
    assert "**Evidence rows:**" in md


def test_render_metric_float_formatting():
    result = _make_result()
    result.issues[1].metrics = {"rate": 0.5, "count": 3}
    md = render_markdown_report(result, max_evidence=20)
    assert "50.00%" in md
    assert "| count | 3 |" in md
