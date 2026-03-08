"""Optional HTML report renderer placeholder.

MVP intentionally emits `results.json` and `report.md` only.
"""

from __future__ import annotations

from quantlint.schema import ScanResult


def render_html_report(result: ScanResult, max_evidence: int) -> str:
    """Return a minimal HTML representation of summary and issue count."""
    return (
        "<html><head><meta charset='utf-8'><title>Quantlint Report</title></head>"
        f"<body><h1>Quantlint Report</h1><p>Total issues: {result.summary.total_issues}</p>"
        "<p>Detailed markdown report is generated as report.md in MVP.</p></body></html>"
    )
