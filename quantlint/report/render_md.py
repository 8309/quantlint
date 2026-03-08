from __future__ import annotations

from collections import defaultdict

from quantlint.schema import ScanResult

_SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def _summary_table(result: ScanResult) -> list[str]:
    s = result.summary
    by_severity = s.by_severity
    lines = [
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total_issues | {s.total_issues} |",
        f"| features_rows | {s.features_rows} |",
        f"| labels_rows | {s.labels_rows} |",
        f"| scanned_assets | {s.scanned_assets} |",
        f"| critical | {by_severity.get('CRITICAL', 0)} |",
        f"| high | {by_severity.get('HIGH', 0)} |",
        f"| medium | {by_severity.get('MEDIUM', 0)} |",
        f"| low | {by_severity.get('LOW', 0)} |",
        "",
    ]
    return lines


def _fmt_metric_value(value: object) -> str:
    if isinstance(value, float):
        if abs(value) <= 1.0:
            return f"{value:.2%}"
        return f"{value:.4f}"
    return str(value)


def _metrics_table(metrics: dict[str, object]) -> list[str]:
    lines = [
        "Metrics:",
        "",
        "| key | value |",
        "|---|---:|",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"| {key} | {_fmt_metric_value(value)} |")
    lines.append("")
    return lines


def _compact(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def render_markdown_report(result: ScanResult, max_evidence: int) -> str:
    lines: list[str] = ["# Quantlint Report", ""]
    lines.extend(_summary_table(result))

    grouped = defaultdict(list)
    for issue in result.issues:
        grouped[issue.severity.value].append(issue)

    for severity in _SEVERITY_ORDER:
        issues = grouped.get(severity, [])
        lines.append(f"## {severity} ({len(issues)})")
        lines.append("")

        if not issues:
            lines.append("No issues.")
            lines.append("")
            continue

        for issue in issues:
            anchor = issue.name.replace("_", "-")
            lines.append(f'### <a id="{anchor}"></a>{issue.name}')
            lines.append("")
            evidence_count = len(issue.evidence)
            lines.append(
                f"> **Severity:** {issue.severity.value} | **Evidence rows:** {evidence_count}"
            )
            lines.append("")
            lines.append(issue.description)
            lines.append("")
            lines.append(f"**Suggestion:** {issue.suggestion}")
            lines.append("")

            if issue.metrics:
                lines.extend(_metrics_table(issue.metrics))

            lines.append(f"Evidence (showing up to {max_evidence} rows):")
            lines.append("")
            lines.append("| row_index | asset | ts | columns | values |")
            lines.append("|---:|---|---|---|---|")
            for ev in issue.evidence[:max_evidence]:
                cols = _compact(", ".join(ev.columns))
                vals = _compact(", ".join(str(v) for v in ev.values))
                row_index = "" if ev.row_index is None else str(ev.row_index)
                lines.append(f"| {row_index} | {ev.asset} | {ev.ts} | {cols} | {vals} |")
            if not issue.evidence:
                lines.append("|  |  |  |  |  |")
            lines.append("")

    return "\n".join(lines).strip() + "\n"
