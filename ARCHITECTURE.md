# quantlint Architecture

## Data flow (ASCII)

```text
features/labels (csv|parquet)
        |
        v
quantlint/io.py
  - read_table
  - normalize_features / normalize_labels
        |
        v
quantlint/engine.py::run_scan
  - execute checks in sequence
  - aggregate issues + summary
  - compute fail-on exit code
        |
        +--> quantlint/checks/*
        |      - monotonic
        |      - alignment
        |      - imputation
        |      - splits
        |      - rolling hints
        |      - calendar hints
        |
        v
ScanResult (schema)
        |
        +--> JSON output (results.json) via CLI
        +--> Markdown output (report.md) via render_md
```

## Core schema (key fields)

### Issue
- `name: str`
- `severity: CRITICAL|HIGH|MEDIUM|LOW`
- `description: str`
- `suggestion: str`
- `evidence: list[EvidenceItem]`
- `metrics: dict[str, Any] | None` (optional additive)

### EvidenceItem
- `asset: str`
- `ts: str`
- `columns: list[str>`
- `values: list[Any]`
- `row_index: int | None` (optional additive)

### ScanResult
- `summary` with issue counts, row counts, config snapshot
- `issues` list
