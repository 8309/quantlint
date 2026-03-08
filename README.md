# quantlint
[![CI](https://github.com/8309/quantlint/actions/workflows/ci.yml/badge.svg)](https://github.com/8309/quantlint/actions/workflows/ci.yml)

`quantlint` is a lightweight scanner for quant backtest data quality and leakage/alignment risks.
It checks long-table `features` + `labels` inputs and outputs machine-readable and human-readable reports.

## Quickstart (60 seconds)

```bash
# 1) install
python -m pip install -e .

# 2) generate reproducible demo data
python scripts/make_demo_data.py --out-dir ./demo_data --format csv

# 3) run scan
quantlint scan \
  --features ./demo_data/features.csv \
  --labels ./demo_data/labels.csv \
  --label-time-mode label_ts \
  --horizon 1m \
  --out-dir ./quantlint_out_demo

# 4) inspect outputs
ls -la ./quantlint_out_demo
sed -n '1,30p' ./quantlint_out_demo/report.md
```

CI note: the e2e workflow job uploads artifact `quantlint-ci-out` (contains `results.json` and `report.md`).

Expected top section in `report.md` for the bundled demo data (example):

```md
# Quantlint Report

## Summary

| metric | value |
|---|---:|
| total_issues | 7 |
| features_rows | 10 |
| labels_rows | 10 |
| scanned_assets | 3 |
| critical | 1 |
| high | 3 |
| medium | 1 |
| low | 2 |

## CRITICAL (1)

### alignment_lookahead_label_too_early
```

Optional: the same demo data also supports extra checks when you pass
`--rolling-cols rolling_px --train-end 2024-01-05T16:00:00`.

## What it checks

- A. Monotonic & duplicate timestamps per asset
- B. Alignment look-ahead risks (`label_ts` and `feature_ts` modes)
- C. Rolling-not-shifted hints (LOW, when `--rolling-cols` is provided)
- D. Imputation risks (`bfill` as HIGH/CRITICAL, `ffill` as LOW/MEDIUM hints)
- E. Split-boundary bleed checks (`--train-end`, `--val-end`)
- F. Calendar/time hints (weekend/off-hours ratios)

## Install

Requirements: Python 3.10+

```bash
python -m pip install -e .
```

In this workspace, commands are usually run via micromamba:

```bash
micromamba run -n quantlint-py311 python -m pip install -e .
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

## Temporary Outputs

- Stable demo inputs: `./demo_data`
- Stable demo outputs: `./quantlint_out_demo`
- CI-only artifacts: `./_ci_demo`, `./_ci_out`
- Ad hoc local experiments: `./_tmp_demo_<topic>`, `./_tmp_out_<topic>`
- Prefer new temp directories over delete-and-recreate flows; reuse fixed demo paths only when intentional.

## Input schema

- features must include: `asset` (str), `ts` (datetime), and feature columns
- labels must include: `asset` (str), `ts` (datetime), `y` (float)

Both `.csv` and `.parquet` are supported.

## Minimal command

```bash
quantlint scan \
  --features ./features.csv \
  --labels ./labels.csv \
  --label-time-mode label_ts \
  --horizon 1D \
  --out-dir ./quantlint_out
```

## Output controls

- `--format json|md|both` (default: `both`) controls output files on disk
- `--stdout` prints the `results.json` payload to stdout for CI/pipeline collection

## `results.json` shape

```json
{
  "summary": {
    "total_issues": 4,
    "by_severity": {"CRITICAL": 1, "HIGH": 3, "MEDIUM": 0, "LOW": 0},
    "scanned_assets": 2,
    "features_rows": 5,
    "labels_rows": 5,
    "config": {"label_time_mode": "label_ts", "horizon": "1m"}
  },
  "issues": [
    {
      "name": "alignment_lookahead_label_too_early",
      "severity": "CRITICAL",
      "description": "...",
      "suggestion": "...",
      "metrics": {"violations_count": 2},
      "evidence": [
        {
          "asset": "A",
          "ts": "2024-01-02 10:00:00",
          "columns": ["label_ts", "min_label_ts", "__idx"],
          "values": ["2024-01-02 10:00:20", "2024-01-02 10:01:00", 0],
          "row_index": 0
        }
      ]
    }
  ]
}
```

## Exit code behavior

- default: exit code `0`
- with `--fail-on LEVEL`: exit code `2` when any issue severity is `>= LEVEL`
