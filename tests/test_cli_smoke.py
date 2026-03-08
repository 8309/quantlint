import json
from datetime import datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from quantlint.cli import app


def _write_demo_data(tmp_path: Path) -> tuple[Path, Path]:
    features = pl.DataFrame(
        {
            "asset": ["A", "A"],
            "ts": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "f": [1.0, 2.0],
        }
    )
    labels = pl.DataFrame(
        {
            "asset": ["A", "A"],
            "ts": [datetime(2024, 1, 1, 10, 0, 20), datetime(2024, 1, 1, 10, 1, 20)],
            "y": [0.1, 0.2],
        }
    )
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    features.write_csv(features_path)
    labels.write_csv(labels_path)
    return features_path, labels_path


def test_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["scan", "--help"], color=False)
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "scan" in result.output
    assert "--features" in result.output
    assert "--labels" in result.output
    assert "--format" in result.output
    assert "--stdout" in result.output


def test_cli_fail_on_returns_exit_2_and_writes_outputs(tmp_path: Path) -> None:
    features_path, labels_path = _write_demo_data(tmp_path)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--label-time-mode",
            "label_ts",
            "--horizon",
            "1m",
            "--fail-on",
            "HIGH",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 2
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.md"
    assert results_path.exists()
    assert report_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    names = {issue["name"] for issue in payload["issues"]}
    assert "alignment_lookahead_label_too_early" in names

    report_text = report_path.read_text(encoding="utf-8")
    assert "Evidence (showing up to 20 rows):" in report_text
    assert "| row_index | asset | ts | columns | values |" in report_text


def test_cli_format_json_only(tmp_path: Path) -> None:
    features_path, labels_path = _write_demo_data(tmp_path)
    out_dir = tmp_path / "out_json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--label-time-mode",
            "label_ts",
            "--horizon",
            "1m",
            "--format",
            "json",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "results.json").exists()
    assert not (out_dir / "report.md").exists()


def test_cli_stdout_prints_json(tmp_path: Path) -> None:
    features_path, labels_path = _write_demo_data(tmp_path)
    out_dir = tmp_path / "out_stdout"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--label-time-mode",
            "label_ts",
            "--horizon",
            "1m",
            "--format",
            "json",
            "--stdout",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "summary" in payload
    assert "issues" in payload
    assert (out_dir / "results.json").exists()


def test_cli_missing_input_returns_exit_1_without_traceback(tmp_path: Path) -> None:
    out_dir = tmp_path / "out_missing"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            "--features",
            str(tmp_path / "missing_features.csv"),
            "--labels",
            str(tmp_path / "missing_labels.csv"),
            "--label-time-mode",
            "label_ts",
            "--horizon",
            "1m",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 1
    assert "No such file or directory" in result.output
    assert "Traceback" not in result.output


def test_cli_format_md_only_reports_only_written_files(tmp_path: Path) -> None:
    features_path, labels_path = _write_demo_data(tmp_path)
    out_dir = tmp_path / "out_md"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--label-time-mode",
            "label_ts",
            "--horizon",
            "1m",
            "--format",
            "md",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert not (out_dir / "results.json").exists()
    assert (out_dir / "report.md").exists()
    assert "report=" in result.output
    assert "results=" not in result.output
