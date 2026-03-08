import json
import subprocess
import sys
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from quantlint.cli import app


def _generate_demo(tmp_path: Path) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "make_demo_data.py"

    demo_dir = tmp_path / "demo"
    subprocess.run(
        [sys.executable, str(script_path), "--out-dir", str(demo_dir), "--format", "csv"],
        check=True,
    )

    features_path = demo_dir / "features.csv"
    labels_path = demo_dir / "labels.csv"
    assert features_path.exists()
    assert labels_path.exists()
    return features_path, labels_path


def test_make_demo_data_script_and_default_scan(tmp_path: Path) -> None:
    features_path, labels_path = _generate_demo(tmp_path)
    features = pl.read_csv(features_path, try_parse_dates=True)
    labels = pl.read_csv(labels_path, try_parse_dates=True)

    assert features.height == 10
    assert labels.height == 10
    assert {"asset", "ts", "f", "px", "px_bfill", "vol", "vol_ffill", "rolling_px"} <= set(
        features.columns
    )
    assert {"asset", "ts", "y"} <= set(labels.columns)

    out_dir = tmp_path / "scan_out"
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
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.md"
    assert results_path.exists()
    assert report_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    names = {issue["name"] for issue in payload["issues"]}
    assert "alignment_lookahead_label_too_early" in names
    assert "duplicate_asset_timestamp" in names
    assert "imputation_bfill_future_fill" in names
    assert "imputation_ffill_hint" in names
    assert "calendar_weekend_ratio_hint" in names
    assert "calendar_offhours_ratio_hint" in names


def test_make_demo_data_optional_checks(tmp_path: Path) -> None:
    features_path, labels_path = _generate_demo(tmp_path)

    out_dir = tmp_path / "scan_out_optional"
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
            "--rolling-cols",
            "rolling_px",
            "--train-end",
            "2024-01-05T16:00:00",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0

    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    names = {issue["name"] for issue in payload["issues"]}
    assert "rolling_not_shifted_hint" in names
    assert "split_train_boundary_label_bleed" in names
