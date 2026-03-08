from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl


def make_demo_data(out_dir: Path, file_format: str = "csv") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pl.DataFrame(
        {
            "asset": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "ts": [
                datetime(2024, 1, 5, 10, 0),
                datetime(2024, 1, 5, 10, 1),
                datetime(2024, 1, 5, 10, 1),
                datetime(2024, 1, 5, 10, 2),
                datetime(2024, 1, 5, 15, 59),
                datetime(2024, 1, 5, 16, 0),
                datetime(2024, 1, 5, 16, 1),
                datetime(2024, 1, 6, 20, 0),
                datetime(2024, 1, 6, 20, 1),
                datetime(2024, 1, 6, 20, 2),
            ],
            "f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "px": [None, 100.0, 101.0, 102.0, None, 200.0, 201.0, 300.0, 301.0, 302.0],
            "px_bfill": [100.0, 100.0, 101.0, 102.0, 200.0, 200.0, 201.0, 300.0, 301.0, 302.0],
            "vol": [1.0, None, 1.2, 1.3, 2.0, None, 2.1, 3.0, None, 3.1],
            "vol_ffill": [1.0, 1.0, 1.2, 1.3, 2.0, 2.0, 2.1, 3.0, 3.0, 3.1],
            "rolling_px": [None, 100.0, 101.0, 102.0, None, 200.0, 201.0, 300.0, 301.0, 302.0],
        }
    )

    labels = pl.DataFrame(
        {
            "asset": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "ts": [
                datetime(2024, 1, 5, 10, 0, 20),
                datetime(2024, 1, 5, 10, 1, 20),
                datetime(2024, 1, 5, 10, 1, 40),
                datetime(2024, 1, 5, 10, 2, 20),
                datetime(2024, 1, 5, 15, 59, 20),
                datetime(2024, 1, 5, 16, 0, 20),
                datetime(2024, 1, 5, 16, 1, 20),
                datetime(2024, 1, 6, 20, 0, 20),
                datetime(2024, 1, 6, 20, 1, 20),
                datetime(2024, 1, 6, 20, 2, 20),
            ],
            "y": [0.10, 0.20, 0.25, 0.30, -0.10, -0.15, -0.20, 0.05, 0.04, 0.03],
        }
    )

    ext = "parquet" if file_format == "parquet" else "csv"
    features_path = out_dir / f"features.{ext}"
    labels_path = out_dir / f"labels.{ext}"

    if ext == "parquet":
        features.write_parquet(features_path)
        labels.write_parquet(labels_path)
    else:
        features.write_csv(features_path)
        labels.write_csv(labels_path)

    return features_path, labels_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible demo data for quantlint.")
    parser.add_argument(
        "--out-dir", default="./demo_data", help="Output directory for features/labels files"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    features_path, labels_path = make_demo_data(out_dir=out_dir, file_format=args.format)
    print(f"Generated demo features: {features_path}")
    print(f"Generated demo labels: {labels_path}")


if __name__ == "__main__":
    main()
