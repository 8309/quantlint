from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from quantlint import __version__
from quantlint.engine import run_scan
from quantlint.io import TableValidationError, UnsupportedFormatError
from quantlint.report.render_md import render_markdown_report
from quantlint.schema import FailOnSeverity, LabelTimeMode, OutputFormat, ScanConfig

app = typer.Typer(help="Quantlint CLI")


@app.command("scan")
def scan(
    features: str = typer.Option(..., help="Path to features table (csv/parquet)"),
    labels: str = typer.Option(..., help="Path to labels table (csv/parquet)"),
    id_col: str = typer.Option("asset", help="Entity id column"),
    time_col: str = typer.Option("ts", help="Timestamp column"),
    label_col: str = typer.Option("y", help="Label column"),
    label_time_mode: LabelTimeMode = typer.Option(..., help="feature_ts or label_ts"),
    horizon: str = typer.Option(..., help="Prediction horizon like 1D/5m/1h"),
    rolling_cols: str | None = typer.Option(None, help="Comma-separated rolling feature columns"),
    train_end: str | None = typer.Option(None, help="Train boundary, ISO timestamp"),
    val_end: str | None = typer.Option(None, help="Validation boundary, ISO timestamp"),
    out_dir: str = typer.Option("./quantlint_out", help="Output directory"),
    max_evidence: int = typer.Option(20, help="Max evidence rows per issue"),
    fail_on: FailOnSeverity | None = typer.Option(None, help="Exit code 2 when issue >= level"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.BOTH, "--format", help="Output file format: json, md, or both"
    ),
    stdout: bool = typer.Option(False, "--stdout", help="Print results.json payload to stdout"),
) -> None:
    """Scan feature/label tables and report leakage or alignment issues."""
    try:
        config = ScanConfig(
            features=features,
            labels=labels,
            id_col=id_col,
            time_col=time_col,
            label_col=label_col,
            label_time_mode=label_time_mode,
            horizon=horizon,
            rolling_cols=rolling_cols,
            train_end=train_end,
            val_end=val_end,
            out_dir=out_dir,
            max_evidence=max_evidence,
            fail_on=fail_on,
            output_format=output_format,
            stdout=stdout,
        )
    except ValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        result, exit_code = run_scan(config)
    except (FileNotFoundError, TableValidationError, UnsupportedFormatError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    json_text = json.dumps(result.model_dump(), indent=2, default=str)

    out_dir_path = Path(config.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    results_path = out_dir_path / "results.json"
    report_path = out_dir_path / "report.md"
    written_outputs: list[str] = []

    if config.output_format in {OutputFormat.BOTH, OutputFormat.JSON}:
        results_path.write_text(json_text, encoding="utf-8")
        written_outputs.append(f"results={results_path}")

    if config.output_format in {OutputFormat.BOTH, OutputFormat.MD}:
        report_path.write_text(
            render_markdown_report(result, config.max_evidence), encoding="utf-8"
        )
        written_outputs.append(f"report={report_path}")

    if config.stdout:
        typer.echo(json_text)
    else:
        typer.echo(
            f"quantlint scan completed. {' '.join(written_outputs)} issues={result.summary.total_issues}"
        )

    raise typer.Exit(code=exit_code)


@app.command("version")
def version() -> None:
    """Show quantlint version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
