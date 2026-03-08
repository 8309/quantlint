"""Shared helpers for check modules."""

from __future__ import annotations

import polars as pl

from quantlint.schema import EvidenceItem


def build_evidence(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    columns: list[str],
    row_index_col: str | None = None,
) -> list[EvidenceItem]:
    """Build a list of EvidenceItem from a Polars DataFrame.

    Unified evidence row builder used by all check modules.
    """
    evidence: list[EvidenceItem] = []
    for row in df.iter_rows(named=True):
        values = [row.get(c) for c in columns]
        row_index = row.get(row_index_col) if row_index_col else None
        evidence.append(
            EvidenceItem(
                asset=str(row[id_col]),
                ts=str(row[time_col]),
                columns=columns,
                values=values,
                row_index=int(row_index) if row_index is not None else None,
            )
        )
    return evidence
