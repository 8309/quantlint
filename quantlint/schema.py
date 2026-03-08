from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FailOnSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


class OutputFormat(str, Enum):
    JSON = "json"
    MD = "md"
    BOTH = "both"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class LabelTimeMode(str, Enum):
    FEATURE_TS = "feature_ts"
    LABEL_TS = "label_ts"


class EvidenceItem(BaseModel):
    asset: str
    ts: str
    columns: list[str] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)
    row_index: int | None = None


class Issue(BaseModel):
    name: str
    severity: Severity
    description: str
    suggestion: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    metrics: dict[str, Any] | None = None


class Summary(BaseModel):
    total_issues: int
    by_severity: dict[str, int]
    scanned_assets: int
    features_rows: int
    labels_rows: int
    config: dict[str, Any] = Field(default_factory=dict)


class ScanResult(BaseModel):
    summary: Summary
    issues: list[Issue] = Field(default_factory=list)


class ScanConfig(BaseModel):
    features: str
    labels: str
    id_col: str = "asset"
    time_col: str = "ts"
    label_col: str = "y"
    label_time_mode: LabelTimeMode
    horizon: str
    rolling_cols: list[str] = Field(default_factory=list)
    train_end: datetime | None = None
    val_end: datetime | None = None
    out_dir: str = "./quantlint_out"
    max_evidence: int = 20
    fail_on: FailOnSeverity | None = None
    output_format: OutputFormat = OutputFormat.BOTH
    stdout: bool = False

    @field_validator("id_col", "time_col", "label_col", "features", "labels", "out_dir")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, v: str) -> str:
        if not re.match(r"^\d+[mMhHdDwW]$", v):
            raise ValueError("horizon must look like 5m, 1h, 1D")
        return v

    @field_validator("max_evidence")
    @classmethod
    def _validate_max_evidence(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_evidence must be > 0")
        return v

    @field_validator("rolling_cols", mode="before")
    @classmethod
    def _normalize_rolling_cols(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        if isinstance(v, list):
            return [str(c).strip() for c in v if str(c).strip()]
        raise ValueError("rolling_cols must be a comma-separated string or list")

    @model_validator(mode="after")
    def _validate_ranges(self) -> "ScanConfig":
        if self.train_end and self.val_end and self.val_end < self.train_end:
            raise ValueError("val_end must be >= train_end")
        return self


def meets_fail_threshold(issue_severity: Severity, fail_on: FailOnSeverity | None) -> bool:
    if fail_on is None:
        return False
    return SEVERITY_RANK[issue_severity] >= SEVERITY_RANK[Severity(fail_on.value)]
