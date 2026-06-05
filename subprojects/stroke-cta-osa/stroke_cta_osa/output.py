"""Output writers — CSV (features + QC), JSON metadata, JSONL processing log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from . import PIPELINE_NAME, __version__
from .logging_utils import get_logger
from .metric_registry import all_metrics, empty_row, feature_names
from .types import CaseResult

log = get_logger("output")


# Stable column ordering helpers: primary keys + qc + airway + fat + extras.
_PRIMARY_KEYS = (
    "pipeline", "pipeline_version", "config_hash", "processing_timestamp",
    "patient_id", "study_id", "scan_id", "input_path_hash", "input_kind",
    "airway_source", "airway_provider_notes",
)
_QC_KEYS = (
    "qc_pass", "qc_warning_count", "qc_failure_reasons", "qc_coverage_score",
    "qc_dental_artifact_score", "qc_has_upper_airway",
    "qc_has_cervical_soft_tissue", "qc_has_hyoid_region",
    "qc_has_epiglottis_region", "qc_truncation_flag",
    "qc_spacing_x_mm", "qc_spacing_y_mm", "qc_spacing_z_mm",
    "qc_contrast_enhanced", "qc_z_extent_mm",
)


def write_outputs(
    results: Iterable[CaseResult],
    out_dir: Path,
    long_format: bool = False,
    feature_metadata_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Write features.csv, qc.csv and feature_metadata.json under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build rows starting from the registry's empty default row — this
    # guarantees every column the registry knows about is present, with
    # registry-defined missing-value defaults, even when a module didn't
    # populate it.
    base = empty_row()
    rows: list[dict] = []
    for r in results:
        row = dict(base)
        row.update(r.to_feature_row())
        rows.append(row)
    if not rows:
        log.warning("write_outputs called with no results.")
    qc_rows = [r.to_qc_row() for r in results]

    df = pd.DataFrame(rows)
    df = _reorder_columns(df)
    feat_path = out_dir / "features.csv"
    df.to_csv(feat_path, index=False)

    qc_df = pd.DataFrame(qc_rows)
    qc_df = _reorder_columns(qc_df)
    qc_path = out_dir / "qc.csv"
    qc_df.to_csv(qc_path, index=False)

    paths = {"features": feat_path, "qc": qc_path}

    if long_format and not df.empty:
        id_cols = [c for c in _PRIMARY_KEYS if c in df.columns]
        long = df.melt(id_vars=id_cols, var_name="feature", value_name="value")
        long_path = out_dir / "features_long.csv"
        long.to_csv(long_path, index=False)
        paths["features_long"] = long_path

    if feature_metadata_path is None:
        feature_metadata_path = out_dir / "feature_metadata.json"
    feature_metadata_path.write_text(json.dumps(
        {"pipeline": PIPELINE_NAME, "pipeline_version": __version__,
         "columns": list(df.columns), "n_rows": len(df)},
        indent=2,
    ))
    paths["feature_metadata"] = feature_metadata_path

    return paths


def append_processing_log(
    log_path: Path, result: CaseResult, extra: Optional[dict] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "patient_id": result.identifiers.get("patient_id"),
        "scan_id": result.identifiers.get("scan_id"),
        "qc_pass": result.qc.get("qc_pass"),
        "qc_failure_reasons": result.qc.get("qc_failure_reasons"),
        "warnings": result.warnings,
        "errors": result.errors,
    }
    if extra:
        entry.update(extra)
    with log_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical column order: identifiers + qc explicit groups + registry order.

    For columns NOT in the registry (e.g. dental ``*_from_dental`` aliases,
    radiomics features whose exact names depend on the PyRadiomics version),
    we append them alphabetically after the registry-known columns.
    """
    if df.empty:
        return df
    cols = list(df.columns)
    ordered: list[str] = []
    for k in _PRIMARY_KEYS:
        if k in cols and k not in ordered:
            ordered.append(k)
    for k in _QC_KEYS:
        if k in cols and k not in ordered:
            ordered.append(k)
    for k in feature_names():  # registry-canonical order for everything else
        if k in cols and k not in ordered:
            ordered.append(k)
    for k in sorted(c for c in cols if c not in ordered):
        ordered.append(k)
    return df[ordered]
