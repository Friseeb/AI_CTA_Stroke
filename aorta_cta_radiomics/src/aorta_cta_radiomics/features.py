"""Feature table utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "case_id",
    "region",
    "feature_group",
    "feature_name",
    "feature_value",
    "units",
    "threshold_if_applicable",
    "mask_name",
    "software_version",
]


def feature_row(
    case_id: str,
    region: str,
    feature_group: str,
    feature_name: str,
    feature_value: object,
    units: str = "",
    threshold_if_applicable: object = "",
    mask_name: str = "",
    software_version: str = "0.1.0",
) -> dict[str, object]:
    """Create a normalized long-format feature row."""
    return {
        "case_id": case_id,
        "region": region,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "feature_value": feature_value,
        "units": units,
        "threshold_if_applicable": threshold_if_applicable,
        "mask_name": mask_name,
        "software_version": software_version,
    }


def ensure_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the standard feature columns in order."""
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    out = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out[FEATURE_COLUMNS]


def long_to_wide_features(long_features: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized long features to one-row-per-case wide format."""
    if long_features.empty:
        return pd.DataFrame()
    frame = ensure_feature_columns(long_features)
    feature_key = (
        frame["region"].astype(str)
        + "__"
        + frame["feature_group"].astype(str)
        + "__"
        + frame["feature_name"].astype(str)
        + frame["threshold_if_applicable"].map(lambda v: "" if pd.isna(v) or v == "" else f"__thr_{v}")
    )
    wide_source = frame.assign(feature_key=feature_key)
    wide = wide_source.pivot_table(
        index="case_id",
        columns="feature_key",
        values="feature_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
