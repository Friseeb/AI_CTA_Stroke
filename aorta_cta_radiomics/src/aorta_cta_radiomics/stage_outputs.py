"""Shared helpers for separated per-case stage outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import ensure_feature_columns, long_to_wide_features, write_csv


LONG_FEATURE_FILES = [
    "case_level_features.csv",
    "calcification_features.csv",
    "calcium_omics_features.csv",
    "fat_omics_features.csv",
    "lumen_protrusion_summary_features.csv",
    "wall_from_fat_features.csv",
    "wall_thickness_summary.csv",
    "wall_thickness_gt_4mm_TEE_analogue_summary.csv",
    "wall_thickness_summary_with_thresholds.csv",
    "radiomics_features.csv",
]


def rebuild_modeling_wide(features_dir: str | Path) -> Path:
    """Rebuild one case's wide modeling table from available long feature CSVs."""
    features_path = Path(features_dir)
    frames: list[pd.DataFrame] = []
    for name in LONG_FEATURE_FILES:
        path = features_path / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if {"case_id", "region", "feature_group", "feature_name", "feature_value"}.issubset(frame.columns):
            frames.append(ensure_feature_columns(frame))
    wide = long_to_wide_features(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    return write_csv(wide, features_path / "modeling_wide_features.csv")
