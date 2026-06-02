"""Clinical CSV merge.

Kept deliberately minimal: this module joins a features.csv with an external
clinical CSV on patient_id (or scan_id) and reports unmatched rows. It does
NO inferential statistics — outcome modelling lives in a separate analysis
script that the user runs against the merged file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .logging_utils import get_logger

log = get_logger("clinical")


def merge_clinical(
    features_csv: Path,
    clinical_csv: Path,
    out_path: Path,
    patient_id_column: str = "patient_id",
    scan_id_column: str = "scan_id",
) -> dict:
    feat = pd.read_csv(features_csv)
    clin = pd.read_csv(clinical_csv)

    # Validate join keys
    if patient_id_column not in feat.columns:
        raise KeyError(f"Features CSV missing column '{patient_id_column}'")
    if patient_id_column not in clin.columns:
        raise KeyError(f"Clinical CSV missing column '{patient_id_column}'")

    join_cols = [patient_id_column]
    if scan_id_column in feat.columns and scan_id_column in clin.columns:
        join_cols.append(scan_id_column)

    merged = feat.merge(clin, on=join_cols, how="left", indicator=True)
    matched = int((merged["_merge"] == "both").sum())
    unmatched = merged.loc[merged["_merge"] == "left_only", join_cols].copy()

    # Patients in the clinical file with no imaging row
    extra = clin.merge(feat[join_cols].drop_duplicates(), on=join_cols, how="left",
                       indicator=True)
    extra_only = extra.loc[extra["_merge"] == "left_only", join_cols]

    merged = merged.drop(columns=["_merge"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    summary = {
        "n_feature_rows": int(len(feat)),
        "n_clinical_rows": int(len(clin)),
        "n_merged_rows": int(len(merged)),
        "n_matched": matched,
        "n_features_without_clinical": int(len(unmatched)),
        "n_clinical_without_features": int(len(extra_only)),
        "join_columns": join_cols,
        "merged_path": str(out_path),
    }
    log.info(
        "Clinical merge: %d features + %d clinical → %d rows (%d matched, "
        "%d features w/o clinical, %d clinical w/o features).",
        summary["n_feature_rows"], summary["n_clinical_rows"],
        summary["n_merged_rows"], summary["n_matched"],
        summary["n_features_without_clinical"], summary["n_clinical_without_features"],
    )
    return summary
