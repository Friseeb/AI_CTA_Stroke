"""Compare CTA-derived airway features to the dental/CBCT pipeline.

Either pipeline writes its own features.csv. Both may include the columns
listed in `shared_schema.SHARED_FEATURE_NAMES`. This utility joins them on
patient_id / scan_id, emits diffs, and produces a Bland-Altman-ready table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .shared_schema import SHARED_FEATURE_NAMES

log = get_logger("compare_dental")


def compare_with_dental(
    cta_features_csv: Path,
    dental_features_csv: Path,
    out_dir: Path,
    patient_id_column: str = "patient_id",
    scan_id_column: str = "scan_id",
    date_column_cta: Optional[str] = None,
    date_column_dental: Optional[str] = None,
) -> dict:
    cta = pd.read_csv(cta_features_csv)
    dent = pd.read_csv(dental_features_csv)

    join_cols = [c for c in (patient_id_column, scan_id_column)
                 if c in cta.columns and c in dent.columns]
    if not join_cols:
        raise KeyError("No shared join columns between CTA and dental CSVs.")

    # Pull only shared columns (plus join) on each side, rename to *_cta / *_dental.
    shared_in_cta = [c for c in SHARED_FEATURE_NAMES if c in cta.columns]
    shared_in_dent = [c for c in SHARED_FEATURE_NAMES if c in dent.columns]
    cta_side = cta[join_cols + shared_in_cta].rename(
        columns={c: f"{c}_cta" for c in shared_in_cta}
    )
    dent_side = dent[join_cols + shared_in_dent].rename(
        columns={c: f"{c}_dental" for c in shared_in_dent}
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = cta_side.merge(dent_side, on=join_cols, how="outer", indicator=True)
    merged_path = out_dir / "dental_cta_feature_comparison.csv"
    merged.to_csv(merged_path, index=False)

    # Bland-Altman table for each shared feature with both sides populated
    ba_rows: list[dict] = []
    common = sorted(set(shared_in_cta) & set(shared_in_dent))
    for feat in common:
        cta_col, dent_col = f"{feat}_cta", f"{feat}_dental"
        if cta_col not in merged.columns or dent_col not in merged.columns:
            continue
        sub = merged[[*join_cols, cta_col, dent_col]].dropna()
        if sub.empty:
            continue
        sub = sub.copy()
        sub["mean"] = (sub[cta_col] + sub[dent_col]) / 2.0
        sub["diff"] = sub[cta_col] - sub[dent_col]
        sub["feature"] = feat
        ba_rows.append(sub[[*join_cols, "feature", cta_col, dent_col, "mean", "diff"]]
                       .rename(columns={cta_col: "cta_value", dent_col: "dental_value"}))
    if ba_rows:
        ba = pd.concat(ba_rows, ignore_index=True)
        ba_path = out_dir / "bland_altman_table.csv"
        ba.to_csv(ba_path, index=False)
    else:
        ba_path = None

    # Quick correlation summary
    corr_rows = []
    for feat in common:
        cta_col, dent_col = f"{feat}_cta", f"{feat}_dental"
        sub = merged[[cta_col, dent_col]].dropna()
        if len(sub) < 3:
            continue
        r = float(np.corrcoef(sub[cta_col], sub[dent_col])[0, 1])
        bias = float((sub[cta_col] - sub[dent_col]).mean())
        loa_low = float((sub[cta_col] - sub[dent_col]).mean()
                        - 1.96 * (sub[cta_col] - sub[dent_col]).std())
        loa_high = float((sub[cta_col] - sub[dent_col]).mean()
                         + 1.96 * (sub[cta_col] - sub[dent_col]).std())
        corr_rows.append({
            "feature": feat, "n": int(len(sub)), "pearson_r": round(r, 3),
            "bias_cta_minus_dental": round(bias, 3),
            "loa_low": round(loa_low, 3), "loa_high": round(loa_high, 3),
        })
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_path = out_dir / "correlation_summary.csv"
        corr_df.to_csv(corr_path, index=False)
    else:
        corr_path = None

    miss_path = out_dir / "missingness_summary.csv"
    miss = pd.DataFrame({
        "column": merged.columns,
        "n_missing": merged.isna().sum().values,
        "pct_missing": (merged.isna().sum() / max(len(merged), 1) * 100).round(1).values,
    })
    miss.to_csv(miss_path, index=False)

    summary = {
        "n_cta_rows": int(len(cta)),
        "n_dental_rows": int(len(dent)),
        "n_merged_rows": int(len(merged)),
        "matched_features": common,
        "merged_csv": str(merged_path),
        "bland_altman_csv": str(ba_path) if ba_path else None,
        "correlation_csv": str(corr_path) if corr_path else None,
        "missingness_csv": str(miss_path),
    }
    return summary
