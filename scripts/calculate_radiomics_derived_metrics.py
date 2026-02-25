#!/usr/bin/env python3
"""Compute derived metrics from an existing radiomics CSV."""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd


METRIC_MAP = {
    "original_firstorder_Mean": "mean",
    "original_firstorder_10Percentile": "p10",
    "original_firstorder_90Percentile": "p90",
    "original_firstorder_Minimum": "min",
    "original_firstorder_Maximum": "max",
}

OPTIONAL_FEATURE_MAP = {
    "original_firstorder_Entropy": "entropy",
    "original_firstorder_Uniformity": "uniformity",
    "original_firstorder_Variance": "variance",
    "original_glcm_Imc1": "glcm_imc1",
    "original_glszm_SizeZoneNonUniformity": "glszm_sznu",
    "original_glszm_SizeZoneNonUniformityNormalized": "glszm_sznun",
    "original_glszm_SmallAreaEmphasis": "glszm_sae",
    "original_gldm_DependenceVariance": "gldm_depvar",
    "original_shape_MeshVolume": "mesh_volume",
}

ROI_ALIASES = {
    "la": "LA",
    "left_atrium": "LA",
    "left_atrium_highres": "LA",
    "laa": "LAA",
    "laa_nudf": "LAA",
    "left_atrial_appendage": "LAA",
    "ao": "Ao",
    "aorta": "Ao",
    "aorta_highres": "Ao",
    "lv": "LV",
    "left_ventricle": "LV",
    "rv": "RV",
    "right_ventricle": "RV",
    "ra": "RA",
    "right_atrium": "RA",
    "pa": "PA",
    "pulmonary_artery": "PA",
    "svc": "SVC",
    "superior_vena_cava": "SVC",
    "ivc": "IVC",
    "inferior_vena_cava": "IVC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute derived hemodynamic/mixing metrics from radiomics CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input radiomics CSV path")
    parser.add_argument(
        "--output-csv",
        default="radiomics_derived_metrics.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {"patient_id", "ROI", *METRIC_MAP.keys()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    patient_candidates = ["patient_id", "case_id", "subject_id"]
    roi_candidates = ["ROI", "roi", "region"]

    patient_col = next((c for c in patient_candidates if c in df.columns), None)
    roi_col = next((c for c in roi_candidates if c in df.columns), None)
    if patient_col is None or roi_col is None:
        raise ValueError(
            "Could not find patient/ROI columns. "
            "Expected one of patient_id/case_id/subject_id and ROI/region."
        )

    out = df.copy()
    out = out.rename(columns={patient_col: "patient_id", roi_col: "ROI"})

    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().eq("success")].copy()

    roi_raw = out["ROI"].astype(str).str.strip()
    roi_norm = roi_raw.str.lower().map(ROI_ALIASES)
    out["ROI"] = roi_norm.fillna(roi_raw)

    return out


def reshape_to_patient_wide(df: pd.DataFrame) -> pd.DataFrame:
    feature_map = build_feature_map(df)

    wide_parts: list[pd.DataFrame] = []
    for source_col, short_name in feature_map.items():
        part = df.pivot_table(
            index="patient_id",
            columns="ROI",
            values=source_col,
            aggfunc="first",
        )
        part = part.rename(columns=lambda roi: f"{roi}_{short_name}")
        wide_parts.append(part)

    wide_df = pd.concat(wide_parts, axis=1).reset_index()
    value_cols = [c for c in wide_df.columns if c != "patient_id"]
    wide_df[value_cols] = wide_df[value_cols].apply(pd.to_numeric, errors="coerce")
    return wide_df


def _compact_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _find_feature_column(
    columns: list[str],
    include_tokens: list[str],
    exclude_tokens: list[str] | None = None,
    prefer_tokens: list[str] | None = None,
) -> str | None:
    exclude_tokens = exclude_tokens or []
    prefer_tokens = prefer_tokens or []

    candidates: list[tuple[int, str]] = []
    for col in columns:
        compact = _compact_name(col)
        if not all(tok in compact for tok in include_tokens):
            continue
        if any(tok in compact for tok in exclude_tokens):
            continue
        score = sum(tok in compact for tok in prefer_tokens)
        candidates.append((score, col))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def build_feature_map(df: pd.DataFrame) -> dict[str, str]:
    feature_map = dict(METRIC_MAP)

    for col, short_name in OPTIONAL_FEATURE_MAP.items():
        if col in df.columns:
            feature_map[col] = short_name

    all_cols = list(df.columns)
    transformed_candidates = {
        "paper_wavelet_sznu": _find_feature_column(
            all_cols,
            include_tokens=["wavelet", "glszm", "sizezonenonuniformity"],
            exclude_tokens=["normalized"],
        ),
        "paper_square_depvar": _find_feature_column(
            all_cols,
            include_tokens=["square", "gldm", "dependencevariance"],
        ),
        "paper_logsigma15_sae": _find_feature_column(
            all_cols,
            include_tokens=["logsigma", "glszm", "smallareaemphasis"],
            prefer_tokens=["15"],
        ),
        "paper_wavelet_p10": _find_feature_column(
            all_cols,
            include_tokens=["wavelet", "firstorder", "10percentile"],
        ),
        "paper_logsigma45_sznun": _find_feature_column(
            all_cols,
            include_tokens=["logsigma", "glszm", "sizezonenonuniformitynormalized"],
            prefer_tokens=["45"],
        ),
    }

    for short_name, col in transformed_candidates.items():
        if col is not None:
            feature_map[col] = short_name

    return feature_map


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype="float64")
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype="float64")
    out = np.full(num.shape, np.nan, dtype="float64")
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[valid] = num[valid] / den[valid]
    return pd.Series(out, index=numerator.index)


def safe_zscore(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    out = np.full(arr.shape, np.nan, dtype="float64")
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return pd.Series(out, index=values.index)

    mean = arr[valid].mean()
    std = arr[valid].std(ddof=0)
    if std == 0:
        out[valid] = 0.0
    else:
        out[valid] = (arr[valid] - mean) / std
    return pd.Series(out, index=values.index)


def row_nanmean(series_list: list[pd.Series], index: pd.Index) -> pd.Series:
    if not series_list:
        return pd.Series(np.nan, index=index, dtype="float64")

    matrix = np.column_stack([pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64") for s in series_list])
    valid = np.isfinite(matrix)
    count = valid.sum(axis=1)
    total = np.where(valid, matrix, 0.0).sum(axis=1)
    out = np.full(matrix.shape[0], np.nan, dtype="float64")
    nonzero = count > 0
    out[nonzero] = total[nonzero] / count[nonzero]
    return pd.Series(out, index=index)


def compute_derived_metrics(wide_df: pd.DataFrame) -> pd.DataFrame:
    ao_mean = get_series(wide_df, "Ao_mean")
    svc_mean = get_series(wide_df, "SVC_mean")
    pa_mean = get_series(wide_df, "PA_mean")
    lv_mean = get_series(wide_df, "LV_mean")
    rv_mean = get_series(wide_df, "RV_mean")
    ivc_mean = get_series(wide_df, "IVC_mean")
    la_mean = get_series(wide_df, "LA_mean")
    laa_mean = get_series(wide_df, "LAA_mean")
    la_p10 = get_series(wide_df, "LA_p10")
    laa_p10 = get_series(wide_df, "LAA_p10")
    laa_p90 = get_series(wide_df, "LAA_p90")
    ra_mean = get_series(wide_df, "RA_mean")
    laa_entropy = get_series(wide_df, "LAA_entropy")
    laa_uniformity = get_series(wide_df, "LAA_uniformity")
    laa_variance = get_series(wide_df, "LAA_variance")
    laa_imc1 = get_series(wide_df, "LAA_glcm_imc1")
    laa_sznu = get_series(wide_df, "LAA_glszm_sznu")
    laa_sznun = get_series(wide_df, "LAA_glszm_sznun")
    laa_sae = get_series(wide_df, "LAA_glszm_sae")
    laa_depvar = get_series(wide_df, "LAA_gldm_depvar")
    laa_mesh_volume = get_series(wide_df, "LAA_mesh_volume")

    la_wavelet_sznu = get_series(wide_df, "LAA_paper_wavelet_sznu")
    la_square_depvar = get_series(wide_df, "LAA_paper_square_depvar")
    la_logsigma15_sae = get_series(wide_df, "LAA_paper_logsigma15_sae")
    la_wavelet_p10 = get_series(wide_df, "LAA_paper_wavelet_p10")
    la_logsigma45_sznun = get_series(wide_df, "LAA_paper_logsigma45_sznun")

    out = pd.DataFrame({"patient_id": wide_df["patient_id"]})

    out["Ao_SVC_ratio"] = safe_divide(ao_mean, svc_mean)
    out["PA_Ao_ratio"] = safe_divide(pa_mean, ao_mean)
    out["LV_RV_ratio"] = safe_divide(lv_mean, rv_mean)
    out["Ao_IVC_ratio"] = safe_divide(ao_mean, ivc_mean)

    out["LA_LAA_delta"] = la_mean - laa_mean
    out["LA_LAA_ratio"] = safe_divide(laa_mean, la_mean)
    out["LA_LAA_p10_delta"] = la_p10 - laa_p10

    out["Ao_LA_delta"] = ao_mean - la_mean
    out["LV_LA_delta"] = lv_mean - la_mean
    out["RA_RV_delta"] = ra_mean - rv_mean

    out["Normalized_LAA_defect"] = safe_divide(la_mean - laa_mean, ao_mean)

    # Paper-informed context metrics from Ebrahimian et al. (PMCID: PMC7863854).
    out["LAA_to_Ao_HU_ratio"] = safe_divide(laa_mean, ao_mean)
    out["LAA_to_LA_HU_ratio"] = safe_divide(laa_mean, la_mean)
    out["LAA_minus_Ao_HU_delta"] = laa_mean - ao_mean
    out["LAA_minus_LA_HU_delta"] = laa_mean - la_mean
    out["LAA_p10_to_Ao_HU_ratio"] = safe_divide(laa_p10, ao_mean)
    out["LAA_p90_p10_spread"] = laa_p90 - laa_p10
    out["LAA_entropy"] = laa_entropy
    out["LAA_uniformity"] = laa_uniformity
    out["LAA_variance"] = laa_variance
    out["LAA_glcm_Imc1"] = laa_imc1
    out["LAA_glszm_SizeZoneNonUniformity"] = laa_sznu
    out["LAA_glszm_SizeZoneNonUniformityNormalized"] = laa_sznun
    out["LAA_glszm_SmallAreaEmphasis"] = laa_sae
    out["LAA_gldm_DependenceVariance"] = laa_depvar
    out["LAA_volume_ml"] = laa_mesh_volume / 1000.0

    mix_component_sznu = la_wavelet_sznu.combine_first(laa_sznu)
    mix_component_depvar = la_square_depvar.combine_first(laa_depvar)
    mix_component_sae = la_logsigma15_sae.combine_first(laa_sae)
    no_thrombus_component_p10 = la_wavelet_p10.combine_first(laa_p10)
    no_thrombus_component_sznun = la_logsigma45_sznun.combine_first(laa_sznun)

    out["Paper2021_mix_vs_thrombus_proxy_zmean"] = row_nanmean(
        [
            safe_zscore(mix_component_sznu),
            safe_zscore(mix_component_depvar),
            safe_zscore(mix_component_sae),
        ],
        index=wide_df.index,
    )
    out["Paper2021_thrombus_vs_no_thrombus_proxy_zmean"] = row_nanmean(
        [
            safe_zscore(no_thrombus_component_p10),
            safe_zscore(no_thrombus_component_sznun),
            safe_zscore(laa_imc1),
        ],
        index=wide_df.index,
    )
    out["Paper2021_transformed_feature_count"] = (
        la_wavelet_sznu.notna().astype(int)
        + la_square_depvar.notna().astype(int)
        + la_logsigma15_sae.notna().astype(int)
        + la_wavelet_p10.notna().astype(int)
        + la_logsigma45_sznun.notna().astype(int)
    )

    out["LAA_HU_pattern_paper"] = np.select(
        condlist=[
            laa_mean >= 250.0,
            laa_mean <= 150.0,
        ],
        choicelist=[
            "high_hu_like_normal_opacification",
            "low_hu_like_filling_defect",
        ],
        default="intermediate_hu",
    )

    ao_svc = out["Ao_SVC_ratio"]
    pa_ao = out["PA_Ao_ratio"]
    lv_rv = out["LV_RV_ratio"]

    out["phase"] = np.select(
        condlist=[
            pa_ao > 1.5,
            (ao_svc > 1.5) & (lv_rv > 1.0),
            ao_svc.between(0.8, 1.2, inclusive="both"),
        ],
        choicelist=[
            "very_early_right_dominant",
            "arterial",
            "mildly_delayed",
        ],
        default="unclassified",
    )

    return out


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = normalize_input_columns(df)
    validate_input_columns(df)
    wide_df = reshape_to_patient_wide(df)
    derived_df = compute_derived_metrics(wide_df)
    derived_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(derived_df)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
