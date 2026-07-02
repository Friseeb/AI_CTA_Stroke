#!/usr/bin/env python
"""Estimate odds ratios for aorta burden by directly labeled stroke etiology."""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = AORTA_ROOT / "outputs" / "aorta_batch_run" / "etiology_slao" / "slao_etiology_aorta_modeling.csv"
DEFAULT_OUTDIR = AORTA_ROOT / "outputs" / "aorta_batch_run" / "etiology_slao" / "aorta_burden_or"
SOURCE_LABEL = "source_etiology_label"
ETIOLOGIES = ("ESUS", "KAF", "AFDAS", "New_ECG_AF")
REFERENCE_ETIOLOGY = "ESUS"


@dataclass(frozen=True)
class BurdenFeature:
    slug: str
    label: str
    column: str
    unit: str
    display_scale: float = 1.0
    family: str = "primary"


BURDEN_FEATURES = [
    BurdenFeature(
        slug="dynamic_calcium_volume",
        label="Dynamic calcium volume",
        column="aorta__calcium_omics__aortic_volume_mm3__thr_dynamic_lumen_referenced_seed500HU",
        unit="ml",
        display_scale=0.001,
        family="primary",
    ),
    BurdenFeature(
        slug="dynamic_agatston",
        label="Dynamic Agatston-like burden",
        column="aorta__calcium_omics__aortic_agatston_modified__thr_dynamic_lumen_referenced_seed500HU",
        unit="arb.",
        family="primary",
    ),
    BurdenFeature(
        slug="dynamic_calcium_per_cm",
        label="Dynamic calcium per cm",
        column="aorta__calcium_omics__calcium_per_cm__thr_dynamic_lumen_referenced_seed500HU",
        unit="mm3/cm",
        family="primary",
    ),
    BurdenFeature(
        slug="dynamic_calcium_mass_per_cm",
        label="Dynamic calcium mass proxy per cm",
        column="aorta__calcium_omics__calcium_mass_proxy_per_cm__thr_dynamic_lumen_referenced_seed500HU",
        unit="HU*mm3/cm",
        family="primary",
    ),
    BurdenFeature(
        slug="wall_dynamic_calcium_volume",
        label="Wall dynamic calcium volume",
        column="aorta_wall_dynamic__calcification__calcium_volume__thr_dynamic_lumen_referenced_seed500HU",
        unit="ml",
        display_scale=0.001,
        family="sensitivity",
    ),
    BurdenFeature(
        slug="wall_dynamic_agatston",
        label="Wall dynamic Agatston-like burden",
        column="aorta_wall_dynamic__calcification__agatston_like_not_ecg_gated__thr_dynamic_lumen_referenced_seed500HU",
        unit="arb.",
        family="sensitivity",
    ),
]


TARGETS = {
    "KAF": lambda frame: frame[SOURCE_LABEL].eq("KAF"),
    "AFDAS": lambda frame: frame[SOURCE_LABEL].eq("AFDAS"),
    "ECG_AF": lambda frame: frame[SOURCE_LABEL].eq("New_ECG_AF"),
    "ESUS": lambda frame: frame[SOURCE_LABEL].eq("ESUS"),
    "AFDAS_or_ECG_AF": lambda frame: frame[SOURCE_LABEL].isin(["AFDAS", "New_ECG_AF"]),
}


def main() -> None:
    args = build_parser().parse_args()
    analysis_path = args.analysis.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(analysis_path, dtype=str)
    frame = frame[frame[SOURCE_LABEL].isin(ETIOLOGIES)].copy()
    if frame.empty:
        raise ValueError(f"No rows with direct source etiology labels found in {analysis_path}")
    frame["age_numeric"] = pd.to_numeric(frame.get("age"), errors="coerce")
    frame.loc[~frame["age_numeric"].between(18, 120), "age_numeric"] = np.nan
    frame["sex_male"] = frame.get("sex", "").map(lambda value: 1 if str(value).strip() == "Male" else 0 if str(value).strip() == "Female" else np.nan)

    summary = burden_summary(frame)
    high_burden_or = high_burden_models(frame)
    etiology_outcome_or = etiology_outcome_models(frame)

    summary_path = outdir / "aorta_burden_by_etiology_summary.csv"
    high_or_path = outdir / "aorta_high_burden_or_by_etiology.csv"
    etiology_or_path = outdir / "etiology_or_per_iqr_aorta_burden.csv"
    report_path = outdir / "aorta_burden_or_by_stroke_etiology.html"
    summary.to_csv(summary_path, index=False)
    high_burden_or.to_csv(high_or_path, index=False)
    etiology_outcome_or.to_csv(etiology_or_path, index=False)
    report_path.write_text(build_report(summary, high_burden_or, etiology_outcome_or, analysis_path), encoding="utf-8")

    print(f"Summary: {summary_path}")
    print(f"High burden ORs: {high_or_path}")
    print(f"Etiology outcome ORs: {etiology_or_path}")
    print(f"Report: {report_path}")


def burden_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in BURDEN_FEATURES:
        if feature.column not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature.column], errors="coerce") * feature.display_scale
        for etiology, group in frame.assign(_value=values).groupby(SOURCE_LABEL, dropna=False):
            observed = group["_value"].dropna()
            rows.append(
                {
                    "feature": feature.slug,
                    "feature_label": feature.label,
                    "family": feature.family,
                    "unit": feature.unit,
                    SOURCE_LABEL: etiology,
                    "n": int(observed.size),
                    "median": safe_float(observed.median()),
                    "q25": safe_float(observed.quantile(0.25)),
                    "q75": safe_float(observed.quantile(0.75)),
                    "mean": safe_float(observed.mean()),
                }
            )
    return pd.DataFrame(rows)


def high_burden_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in BURDEN_FEATURES:
        if feature.column not in frame.columns:
            continue
        raw = pd.to_numeric(frame[feature.column], errors="coerce")
        analytic = frame.loc[raw.notna()].copy()
        analytic["_burden_raw"] = raw.loc[raw.notna()]
        for threshold_name, quantile in [("above_median", 0.50), ("top_quartile", 0.75)]:
            threshold_raw = float(analytic["_burden_raw"].quantile(quantile))
            analytic["_high_burden"] = analytic["_burden_raw"].gt(threshold_raw).astype(int)
            for adjusted in [False, True]:
                rows.extend(
                    fit_high_burden_model(
                        analytic,
                        feature=feature,
                        threshold_name=threshold_name,
                        threshold_raw=threshold_raw,
                        adjusted=adjusted,
                    )
                )
    return pd.DataFrame(rows)


def fit_high_burden_model(
    frame: pd.DataFrame,
    *,
    feature: BurdenFeature,
    threshold_name: str,
    threshold_raw: float,
    adjusted: bool,
) -> list[dict[str, object]]:
    data = frame[[SOURCE_LABEL, "_high_burden", "age_numeric", "sex_male"]].copy()
    if adjusted:
        data = data.dropna(subset=["age_numeric", "sex_male"])
    dummies = pd.get_dummies(data[SOURCE_LABEL], dtype=float)
    for etiology in ETIOLOGIES:
        if etiology not in dummies.columns:
            dummies[etiology] = 0.0
    terms = [etiology for etiology in ETIOLOGIES if etiology != REFERENCE_ETIOLOGY]
    x = dummies[terms].copy()
    if adjusted:
        x["age"] = data["age_numeric"].astype(float)
        x["sex_male"] = data["sex_male"].astype(float)
    x = sm.add_constant(x, has_constant="add")
    y = data["_high_burden"].astype(int)
    model_result = fit_logit(y, x)
    rows = []
    for term in terms:
        rows.append(
            result_row(
                model_result,
                term=term,
                feature=feature,
                model="high_burden_by_etiology",
                contrast=f"{term} vs {REFERENCE_ETIOLOGY}",
                outcome=f"aorta_burden_{threshold_name}",
                adjusted=adjusted,
                n=int(len(data)),
                events=int(y.sum()),
                threshold_raw=threshold_raw,
                threshold_display=threshold_raw * feature.display_scale,
                threshold_unit=feature.unit,
            )
        )
    return rows


def etiology_outcome_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in BURDEN_FEATURES:
        if feature.column not in frame.columns:
            continue
        raw = pd.to_numeric(frame[feature.column], errors="coerce")
        transformed = np.log1p(raw)
        iqr = transformed.quantile(0.75) - transformed.quantile(0.25)
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        analytic = frame.loc[transformed.notna()].copy()
        analytic["_burden_iqr"] = transformed.loc[transformed.notna()] / iqr
        for target_name, target_fn in TARGETS.items():
            analytic["_target"] = target_fn(analytic).astype(int)
            for adjusted in [False, True]:
                rows.append(
                    fit_etiology_outcome_model(
                        analytic,
                        feature=feature,
                        target_name=target_name,
                        adjusted=adjusted,
                        log_iqr=float(iqr),
                    )
                )
    return pd.DataFrame(rows)


def fit_etiology_outcome_model(
    frame: pd.DataFrame,
    *,
    feature: BurdenFeature,
    target_name: str,
    adjusted: bool,
    log_iqr: float,
) -> dict[str, object]:
    columns = ["_target", "_burden_iqr", "age_numeric", "sex_male"]
    data = frame[columns].copy()
    if adjusted:
        data = data.dropna(subset=["age_numeric", "sex_male"])
    x = data[["_burden_iqr"]].astype(float).rename(columns={"_burden_iqr": "burden_per_iqr_log1p"})
    if adjusted:
        x["age"] = data["age_numeric"].astype(float)
        x["sex_male"] = data["sex_male"].astype(float)
    x = sm.add_constant(x, has_constant="add")
    y = data["_target"].astype(int)
    model_result = fit_logit(y, x)
    return result_row(
        model_result,
        term="burden_per_iqr_log1p",
        feature=feature,
        model="etiology_outcome_per_iqr_burden",
        contrast=f"{target_name} vs other direct etiologies",
        outcome=target_name,
        adjusted=adjusted,
        n=int(len(data)),
        events=int(y.sum()),
        log1p_iqr=log_iqr,
    )


def fit_logit(y: pd.Series, x: pd.DataFrame):
    if y.nunique(dropna=True) < 2:
        return None
    try:
        return sm.Logit(y, x.astype(float)).fit(disp=False, maxiter=200)
    except Exception:
        return None


def result_row(model_result, *, term: str, feature: BurdenFeature, model: str, contrast: str, outcome: str, adjusted: bool, n: int, events: int, **extra) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "feature": feature.slug,
        "feature_label": feature.label,
        "family": feature.family,
        "contrast": contrast,
        "outcome": outcome,
        "adjusted": bool(adjusted),
        "adjustment": "age + sex" if adjusted else "none",
        "n": n,
        "events": events,
        "or": "",
        "ci_low": "",
        "ci_high": "",
        "p_value": "",
        **extra,
    }
    if model_result is None or term not in model_result.params.index:
        row["fit_status"] = "failed"
        return row
    coefficient = float(model_result.params[term])
    ci_low, ci_high = model_result.conf_int().loc[term].astype(float)
    row.update(
        {
            "or": math.exp(coefficient),
            "ci_low": math.exp(float(ci_low)),
            "ci_high": math.exp(float(ci_high)),
            "p_value": float(model_result.pvalues[term]),
            "fit_status": "ok",
        }
    )
    return row


def build_report(summary: pd.DataFrame, high_burden_or: pd.DataFrame, etiology_outcome_or: pd.DataFrame, analysis_path: Path) -> str:
    primary_summary = summary[summary["family"].eq("primary")].copy()
    primary_high = high_burden_or[
        high_burden_or["family"].eq("primary")
        & high_burden_or["outcome"].eq("aorta_burden_above_median")
        & high_burden_or["adjusted"].eq(True)
    ].copy()
    primary_etiology = etiology_outcome_or[
        etiology_outcome_or["family"].eq("primary") & etiology_outcome_or["adjusted"].eq(True)
    ].copy()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Aorta Burden OR by Stroke Etiology</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #222; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    p {{ max-width: 920px; line-height: 1.45; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, td:nth-child(2), td:nth-child(3), td:nth-child(4) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .note {{ color: #555; }}
    code {{ background: #f5f5f5; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Aorta Burden OR by Stroke Etiology</h1>
  <p class="note">Source: <code>{html.escape(str(analysis_path))}</code>. Direct etiologies: ESUS, KAF, AFDAS, and ECG-AF.</p>
  <p>Primary OR table: outcome is high aorta burden, defined as above the cohort median for that feature. ORs compare each etiology with ESUS as the reference and are adjusted for age and sex. Secondary table reverses the direction: odds of each etiology per IQR increase in log-transformed burden.</p>
  <h2>Primary Adjusted OR: High Aorta Burden by Etiology</h2>
  {format_table(primary_high)}
  <h2>Adjusted OR: Etiology per IQR Increase in Aorta Burden</h2>
  {format_table(primary_etiology)}
  <h2>Burden Distribution by Etiology</h2>
  {format_table(primary_summary)}
</body>
</html>
"""


def format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(format_number)
    return display.to_html(index=False, escape=True)


def format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def safe_float(value: object) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if np.isfinite(number) else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


if __name__ == "__main__":
    main()
