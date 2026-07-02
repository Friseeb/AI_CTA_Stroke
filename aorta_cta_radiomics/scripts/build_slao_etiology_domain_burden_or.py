#!/usr/bin/env python
"""Estimate stroke-etiology odds ratios for mixed and wall-only aorta domain scores."""

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
DEFAULT_DOMAIN_SCORES = AORTA_ROOT / "outputs" / "aorta_batch_run" / "pca_clustering" / "domain_state_case_features.csv"
DEFAULT_OUTDIR = AORTA_ROOT / "outputs" / "aorta_batch_run" / "etiology_slao" / "domain_burden_or"

CASE_ID = "case_id"
SOURCE_LABEL = "source_etiology_label"
ETIOLOGIES = ("ESUS", "KAF", "AFDAS", "New_ECG_AF")
REFERENCE_ETIOLOGY = "ESUS"


@dataclass(frozen=True)
class DomainScore:
    slug: str
    label: str
    columns: tuple[str, ...]
    family: str = "primary"
    require_all_columns: bool = True


DOMAIN_SCORES = [
    DomainScore(
        slug="mixed_wall_fat_calcium",
        label="Mixed wall/fat/calcium domain score",
        columns=("wall_score", "peri_fat_score", "calcium_score"),
        family="primary",
        require_all_columns=True,
    ),
    DomainScore(
        slug="wall_only",
        label="Wall-only domain score",
        columns=("wall_score",),
        family="primary",
    ),
    DomainScore(
        slug="wall_fat_mix",
        label="Wall/fat domain score",
        columns=("wall_score", "peri_fat_score"),
        family="sensitivity",
        require_all_columns=True,
    ),
    DomainScore(
        slug="calcium_only",
        label="Calcium-only domain score",
        columns=("calcium_score",),
        family="context",
    ),
    DomainScore(
        slug="peri_fat_only",
        label="Periaortic fat-only domain score",
        columns=("peri_fat_score",),
        family="context",
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
    domain_scores_path = args.domain_scores.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(analysis_path, domain_scores_path)
    add_domain_composites(frame)

    summary = domain_summary(frame)
    high_score_or = high_score_models(frame)
    etiology_outcome_or = etiology_outcome_models(frame)

    summary_path = outdir / "domain_score_by_etiology_summary.csv"
    high_or_path = outdir / "high_domain_score_or_by_etiology.csv"
    etiology_or_path = outdir / "etiology_or_per_iqr_domain_score.csv"
    report_path = outdir / "domain_burden_or_by_stroke_etiology.html"
    summary.to_csv(summary_path, index=False)
    high_score_or.to_csv(high_or_path, index=False)
    etiology_outcome_or.to_csv(etiology_or_path, index=False)
    report_path.write_text(build_report(summary, high_score_or, etiology_outcome_or, analysis_path, domain_scores_path), encoding="utf-8")

    print(f"Summary: {summary_path}")
    print(f"High domain score ORs: {high_or_path}")
    print(f"Etiology outcome ORs: {etiology_or_path}")
    print(f"Report: {report_path}")


def load_frame(analysis_path: Path, domain_scores_path: Path) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_path, dtype=str)
    domain_scores = pd.read_csv(domain_scores_path, dtype=str)
    keep = [CASE_ID, "calcium_score", "wall_score", "peri_fat_score", "forced_domain_phenotype"]
    keep = [column for column in keep if column in domain_scores.columns]
    frame = analysis.merge(domain_scores[keep], on=CASE_ID, how="left")
    frame = frame[frame[SOURCE_LABEL].isin(ETIOLOGIES)].copy()
    if frame.empty:
        raise ValueError(f"No direct etiology labels found in {analysis_path}")
    frame["age_numeric"] = pd.to_numeric(frame.get("age"), errors="coerce")
    frame.loc[~frame["age_numeric"].between(18, 120), "age_numeric"] = np.nan
    frame["sex_male"] = frame.get("sex", "").map(
        lambda value: 1 if str(value).strip() == "Male" else 0 if str(value).strip() == "Female" else np.nan
    )
    for column in ["calcium_score", "wall_score", "peri_fat_score"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def add_domain_composites(frame: pd.DataFrame) -> None:
    for score in DOMAIN_SCORES:
        available = [column for column in score.columns if column in frame.columns]
        if len(available) != len(score.columns):
            frame[score.slug] = np.nan
            continue
        values = frame[available].apply(pd.to_numeric, errors="coerce")
        if score.require_all_columns:
            frame[score.slug] = values.mean(axis=1).where(values.notna().all(axis=1))
        else:
            frame[score.slug] = values.mean(axis=1)


def domain_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in DOMAIN_SCORES:
        if score.slug not in frame.columns:
            continue
        for etiology, group in frame.groupby(SOURCE_LABEL, dropna=False):
            values = pd.to_numeric(group[score.slug], errors="coerce").dropna()
            rows.append(
                {
                    "score": score.slug,
                    "score_label": score.label,
                    "family": score.family,
                    SOURCE_LABEL: etiology,
                    "n": int(values.size),
                    "median": safe_float(values.median()),
                    "q25": safe_float(values.quantile(0.25)),
                    "q75": safe_float(values.quantile(0.75)),
                    "mean": safe_float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def high_score_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in DOMAIN_SCORES:
        if score.slug not in frame.columns:
            continue
        values = pd.to_numeric(frame[score.slug], errors="coerce")
        analytic = frame.loc[values.notna()].copy()
        analytic["_score"] = values.loc[values.notna()]
        for threshold_name, quantile in [("above_median", 0.50), ("top_quartile", 0.75)]:
            threshold = float(analytic["_score"].quantile(quantile))
            analytic["_high_score"] = analytic["_score"].gt(threshold).astype(int)
            for adjusted in [False, True]:
                rows.extend(
                    fit_high_score_model(
                        analytic,
                        score=score,
                        threshold_name=threshold_name,
                        threshold=threshold,
                        adjusted=adjusted,
                    )
                )
    return pd.DataFrame(rows)


def fit_high_score_model(
    frame: pd.DataFrame,
    *,
    score: DomainScore,
    threshold_name: str,
    threshold: float,
    adjusted: bool,
) -> list[dict[str, object]]:
    data = frame[[SOURCE_LABEL, "_high_score", "age_numeric", "sex_male"]].copy()
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
    y = data["_high_score"].astype(int)
    result = fit_logit(y, x)
    return [
        result_row(
            result,
            term=term,
            score=score,
            model="high_domain_score_by_etiology",
            contrast=f"{term} vs {REFERENCE_ETIOLOGY}",
            outcome=f"{score.slug}_{threshold_name}",
            adjusted=adjusted,
            n=int(len(data)),
            events=int(y.sum()),
            threshold=threshold,
        )
        for term in terms
    ]


def etiology_outcome_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in DOMAIN_SCORES:
        if score.slug not in frame.columns:
            continue
        values = pd.to_numeric(frame[score.slug], errors="coerce")
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        analytic = frame.loc[values.notna()].copy()
        analytic["_score_iqr"] = values.loc[values.notna()] / iqr
        for target_name, target_fn in TARGETS.items():
            analytic["_target"] = target_fn(analytic).astype(int)
            for adjusted in [False, True]:
                rows.append(
                    fit_etiology_outcome_model(
                        analytic,
                        score=score,
                        target_name=target_name,
                        adjusted=adjusted,
                        iqr=iqr,
                    )
                )
    return pd.DataFrame(rows)


def fit_etiology_outcome_model(
    frame: pd.DataFrame,
    *,
    score: DomainScore,
    target_name: str,
    adjusted: bool,
    iqr: float,
) -> dict[str, object]:
    data = frame[["_target", "_score_iqr", "age_numeric", "sex_male"]].copy()
    if adjusted:
        data = data.dropna(subset=["age_numeric", "sex_male"])
    x = data[["_score_iqr"]].astype(float).rename(columns={"_score_iqr": "score_per_iqr"})
    if adjusted:
        x["age"] = data["age_numeric"].astype(float)
        x["sex_male"] = data["sex_male"].astype(float)
    x = sm.add_constant(x, has_constant="add")
    y = data["_target"].astype(int)
    result = fit_logit(y, x)
    return result_row(
        result,
        term="score_per_iqr",
        score=score,
        model="etiology_outcome_per_iqr_domain_score",
        contrast=f"{target_name} vs other direct etiologies",
        outcome=target_name,
        adjusted=adjusted,
        n=int(len(data)),
        events=int(y.sum()),
        score_iqr=iqr,
    )


def fit_logit(y: pd.Series, x: pd.DataFrame):
    if y.nunique(dropna=True) < 2:
        return None
    try:
        return sm.Logit(y, x.astype(float)).fit(disp=False, maxiter=200)
    except Exception:
        return None


def result_row(
    model_result,
    *,
    term: str,
    score: DomainScore,
    model: str,
    contrast: str,
    outcome: str,
    adjusted: bool,
    n: int,
    events: int,
    **extra,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "score": score.slug,
        "score_label": score.label,
        "family": score.family,
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
        "fit_status": "failed",
        **extra,
    }
    if model_result is None or term not in model_result.params.index:
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


def build_report(
    summary: pd.DataFrame,
    high_score_or: pd.DataFrame,
    etiology_outcome_or: pd.DataFrame,
    analysis_path: Path,
    domain_scores_path: Path,
) -> str:
    primary_summary = summary[summary["family"].eq("primary")].copy()
    primary_high = high_score_or[
        high_score_or["family"].eq("primary")
        & high_score_or["outcome"].str.endswith("above_median")
        & high_score_or["adjusted"].eq(True)
    ].copy()
    primary_etiology = etiology_outcome_or[
        etiology_outcome_or["family"].eq("primary") & etiology_outcome_or["adjusted"].eq(True)
    ].copy()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Domain Burden OR by Stroke Etiology</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #222; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    p {{ max-width: 980px; line-height: 1.45; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, td:nth-child(2), td:nth-child(3), td:nth-child(4) {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    .note {{ color: #555; }}
    code {{ background: #f5f5f5; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>Domain Burden OR by Stroke Etiology</h1>
  <p class="note">Etiology source: <code>{html.escape(str(analysis_path))}</code><br>
  Domain scores: <code>{html.escape(str(domain_scores_path))}</code></p>
  <p>Primary exposures are neutral domain-state scores from the PCA/clustering output, not target-trained etiology prediction scores. Mixed score is the mean of wall, periaortic fat, and calcium domain scores when all three are available. ORs compare each etiology with ESUS as reference and are adjusted for age and sex.</p>
  <h2>Primary Adjusted OR: High Domain Score by Etiology</h2>
  {format_table(primary_high)}
  <h2>Adjusted OR: Etiology per IQR Increase in Domain Score</h2>
  {format_table(primary_etiology)}
  <h2>Domain Score Distribution by Etiology</h2>
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
    parser.add_argument("--domain-scores", type=Path, default=DEFAULT_DOMAIN_SCORES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser


if __name__ == "__main__":
    main()
