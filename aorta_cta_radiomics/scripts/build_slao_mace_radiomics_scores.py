#!/usr/bin/env python
"""Build imaging-only MACE risk composites from SLAO aorta radiomics features."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = AORTA_ROOT / "outputs" / "aorta_batch_run" / "mace_slao"
DEFAULT_ANALYSIS = DEFAULT_OUTDIR / "slao_mace_aorta_modeling.csv"
DEFAULT_SCORES = DEFAULT_OUTDIR / "radiomics_scores" / "slao_mace_radiomics_scores.csv"
DEFAULT_SUMMARY = DEFAULT_OUTDIR / "radiomics_scores" / "radiomics_score_summary.json"
DEFAULT_PERFORMANCE = DEFAULT_OUTDIR / "radiomics_scores" / "radiomics_score_performance.csv"
DEFAULT_FEATURES = DEFAULT_OUTDIR / "radiomics_scores" / "radiomics_score_selected_features.csv"
DEFAULT_COLLINEARITY = DEFAULT_OUTDIR / "radiomics_scores" / "radiomics_score_collinearity_dropped.csv"
DEFAULT_STABILITY = DEFAULT_OUTDIR / "radiomics_scores" / "radiomics_score_stability_selection.csv"
DEFAULT_REPORT = DEFAULT_OUTDIR / "radiomics_scores" / "radiomics_score_report.html"
DEFAULT_ASSETS = DEFAULT_OUTDIR / "radiomics_scores" / "report_assets"

CASE_ID = "case_id"
OUTCOME = "mace_primary"
REPORT_EYEBROW = "SLAO MACE imaging-only composites"
REPORT_TITLE = "Radiomics risk dashboard"
REPORT_PAGE_TITLE = "SLAO Radiomics MACE Scores"
REPORT_OUTCOME_NAME = "MACE"
REPORT_POSITIVE_LABEL = "MACE"
REPORT_NEGATIVE_LABEL = "No MACE"
REPORT_EXTRA_NOTE = ""
CASE_GROUP_LABEL = "Study arm"

DOMAIN_LABELS = {
    "calcium": "Calcium",
    "fat": "Periaortic fat",
    "wall_from_fat": "Wall-from-fat",
    "wall_thickness": "Wall thickness",
    "all_imaging": "All imaging",
    "domain_sum": "Domain sum",
}

DOMAIN_COLORS = {
    "calcium": "#8a5a2b",
    "fat": "#4c7f73",
    "wall_from_fat": "#6b5876",
    "wall_thickness": "#335f86",
    "all_imaging": "#333333",
    "domain_sum": "#8c3d3d",
}


@dataclass
class FeatureSelection:
    domain: str
    feature: str
    rank: int
    n_train: int
    train_auc: float
    train_smd: float
    sign: int
    missing_fraction: float
    selection_method: str
    coefficient: float = math.nan


@dataclass
class CollinearityDrop:
    domain: str
    feature: str
    rank: int
    correlated_with: str
    abs_correlation: float
    correlation_method: str
    n_pair: int
    train_auc: float
    train_smd: float
    kept_train_auc: float
    kept_train_smd: float
    missing_fraction: float


@dataclass
class RobustnessFilter:
    status: str
    path: str
    allowed_features: set[str] | None
    rejected_features: set[str]
    total_rows: int = 0
    kept_rows: int = 0
    details: str = ""


def main() -> None:
    args = build_parser().parse_args()
    analysis_path = args.analysis.expanduser().resolve()
    scores_path = args.scores.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    performance_path = args.performance.expanduser().resolve()
    selected_features_path = args.selected_features.expanduser().resolve()
    collinearity_path = args.collinearity_report.expanduser().resolve()
    stability_path = args.stability_report.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(analysis_path, dtype=str)
    if OUTCOME not in raw.columns:
        raise ValueError(f"Expected outcome column '{OUTCOME}' in {analysis_path}.")

    frame = raw.copy()
    add_aorta_size_ratio_features(frame)
    y = pd.to_numeric(frame[OUTCOME], errors="coerce")
    keep = y.isin([0, 1])
    frame = frame.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)

    feature_columns = imaging_feature_columns(frame)
    domains = domain_feature_map(feature_columns)
    if args.include_ratios:
        ratio_domains = domain_feature_map([column for column in frame.columns if column.startswith("ratio_aorta_size__")])
        for domain, columns in ratio_domains.items():
            domains[domain].extend(column for column in columns if column not in domains[domain])
    robustness_filter = load_robustness_filter(args)
    domains = apply_robustness_filter(domains, robustness_filter)

    score_frame = frame[
        [
            CASE_ID,
            "record_id",
            "source_cohort",
            "study_arm",
            OUTCOME,
        ]
    ].copy()

    all_selected_rows: list[FeatureSelection] = []
    all_collinearity_rows: list[CollinearityDrop] = []
    score_columns: list[str] = []
    domain_order = ["calcium", "fat", "wall_from_fat", "wall_thickness"]
    for domain in domain_order:
        columns = domains.get(domain, [])
        if not columns:
            continue
        result = cross_validated_scores(
            frame=frame,
            y=y,
            columns=columns,
            domain=domain,
            top_k=args.top_k,
            n_splits=args.folds,
            max_missing=args.max_missing,
            min_variance=args.min_variance,
            correlation_threshold=args.correlation_threshold,
            correlation_method=args.correlation_method,
            min_correlation_pair_n=args.min_correlation_pair_n,
            elastic_net_c=args.elastic_net_c,
            elastic_net_l1_ratio=args.elastic_net_l1_ratio,
            random_state=args.random_state,
        )
        append_score_result(score_frame, result)
        all_selected_rows.extend(result["selected_features"])
        all_collinearity_rows.extend(result["collinearity_drops"])
        score_columns.extend(result["score_columns"])

    all_columns = list(dict.fromkeys(column for domain in domain_order for column in domains.get(domain, [])))
    all_result = cross_validated_scores(
        frame=frame,
        y=y,
        columns=all_columns,
        domain="all_imaging",
        top_k=args.all_top_k,
        n_splits=args.folds,
        max_missing=args.max_missing,
        min_variance=args.min_variance,
        correlation_threshold=args.correlation_threshold,
        correlation_method=args.correlation_method,
        min_correlation_pair_n=args.min_correlation_pair_n,
        elastic_net_c=args.elastic_net_c,
        elastic_net_l1_ratio=args.elastic_net_l1_ratio,
        random_state=args.random_state,
    )
    append_score_result(score_frame, all_result)
    all_selected_rows.extend(all_result["selected_features"])
    all_collinearity_rows.extend(all_result["collinearity_drops"])
    score_columns.extend(all_result["score_columns"])

    add_domain_sum_scores(score_frame, domain_order)
    score_columns.extend(
        [
            "domain_sum__signed_z_cv",
            "domain_sum__probability_mean_cv",
            "domain_sum__elastic_net_probability_mean_cv",
        ]
    )
    platt_columns = add_platt_recalibrated_scores(
        score_frame=score_frame,
        y=y,
        score_columns=score_columns,
        n_splits=args.folds,
        random_state=args.random_state,
    )
    score_columns.extend(platt_columns)

    ratio_columns = [column for column in frame.columns if column.startswith("ratio_aorta_size__")]
    output = pd.concat([score_frame, frame[ratio_columns]], axis=1)
    output.to_csv(scores_path, index=False)

    selected_features = enrich_selected_features(pd.DataFrame(selection.__dict__ for selection in all_selected_rows))
    selected_features.to_csv(selected_features_path, index=False)
    collinearity_drops = enrich_collinearity_drops(pd.DataFrame(drop.__dict__ for drop in all_collinearity_rows))
    collinearity_drops.to_csv(collinearity_path, index=False)
    stability_selection = build_stability_selection(
        frame=frame,
        y=y,
        domains=domains,
        args=args,
    )
    stability_selection.to_csv(stability_path, index=False)
    validation_stage = validation_stage_summary(frame, y, args)

    performance = performance_table(score_frame, y, score_columns)
    performance.to_csv(performance_path, index=False)

    summary = build_summary(
        analysis_path=analysis_path,
        scores_path=scores_path,
        performance_path=performance_path,
        selected_features_path=selected_features_path,
        collinearity_path=collinearity_path,
        stability_path=stability_path,
        report_path=report_path,
        raw_rows=len(raw),
        scored_rows=len(score_frame),
        events=int(y.sum()),
        ratio_columns=ratio_columns,
        domains=domains,
        performance=performance,
        selected_features=selected_features,
        collinearity_drops=collinearity_drops,
        stability_selection=stability_selection,
        robustness_filter=robustness_filter,
        validation_stage=validation_stage,
        args=args,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_html = build_report(
        scores=score_frame,
        performance=performance,
        selected_features=selected_features,
        collinearity_drops=collinearity_drops,
        stability_selection=stability_selection,
        summary=summary,
        assets_dir=assets_dir,
    )
    report_path.write_text(report_html, encoding="utf-8")

    print(f"Scored rows: {len(score_frame)}")
    print(f"{REPORT_POSITIVE_LABEL} events: {int(y.sum())}")
    print(f"Ratio features: {len(ratio_columns)}")
    print(f"Collinear candidates dropped: {len(collinearity_drops)}")
    print(f"Stability rows: {len(stability_selection)}")
    print(f"Scores: {scores_path}")
    print(f"Performance: {performance_path}")
    print(f"Report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--performance", type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument("--selected-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--collinearity-report", type=Path, default=DEFAULT_COLLINEARITY)
    parser.add_argument("--stability-report", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--all-top-k", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-missing", type=float, default=0.25)
    parser.add_argument("--min-variance", type=float, default=1e-12)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--correlation-method", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--min-correlation-pair-n", type=int, default=40)
    parser.add_argument("--elastic-net-c", type=float, default=0.2)
    parser.add_argument("--elastic-net-l1-ratio", type=float, default=0.5)
    parser.add_argument("--stability-resamples", type=int, default=100)
    parser.add_argument("--stability-train-fraction", type=float, default=0.75)
    parser.add_argument("--stability-threshold", type=float, default=0.50)
    parser.add_argument("--robustness-csv", type=Path)
    parser.add_argument("--robustness-min-icc", type=float, default=0.75)
    parser.add_argument("--robustness-require-listed", action="store_true")
    parser.add_argument("--validation-group-column")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-ratios", action=argparse.BooleanOptionalAction, default=True)
    return parser


def add_aorta_size_ratio_features(frame: pd.DataFrame) -> None:
    denominator_priority = [
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__input_aorta_volume_mm3",
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__hu_refined_aorta_volume_mm3",
        "aorta__calcium_omics__aortic_volume_mm3__thr_dynamic_lumen_referenced_seed500HU",
    ]
    aorta_volume = first_numeric_column(frame, denominator_priority)
    aortic_length = first_numeric_column(
        frame,
        [
            "aorta__calcium_omics__aortic_length_cm__thr_dynamic_lumen_referenced_seed500HU",
            "aorta__calcium_omics__aortic_length_cm__thr_300HU",
        ],
    )
    wall_volume = first_numeric_column(frame, ["aortic_wall__wall_thickness__wall_volume_mm3"])
    equiv_radius_mm = None
    equiv_diameter_mm = None
    if aorta_volume is not None and aortic_length is not None:
        aortic_length_mm = aortic_length * 10.0
        mean_aortic_area_mm2 = safe_divide(aorta_volume, aortic_length_mm)
        equiv_radius_mm = np.sqrt(mean_aortic_area_mm2.where(mean_aortic_area_mm2 > 0) / math.pi)
        equiv_diameter_mm = equiv_radius_mm * 2.0

    for column in list(frame.columns):
        domain = feature_domain(column)
        if not domain:
            continue
        lower = column.lower()
        if "volume" not in lower:
            continue
        if "aorta_volume" in lower or "aortic_volume" in lower:
            continue
        numerator = pd.to_numeric(frame[column], errors="coerce")
        safe_name = sanitize_feature_name(column)
        if aorta_volume is not None:
            frame[f"ratio_aorta_size__{safe_name}__per_aorta_volume"] = safe_divide(numerator, aorta_volume)
        if aortic_length is not None:
            frame[f"ratio_aorta_size__{safe_name}__per_aortic_length_cm"] = safe_divide(numerator, aortic_length)

    if equiv_radius_mm is not None and equiv_diameter_mm is not None:
        for column in list(frame.columns):
            if feature_domain(column) != "wall_thickness":
                continue
            lower = column.lower()
            if "volume" in lower or "_mm" not in lower:
                continue
            numerator = pd.to_numeric(frame[column], errors="coerce")
            safe_name = sanitize_feature_name(column)
            frame[f"ratio_aorta_size__{safe_name}__per_equiv_aortic_radius_mm"] = safe_divide(
                numerator,
                equiv_radius_mm,
            )
            frame[f"ratio_aorta_size__{safe_name}__per_equiv_aortic_diameter_mm"] = safe_divide(
                numerator,
                equiv_diameter_mm,
            )

    if wall_volume is not None:
        gt4 = "aortic_wall__wall_thickness_threshold__wall_thickness_gt4mm_volume_mm3__thr_> 4 mm"
        if gt4 in frame.columns:
            frame["ratio_aorta_size__wall_gt4mm_volume__per_wall_volume"] = safe_divide(
                pd.to_numeric(frame[gt4], errors="coerce"),
                wall_volume,
            )


def first_numeric_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return None


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator > 0)
    return numerator / denominator


def sanitize_feature_name(column: str) -> str:
    safe = []
    for char in column:
        if char.isalnum():
            safe.append(char.lower())
        else:
            safe.append("_")
    text = "".join(safe)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def imaging_feature_columns(frame: pd.DataFrame) -> list[str]:
    reserved = {
        CASE_ID,
        OUTCOME,
        "record_id",
        "source_cohort",
        "study_arm",
        "mace_composite",
    }
    columns = []
    for column in frame.columns:
        if column in reserved or column.startswith("ratio_aorta_size__"):
            continue
        if is_aorta_size_denominator(column):
            continue
        if feature_domain(column):
            columns.append(column)
    return columns


def domain_feature_map(columns: Iterable[str]) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        domain = feature_domain(column)
        if domain:
            domains[domain].append(column)
    return dict(domains)


def load_robustness_filter(args: argparse.Namespace) -> RobustnessFilter:
    if args.robustness_csv is None:
        return RobustnessFilter(
            status="not_available",
            path="",
            allowed_features=None,
            rejected_features=set(),
            details="No test-retest/segmentation/acquisition robustness CSV was provided.",
        )
    path = args.robustness_csv.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Robustness CSV not found: {path}")
    table = pd.read_csv(path)
    feature_column = first_matching_column(table, ["feature", "feature_name", "variable", "column", "name"])
    if feature_column is None:
        raise ValueError("Robustness CSV must include a feature/feature_name/variable/column/name column.")

    pass_column = first_matching_column(table, ["robust", "keep", "pass", "passes", "include"])
    metric_columns = [
        column
        for column in table.columns
        if column != feature_column and ("icc" in column.lower() or "robust" in column.lower())
    ]
    if pass_column is not None:
        passed = table[pass_column].map(parse_boolish)
    elif metric_columns:
        metrics = table[metric_columns].apply(pd.to_numeric, errors="coerce")
        passed = metrics.ge(args.robustness_min_icc).all(axis=1)
    else:
        raise ValueError(
            "Robustness CSV must include a boolean robust/keep/pass/include column or numeric ICC/robustness columns."
        )

    feature_names = table[feature_column].astype(str)
    allowed = set(feature_names.loc[passed.fillna(False)])
    rejected = set(feature_names.loc[~passed.fillna(False)])
    return RobustnessFilter(
        status="applied",
        path=str(path),
        allowed_features=allowed if args.robustness_require_listed else None,
        rejected_features=rejected,
        total_rows=int(len(table)),
        kept_rows=int(passed.fillna(False).sum()),
        details=(
            "Applied robustness screening from CSV. "
            + ("Unlisted features were rejected." if args.robustness_require_listed else "Unlisted features were allowed.")
        ),
    )


def first_matching_column(table: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in table.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def parse_boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "pass", "passed", "keep", "include", "robust"}


def apply_robustness_filter(
    domains: dict[str, list[str]],
    robustness_filter: RobustnessFilter,
) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {}
    for domain, columns in domains.items():
        kept = []
        for column in columns:
            if robustness_filter.allowed_features is not None and column not in robustness_filter.allowed_features:
                continue
            if column in robustness_filter.rejected_features:
                continue
            kept.append(column)
        filtered[domain] = kept
    return filtered


def feature_domain(name: str) -> str:
    lower = name.lower()
    if "wall_thickness" in lower or "wall_gt4mm" in lower:
        return "wall_thickness"
    if "wall_from_fat" in lower or "experimental_wall_from_fat" in lower:
        return "wall_from_fat"
    if "calcium" in lower or "calcification" in lower or "agatston" in lower:
        return "calcium"
    if "fat" in lower or "periaortic" in lower:
        return "fat"
    return ""


def is_aorta_size_denominator(name: str) -> bool:
    lower = name.lower()
    pure_aorta_size_tokens = [
        "aorta__calcium_omics__aortic_length_cm__",
        "aorta__calcium_omics__aortic_volume_mm3__",
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__input_aorta_volume_mm3",
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__hu_refined_aorta_volume_mm3",
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__lumen_added_outside_input_aorta_volume_mm3",
    ]
    return any(token in lower for token in pure_aorta_size_tokens)


def cross_validated_scores(
    *,
    frame: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    domain: str,
    top_k: int,
    n_splits: int,
    max_missing: float,
    min_variance: float,
    correlation_threshold: float,
    correlation_method: str,
    min_correlation_pair_n: int,
    elastic_net_c: float,
    elastic_net_l1_ratio: float,
    random_state: int,
) -> dict[str, object]:
    if not columns:
        return {
            "domain": domain,
            "score_columns": [],
            "selected_features": [],
            "collinearity_drops": [],
        }
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    signed_score = np.full(len(y), np.nan)
    probability = np.full(len(y), np.nan)
    elastic_net_probability = np.full(len(y), np.nan)
    selected_features: list[FeatureSelection] = []
    collinearity_drops: list[CollinearityDrop] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(matrix, y), start=1):
        x_train = matrix.iloc[train_idx].reset_index(drop=True)
        x_test = matrix.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        usable = usable_features(x_train, max_missing=max_missing, min_variance=min_variance)
        if not usable:
            continue
        ranking = rank_features(x_train[usable], y_train)
        selected, dropped = select_non_collinear_features(
            ranking=ranking,
            train=x_train[usable],
            domain=f"{domain}:fold{fold}",
            top_k=top_k,
            correlation_threshold=correlation_threshold,
            correlation_method=correlation_method,
            min_pair_n=min_correlation_pair_n,
        )
        collinearity_drops.extend(dropped)
        if not selected:
            continue
        selected_columns = [item.feature for item in selected]
        for rank, item in enumerate(selected, start=1):
            selected_features.append(
                FeatureSelection(
                    domain=f"{domain}:fold{fold}",
                    feature=item.feature,
                    rank=rank,
                    n_train=item.n_train,
                    train_auc=item.train_auc,
                    train_smd=item.train_smd,
                    sign=item.sign,
                    missing_fraction=item.missing_fraction,
                    selection_method="relevance_collinearity",
                )
            )

        train_selected = x_train[selected_columns]
        test_selected = x_test[selected_columns]
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train_imp = imputer.fit_transform(train_selected)
        x_test_imp = imputer.transform(test_selected)
        x_train_z = finite_clipped_z(scaler.fit_transform(x_train_imp))
        x_test_z = finite_clipped_z(scaler.transform(x_test_imp))
        signs = np.array([item.sign for item in selected], dtype=float)
        signed_score[test_idx] = (x_test_z * signs).mean(axis=1)

        if len(np.unique(y_train)) > 1:
            model = LogisticRegression(
                C=0.3,
                solver="lbfgs",
                max_iter=5000,
                random_state=random_state,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
                model.fit(x_train_z, y_train)
                finite_model = np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
                decision = model.decision_function(x_test_z) if finite_model else np.array([])
            if finite_model:
                decision = np.clip(decision, -30, 30)
                probability[test_idx] = 1.0 / (1.0 + np.exp(-decision))

            elastic_net = LogisticRegression(
                C=elastic_net_c,
                l1_ratio=elastic_net_l1_ratio,
                solver="saga",
                max_iter=10000,
                random_state=random_state + fold,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
                elastic_net.fit(x_train_z, y_train)
                finite_elastic_net = np.isfinite(elastic_net.coef_).all() and np.isfinite(elastic_net.intercept_).all()
                elastic_net_decision = elastic_net.decision_function(x_test_z) if finite_elastic_net else np.array([])
            if finite_elastic_net:
                elastic_net_decision = np.clip(elastic_net_decision, -30, 30)
                elastic_net_probability[test_idx] = 1.0 / (1.0 + np.exp(-elastic_net_decision))
                coefficients = elastic_net.coef_.ravel()
                nonzero = [
                    (selected[i], float(coefficient))
                    for i, coefficient in enumerate(coefficients)
                    if abs(float(coefficient)) > 1e-8
                ]
                nonzero.sort(key=lambda item: abs(item[1]), reverse=True)
                for rank, (item, coefficient) in enumerate(nonzero, start=1):
                    selected_features.append(
                        FeatureSelection(
                            domain=f"{domain}:fold{fold}",
                            feature=item.feature,
                            rank=rank,
                            n_train=item.n_train,
                            train_auc=item.train_auc,
                            train_smd=item.train_smd,
                            sign=1 if coefficient >= 0 else -1,
                            missing_fraction=item.missing_fraction,
                            selection_method="elastic_net_nonzero",
                            coefficient=coefficient,
                        )
                    )

    signed_column = f"{domain}__signed_z_cv"
    probability_column = f"{domain}__probability_cv"
    elastic_net_column = f"{domain}__elastic_net_probability_cv"
    return {
        "domain": domain,
        "score_columns": [signed_column, probability_column, elastic_net_column],
        "signed_column": signed_column,
        "probability_column": probability_column,
        "elastic_net_column": elastic_net_column,
        "signed_score": signed_score,
        "probability": probability,
        "elastic_net_probability": elastic_net_probability,
        "selected_features": selected_features,
        "collinearity_drops": collinearity_drops,
    }


def build_stability_selection(
    *,
    frame: pd.DataFrame,
    y: pd.Series,
    domains: dict[str, list[str]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.stability_resamples <= 0:
        return pd.DataFrame()
    domain_order = ["calcium", "fat", "wall_from_fat", "wall_thickness"]
    domain_columns = {domain: list(dict.fromkeys(domains.get(domain, []))) for domain in domain_order}
    domain_columns["all_imaging"] = list(
        dict.fromkeys(column for domain in domain_order for column in domain_columns.get(domain, []))
    )
    rows = []
    splitter = StratifiedShuffleSplit(
        n_splits=args.stability_resamples,
        train_size=args.stability_train_fraction,
        random_state=args.random_state + 1000,
    )
    for domain, columns in domain_columns.items():
        if not columns:
            continue
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        top_k = args.all_top_k if domain == "all_imaging" else args.top_k
        selection_counts: Counter[str] = Counter()
        coefficient_sums: defaultdict[str, float] = defaultdict(float)
        coefficient_abs_sums: defaultdict[str, float] = defaultdict(float)
        eligible_resamples = 0
        for resample, (train_idx, _) in enumerate(splitter.split(matrix, y), start=1):
            x_train = matrix.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            if y_train.nunique() < 2:
                continue
            usable = usable_features(x_train, max_missing=args.max_missing, min_variance=args.min_variance)
            if not usable:
                continue
            ranking = rank_features(x_train[usable], y_train)
            selected, _ = select_non_collinear_features(
                ranking=ranking,
                train=x_train[usable],
                domain=f"{domain}:stability{resample}",
                top_k=top_k,
                correlation_threshold=args.correlation_threshold,
                correlation_method=args.correlation_method,
                min_pair_n=args.min_correlation_pair_n,
            )
            if not selected:
                continue
            selected_columns = [item.feature for item in selected]
            x_selected = x_train[selected_columns]
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_imp = imputer.fit_transform(x_selected)
            x_z = finite_clipped_z(scaler.fit_transform(x_imp))
            model = LogisticRegression(
                C=args.elastic_net_c,
                l1_ratio=args.elastic_net_l1_ratio,
                solver="saga",
                max_iter=10000,
                random_state=args.random_state + 1000 + resample,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
                model.fit(x_z, y_train)
            if not np.isfinite(model.coef_).all():
                continue
            eligible_resamples += 1
            for column, coefficient in zip(selected_columns, model.coef_.ravel()):
                coefficient = float(coefficient)
                if abs(coefficient) <= 1e-8:
                    continue
                selection_counts[column] += 1
                coefficient_sums[column] += coefficient
                coefficient_abs_sums[column] += abs(coefficient)

        for feature, selected_count in selection_counts.items():
            rows.append(
                {
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS.get(domain, domain),
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "feature_description": feature_description(feature),
                    "selected_count": int(selected_count),
                    "eligible_resamples": int(eligible_resamples),
                    "selection_probability": float(selected_count / eligible_resamples) if eligible_resamples else math.nan,
                    "stable": bool(selected_count / eligible_resamples >= args.stability_threshold)
                    if eligible_resamples
                    else False,
                    "mean_coefficient": float(coefficient_sums[feature] / selected_count),
                    "mean_abs_coefficient": float(coefficient_abs_sums[feature] / selected_count),
                    "stability_threshold": float(args.stability_threshold),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["domain", "selection_probability", "mean_abs_coefficient"],
        ascending=[True, False, False],
    )


def validation_stage_summary(frame: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> dict[str, object]:
    if not args.validation_group_column:
        candidates = [
            column
            for column in frame.columns
            if looks_like_validation_group_column(column)
        ]
        return {
            "status": "not_available",
            "group_column": "",
            "groups": [],
            "details": (
                "No external cohort or center/site/scanner grouping column is available in this SLAO modeling table. "
                f"Candidate columns found: {', '.join(candidates) if candidates else 'none'}."
            ),
        }
    if args.validation_group_column not in frame.columns:
        return {
            "status": "not_available",
            "group_column": args.validation_group_column,
            "groups": [],
            "details": f"Requested validation group column was not found: {args.validation_group_column}.",
        }
    group_values = frame[args.validation_group_column].astype(str).fillna("missing")
    groups = []
    for group, indices in group_values.groupby(group_values).groups.items():
        y_group = y.loc[list(indices)]
        groups.append({"group": group, "n": int(len(indices)), "events": int(y_group.sum())})
    valid_groups = [row for row in groups if row["events"] > 0 and row["events"] < row["n"]]
    status = "available_for_center_heldout" if len(valid_groups) >= 2 else "not_usable"
    return {
        "status": status,
        "group_column": args.validation_group_column,
        "groups": groups,
        "details": (
            "A group column is present, but this script reports availability only; run a dedicated locked validation "
            "once an external/center-held-out split is finalized."
            if status == "available_for_center_heldout"
            else "Group column does not contain at least two groups with both outcome classes."
        ),
    }


def looks_like_validation_group_column(column: str) -> bool:
    lower = column.lower()
    if any(token in lower for token in ["center", "scanner", "institution"]):
        return True
    tokens = [token for token in "".join(char if char.isalnum() else " " for char in lower).split() if token]
    return "site" in tokens


@dataclass
class RankedFeature:
    feature: str
    n_train: int
    train_auc: float
    train_smd: float
    sign: int
    missing_fraction: float


def usable_features(train: pd.DataFrame, *, max_missing: float, min_variance: float) -> list[str]:
    usable = []
    for column in train.columns:
        values = pd.to_numeric(train[column], errors="coerce")
        missing = float(values.isna().mean())
        if missing > max_missing:
            continue
        variance = float(values.var(skipna=True))
        if math.isnan(variance) or variance <= min_variance:
            continue
        usable.append(column)
    return usable


def rank_features(train: pd.DataFrame, y_train: pd.Series) -> list[RankedFeature]:
    rows: list[RankedFeature] = []
    y_values = y_train.to_numpy()
    for column in train.columns:
        values = pd.to_numeric(train[column], errors="coerce")
        valid = values.notna().to_numpy()
        if valid.sum() < 10:
            continue
        feature_values = values.loc[valid].to_numpy(dtype=float)
        outcomes = y_values[valid]
        if len(np.unique(outcomes)) < 2:
            continue
        event_values = feature_values[outcomes == 1]
        nonevent_values = feature_values[outcomes == 0]
        smd = standardized_mean_difference(event_values, nonevent_values)
        auc = safe_auc(outcomes, feature_values)
        direction_auc = max(auc, 1.0 - auc) if not math.isnan(auc) else 0.5
        sign = 1 if smd >= 0 else -1
        rows.append(
            RankedFeature(
                feature=column,
                n_train=int(valid.sum()),
                train_auc=float(direction_auc),
                train_smd=float(smd),
                sign=sign,
                missing_fraction=float(1 - valid.mean()),
            )
        )
    return sorted(rows, key=lambda item: (abs(item.train_smd), item.train_auc), reverse=True)


def select_non_collinear_features(
    *,
    ranking: list[RankedFeature],
    train: pd.DataFrame,
    domain: str,
    top_k: int,
    correlation_threshold: float,
    correlation_method: str,
    min_pair_n: int,
) -> tuple[list[RankedFeature], list[CollinearityDrop]]:
    selected: list[RankedFeature] = []
    dropped: list[CollinearityDrop] = []
    rank_lookup = {item.feature: rank for rank, item in enumerate(ranking, start=1)}
    for item in ranking:
        if len(selected) >= top_k:
            break
        blocker = most_correlated_selected_feature(
            item=item,
            selected=selected,
            train=train,
            method=correlation_method,
            min_pair_n=min_pair_n,
        )
        if blocker is not None and blocker["abs_correlation"] >= correlation_threshold:
            kept = blocker["kept"]
            dropped.append(
                CollinearityDrop(
                    domain=domain,
                    feature=item.feature,
                    rank=rank_lookup[item.feature],
                    correlated_with=kept.feature,
                    abs_correlation=float(blocker["abs_correlation"]),
                    correlation_method=correlation_method,
                    n_pair=int(blocker["n_pair"]),
                    train_auc=item.train_auc,
                    train_smd=item.train_smd,
                    kept_train_auc=kept.train_auc,
                    kept_train_smd=kept.train_smd,
                    missing_fraction=item.missing_fraction,
                )
            )
            continue
        selected.append(item)
    return selected, dropped


def most_correlated_selected_feature(
    *,
    item: RankedFeature,
    selected: list[RankedFeature],
    train: pd.DataFrame,
    method: str,
    min_pair_n: int,
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for kept in selected:
        correlation, n_pair = pairwise_correlation(
            train[item.feature],
            train[kept.feature],
            method=method,
            min_pair_n=min_pair_n,
        )
        if math.isnan(correlation):
            continue
        abs_correlation = abs(correlation)
        if best is None or abs_correlation > float(best["abs_correlation"]):
            best = {
                "kept": kept,
                "abs_correlation": abs_correlation,
                "n_pair": n_pair,
            }
    return best


def pairwise_correlation(
    left_values: pd.Series,
    right_values: pd.Series,
    *,
    method: str,
    min_pair_n: int,
) -> tuple[float, int]:
    left = pd.to_numeric(left_values, errors="coerce")
    right = pd.to_numeric(right_values, errors="coerce")
    valid = left.notna() & right.notna()
    n_pair = int(valid.sum())
    if n_pair < min_pair_n:
        return math.nan, n_pair
    left = left.loc[valid]
    right = right.loc[valid]
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return math.nan, n_pair
    if method == "spearman":
        left = left.rank(method="average")
        right = right.rank(method="average")
    correlation = left.corr(right, method="pearson")
    if pd.isna(correlation):
        return math.nan, n_pair
    return float(correlation), n_pair


def standardized_mean_difference(event_values: np.ndarray, nonevent_values: np.ndarray) -> float:
    if len(event_values) == 0 or len(nonevent_values) == 0:
        return math.nan
    event_mean = float(np.mean(event_values))
    nonevent_mean = float(np.mean(nonevent_values))
    if len(event_values) + len(nonevent_values) < 3:
        return 0.0
    event_var = float(np.var(event_values, ddof=1)) if len(event_values) > 1 else 0.0
    nonevent_var = float(np.var(nonevent_values, ddof=1)) if len(nonevent_values) > 1 else 0.0
    pooled = ((len(event_values) - 1) * event_var + (len(nonevent_values) - 1) * nonevent_var) / (
        len(event_values) + len(nonevent_values) - 2
    )
    if pooled <= 0 or math.isnan(pooled):
        return 0.0
    return (event_mean - nonevent_mean) / math.sqrt(pooled)


def safe_auc(y_true: np.ndarray | pd.Series, values: np.ndarray | pd.Series) -> float:
    try:
        return float(roc_auc_score(y_true, values))
    except ValueError:
        return math.nan


def append_score_result(score_frame: pd.DataFrame, result: dict[str, object]) -> None:
    signed_column = str(result.get("signed_column", ""))
    probability_column = str(result.get("probability_column", ""))
    elastic_net_column = str(result.get("elastic_net_column", ""))
    if signed_column:
        score_frame[signed_column] = result.get("signed_score")
    if probability_column:
        score_frame[probability_column] = result.get("probability")
    if elastic_net_column:
        score_frame[elastic_net_column] = result.get("elastic_net_probability")


def finite_clipped_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=8.0, neginf=-8.0)
    return np.clip(array, -8.0, 8.0).astype(np.float64, copy=False)


def add_domain_sum_scores(score_frame: pd.DataFrame, domain_order: list[str]) -> None:
    signed_columns = [f"{domain}__signed_z_cv" for domain in domain_order if f"{domain}__signed_z_cv" in score_frame.columns]
    probability_columns = [
        f"{domain}__probability_cv" for domain in domain_order if f"{domain}__probability_cv" in score_frame.columns
    ]
    elastic_net_columns = [
        f"{domain}__elastic_net_probability_cv"
        for domain in domain_order
        if f"{domain}__elastic_net_probability_cv" in score_frame.columns
    ]
    score_frame["domain_sum__signed_z_cv"] = score_frame[signed_columns].mean(axis=1) if signed_columns else np.nan
    score_frame["domain_sum__probability_mean_cv"] = (
        score_frame[probability_columns].mean(axis=1) if probability_columns else np.nan
    )
    score_frame["domain_sum__elastic_net_probability_mean_cv"] = (
        score_frame[elastic_net_columns].mean(axis=1) if elastic_net_columns else np.nan
    )


def add_platt_recalibrated_scores(
    *,
    score_frame: pd.DataFrame,
    y: pd.Series,
    score_columns: list[str],
    n_splits: int,
    random_state: int,
) -> list[str]:
    platt_columns = []
    for score_index, column in enumerate(dict.fromkeys(score_columns)):
        if column not in score_frame.columns or not is_probability_score(column) or is_platt_score(column):
            continue
        probabilities = pd.to_numeric(score_frame[column], errors="coerce").clip(1e-6, 1 - 1e-6)
        valid = probabilities.notna() & y.isin([0, 1])
        if valid.sum() < 10:
            continue
        y_valid = y.loc[valid].astype(int)
        class_counts = y_valid.value_counts()
        if len(class_counts) < 2 or int(class_counts.min()) < 2:
            continue
        folds = min(max(2, n_splits), int(class_counts.min()))
        calibrated = pd.Series(np.nan, index=score_frame.index, dtype=float)
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state + 20_000 + score_index,
        )
        valid_indices = probabilities.loc[valid].index.to_numpy()
        logits = logit_probabilities(probabilities.loc[valid]).to_numpy(dtype=float).reshape(-1, 1)
        labels = y_valid.to_numpy(dtype=int)
        for train_idx, test_idx in splitter.split(logits, labels):
            if len(np.unique(labels[train_idx])) < 2:
                continue
            calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            calibrator.fit(logits[train_idx], labels[train_idx])
            calibrated.loc[valid_indices[test_idx]] = calibrator.predict_proba(logits[test_idx])[:, 1]
        platt_column = platt_score_name(column)
        score_frame[platt_column] = calibrated
        platt_columns.append(platt_column)
    return platt_columns


def logit_probabilities(probabilities: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(probabilities, errors="coerce").clip(1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def performance_table(score_frame: pd.DataFrame, y: pd.Series, score_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in dict.fromkeys(score_columns):
        if column not in score_frame.columns:
            continue
        values = pd.to_numeric(score_frame[column], errors="coerce")
        valid = values.notna() & y.notna()
        if valid.sum() == 0 or y.loc[valid].nunique() < 2:
            continue
        score_values = values.loc[valid].to_numpy(dtype=float)
        y_values = y.loc[valid].to_numpy(dtype=int)
        auc = float(roc_auc_score(y_values, score_values))
        direction_auc = max(auc, 1.0 - auc)
        ap = float(average_precision_score(y_values, score_values))
        rows.append(
            {
                "score_name": column,
                "domain": score_domain(column),
                "score_type": score_type(column),
                "n": int(valid.sum()),
                "events": int(y_values.sum()),
                "auc": auc,
                "directional_auc": direction_auc,
                "average_precision": ap,
                "event_rate_low_quartile": quartile_event_rate(score_values, y_values, "low"),
                "event_rate_high_quartile": quartile_event_rate(score_values, y_values, "high"),
            }
        )
    return pd.DataFrame(rows).sort_values(["directional_auc", "average_precision"], ascending=False)


def score_domain(column: str) -> str:
    if column.startswith("domain_sum"):
        return "domain_sum"
    return column.split("__", 1)[0]


def score_type(column: str) -> str:
    if is_platt_score(column):
        base_type = score_type(platt_base_score_name(column))
        if base_type == "cv_elastic_net_probability":
            return "cv_elastic_net_platt_probability"
        return "cv_platt_probability"
    if "elastic_net" in column:
        return "cv_elastic_net_probability"
    if "probability" in column:
        return "cv_logistic_probability"
    return "cv_signed_zsum"


def quartile_event_rate(score_values: np.ndarray, y_values: np.ndarray, side: str) -> float:
    if len(score_values) < 4:
        return math.nan
    threshold = np.nanquantile(score_values, 0.25 if side == "low" else 0.75)
    mask = score_values <= threshold if side == "low" else score_values >= threshold
    if not mask.any():
        return math.nan
    return float(y_values[mask].mean())


def domain_variable_counts(
    *,
    domains: dict[str, list[str]],
    selected_features: pd.DataFrame,
    collinearity_drops: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    domain_order = ["calcium", "fat", "wall_from_fat", "wall_thickness", "all_imaging", "domain_sum"]
    rows = []
    selected = selected_features.copy()
    drops = collinearity_drops.copy()
    if not selected.empty and "base_domain" not in selected.columns:
        selected["base_domain"] = selected["domain"].str.split(":").str[0]
    if not drops.empty and "base_domain" not in drops.columns:
        drops["base_domain"] = drops["domain"].str.split(":").str[0]

    for domain in domain_order:
        if domain == "all_imaging":
            candidate_count = sum(len(domains.get(base, [])) for base in ["calcium", "fat", "wall_from_fat", "wall_thickness"])
            target_per_fold = args.all_top_k
            note = "Pooled imaging model; selected after domain pooling."
        elif domain == "domain_sum":
            candidate_count = 4
            target_per_fold = 4
            note = "No raw radiomics. Mean of the four domain scores."
        else:
            candidate_count = len(domains.get(domain, []))
            target_per_fold = args.top_k
            note = "Domain model; selected independently inside each fold."

        domain_selected = selected[selected["base_domain"] == domain] if not selected.empty else selected
        selected_primary = (
            domain_selected[domain_selected["selection_method"] == "relevance_collinearity"]
            if not domain_selected.empty
            else domain_selected
        )
        elastic_net_selected = (
            domain_selected[domain_selected["selection_method"] == "elastic_net_nonzero"]
            if not domain_selected.empty
            else domain_selected
        )
        domain_drops = drops[drops["base_domain"] == domain] if not drops.empty else drops
        rows.append(
            {
                "domain": domain,
                "domain_label": DOMAIN_LABELS.get(domain, domain),
                "candidate_count": int(candidate_count),
                "target_inputs_per_fold": int(target_per_fold),
                "selected_rows": int(len(selected_primary)),
                "unique_selected_features": int(selected_primary["feature"].nunique()) if not selected_primary.empty else 0,
                "elastic_net_rows": int(len(elastic_net_selected)),
                "elastic_net_mean_inputs_per_fold": float(len(elastic_net_selected) / args.folds) if args.folds else math.nan,
                "unique_elastic_net_features": int(elastic_net_selected["feature"].nunique()) if not elastic_net_selected.empty else 0,
                "collinearity_drops": int(len(domain_drops)),
                "note": note,
            }
        )
    return rows


def selection_pipeline_summary(
    *,
    args: argparse.Namespace,
    robustness_filter: RobustnessFilter,
    stability_selection: pd.DataFrame,
    validation_stage: dict[str, object],
) -> list[dict[str, object]]:
    stable_count = (
        int(stability_selection["stable"].sum())
        if not stability_selection.empty and "stable" in stability_selection.columns
        else 0
    )
    stability_rows = int(len(stability_selection)) if not stability_selection.empty else 0
    return [
        {
            "step": "IBSI-compliant extraction",
            "status": "upstream",
            "implementation": (
                "Uses the existing aorta radiomics feature table; this script does not re-extract image features. "
                "IBSI compliance must be confirmed in the extraction configuration/provenance."
            ),
        },
        {
            "step": "Test-retest / segmentation / acquisition robustness screening",
            "status": robustness_filter.status,
            "implementation": robustness_filter.details,
        },
        {
            "step": "Missingness and near-zero-variance removal",
            "status": "applied_foldwise",
            "implementation": (
                f"Within each training fold/resample, drops features with missingness > {args.max_missing:g} "
                f"or variance <= {args.min_variance:g}."
            ),
        },
        {
            "step": "Correlation clustering / redundancy filtering",
            "status": "applied_foldwise",
            "implementation": (
                f"Within each training fold/resample, keeps relevance-ranked features and drops later candidates with "
                f"abs({args.correlation_method}) >= {args.correlation_threshold:g}; minimum paired n={args.min_correlation_pair_n}."
            ),
        },
        {
            "step": "Elastic-net model",
            "status": "applied_foldwise",
            "implementation": (
                f"Sparse logistic regression using C={args.elastic_net_c:g}, l1_ratio={args.elastic_net_l1_ratio:g}, "
                "unweighted classes; fitted only on training folds/resamples."
            ),
        },
        {
            "step": "Platt logistic recalibration",
            "status": "applied_cross_fitted",
            "implementation": (
                f"For every out-of-fold probability score, fits outcome ~ logit(score) in {args.folds} stratified "
                "calibration folds and predicts held-out calibrated probabilities."
            ),
        },
        {
            "step": "Stability selection across repeated resamples",
            "status": "applied",
            "implementation": (
                f"{args.stability_resamples} stratified resamples at train fraction {args.stability_train_fraction:g}; "
                f"{stability_rows} nonzero feature rows recorded; {stable_count} met stability probability >= {args.stability_threshold:g}."
            ),
        },
        {
            "step": "External or center-held-out validation",
            "status": validation_stage["status"],
            "implementation": str(validation_stage["details"]),
        },
    ]


def build_summary(
    *,
    analysis_path: Path,
    scores_path: Path,
    performance_path: Path,
    selected_features_path: Path,
    collinearity_path: Path,
    stability_path: Path,
    report_path: Path,
    raw_rows: int,
    scored_rows: int,
    events: int,
    ratio_columns: list[str],
    domains: dict[str, list[str]],
    performance: pd.DataFrame,
    selected_features: pd.DataFrame,
    collinearity_drops: pd.DataFrame,
    stability_selection: pd.DataFrame,
    robustness_filter: RobustnessFilter,
    validation_stage: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    top_scores = performance.head(12).to_dict(orient="records") if not performance.empty else []
    domain_counts = domain_variable_counts(
        domains=domains,
        selected_features=selected_features,
        collinearity_drops=collinearity_drops,
        args=args,
    )
    return {
        "analysis_csv": str(analysis_path),
        "scores_csv": str(scores_path),
        "performance_csv": str(performance_path),
        "selected_features_csv": str(selected_features_path),
        "collinearity_report_csv": str(collinearity_path),
        "stability_selection_csv": str(stability_path),
        "report_html": str(report_path),
        "raw_rows": raw_rows,
        "scored_rows": scored_rows,
        "mace_events": events,
        "mace_nonevents": scored_rows - events,
        "positive_count": events,
        "negative_count": scored_rows - events,
        "outcome_name": REPORT_OUTCOME_NAME,
        "positive_label": REPORT_POSITIVE_LABEL,
        "negative_label": REPORT_NEGATIVE_LABEL,
        "ratio_feature_count": len(ratio_columns),
        "collinearity_dropped_count": int(len(collinearity_drops)),
        "ratio_features": ratio_columns,
        "domain_feature_counts": {domain: len(columns) for domain, columns in sorted(domains.items())},
        "domain_variable_counts": domain_counts,
        "selection_pipeline": selection_pipeline_summary(
            args=args,
            robustness_filter=robustness_filter,
            stability_selection=stability_selection,
            validation_stage=validation_stage,
        ),
        "methods": {
            "ratio_features": "Volume burden features divided by available aorta volume and aortic length denominators; wall-thickness millimeter features are also divided by equivalent aortic radius/diameter derived from aorta volume over length.",
            "domain_signed_z_cv": f"Out-of-fold mean of train-standardized selected features, sign-oriented by train-fold {REPORT_OUTCOME_NAME} SMD; z values are clipped to +/-8 for numerical stability.",
            "domain_probability_cv": "Out-of-fold L2 logistic regression probability using train-fold selected imaging features only; z values are clipped to +/-8 for numerical stability.",
            "domain_elastic_net_probability_cv": (
                "Out-of-fold elastic-net logistic probability. Each fold first screens non-collinear features, then "
                f"elastic-net logistic regression with C={args.elastic_net_c:g} and l1_ratio={args.elastic_net_l1_ratio:g} "
                "shrinks that set to sparse nonzero inputs. Class weights are not balanced, so probabilities retain the training-fold event-rate prior."
            ),
            "platt_recalibration": (
                "Cross-fitted Platt logistic recalibration of each out-of-fold probability score using "
                "outcome ~ logit(raw probability); calibrated probabilities are stored as __platt_cv columns."
            ),
            "domain_sum": "Mean of calcium, fat, wall-from-fat, and wall-thickness out-of-fold domain scores.",
            "collinearity_pruning": (
                f"Within each training fold, features were ranked by {REPORT_OUTCOME_NAME} relevance, then later candidates were dropped "
                f"if abs({args.correlation_method}) correlation with an already selected feature was >= "
                f"{args.correlation_threshold:.2f} using at least {args.min_correlation_pair_n} paired observations."
            ),
            "feature_screen": {
                "top_k_per_domain": args.top_k,
                "all_top_k": args.all_top_k,
                "max_missing": args.max_missing,
                "correlation_threshold": args.correlation_threshold,
                "correlation_method": args.correlation_method,
                "min_correlation_pair_n": args.min_correlation_pair_n,
                "elastic_net_c": args.elastic_net_c,
                "elastic_net_l1_ratio": args.elastic_net_l1_ratio,
                "folds": args.folds,
            },
        },
        "top_scores": top_scores,
    }


def build_report(
    *,
    scores: pd.DataFrame,
    performance: pd.DataFrame,
    selected_features: pd.DataFrame,
    collinearity_drops: pd.DataFrame,
    stability_selection: pd.DataFrame,
    summary: dict[str, object],
    assets_dir: Path,
) -> str:
    score_long = score_long_frame(scores)
    violin_uri = score_violin_plot(score_long, assets_dir / "score_violins.png")
    perf_plot_uri = performance_plot(performance, assets_dir / "score_performance.png")
    event_rate_uri = event_rate_quartile_plot(performance, assets_dir / "event_rate_quartiles.png")
    selected_plot_uri = selected_feature_count_plot(selected_features, assets_dir / "selected_feature_counts.png")
    domain_burden_uri = domain_burden_plot(summary["domain_variable_counts"], assets_dir / "domain_feature_burden.png")
    collinearity_plot_uri = collinearity_drop_plot(collinearity_drops, assets_dir / "collinearity_drops.png")
    stability_plot_uri = stability_selection_plot(stability_selection, assets_dir / "stability_selection.png")
    stable_domain_uri = stable_domain_plot(stability_selection, assets_dir / "stable_features_by_domain.png")
    score_corr_uri = score_correlation_plot(scores, assets_dir / "score_correlation_heatmap.png")
    top_score_name = str(performance.iloc[0]["score_name"]) if not performance.empty else ""
    top_score_label = score_label(top_score_name) if top_score_name else "No valid score"
    calibration_score_name = calibration_score_for_report(performance, top_score_name)
    calibration_score_label = score_label(calibration_score_name) if calibration_score_name else "No valid probability score"
    winner_domain = score_domain(top_score_name) if top_score_name else ""
    winner_calibration = winner_calibration_frame(scores, calibration_score_name)
    winner_calibration_metrics = winner_calibration_summary(scores, calibration_score_name)
    winner_calibration_uri = winner_calibration_plot(
        winner_calibration,
        winner_calibration_metrics,
        calibration_score_label,
        assets_dir / "winner_score_calibration.png",
    )
    winner_feature_plot_uri = winner_feature_plot(
        selected_features,
        top_score_name,
        assets_dir / "winner_score_features.png",
    )
    calibration_note = winner_calibration_note(top_score_name, calibration_score_name)
    winner_feature_title = "Winner domain component inputs" if winner_domain == "domain_sum" else "Winner elastic-net features"
    winner_feature_caption = (
        "Domain input features contributing to the winning summed composite, summarized across outer folds."
        if winner_domain == "domain_sum"
        else "Features with nonzero coefficients in the winning score, summarized across outer folds."
    )
    score_options = score_options_html(performance)
    score_case_tables = score_case_tables_html(scores, performance)
    body = f"""
    <header class="hero">
      <p class="eyebrow">{esc(REPORT_EYEBROW)}</p>
      <h1>{esc(REPORT_TITLE)}</h1>
      <p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}. Scores use aorta imaging/radiomics features only; no age, sex, or clinical variables enter the models.</p>
      {extra_note_html(REPORT_EXTRA_NOTE)}
    </header>
    <section class="stat-strip">
      {stat_card("Scored cases", fmt_int(summary["scored_rows"]))}
      {stat_card(f"{REPORT_POSITIVE_LABEL} positive", fmt_int(summary["positive_count"]))}
      {stat_card("Ratio features", fmt_int(summary["ratio_feature_count"]))}
      {stat_card("Collinear drops", fmt_int(summary["collinearity_dropped_count"]))}
    </section>
    <nav class="tabs" aria-label="Dashboard sections">
      {tab_button("overview", "Overview", active=True)}
      {tab_button("winner", "Winner")}
      {tab_button("scores", "Scores")}
      {tab_button("features", "Features")}
      {tab_button("stability", "Stability")}
      {tab_button("collinearity", "Collinearity")}
      {tab_button("data", "Data")}
      {tab_button("files", "Files")}
    </nav>
    <section class="tab-panel active" id="tab-overview">
      <div class="two-column">
        <div class="narrative">
          <p><span class="newthought">This is the imaging-only score layer.</span> It adds aorta-size ratio features,
          builds out-of-fold domain scores for calcium, periaortic fat, wall-from-fat, and wall thickness, and then tests
          whether the domain scores add when averaged together. Feature selection is relevance-first but collinearity-pruned
          inside each training fold. Each domain uses up to {int(summary["methods"]["feature_screen"]["top_k_per_domain"])} variables per fold before elastic-net; the all-imaging
          model uses up to {int(summary["methods"]["feature_screen"]["all_top_k"])}. Current top score: <strong>{esc(top_score_label)}</strong>.</p>
        </div>
        <div class="summary-box">
          <h2>Model mechanics</h2>
          {pipeline_table_html(summary["selection_pipeline"])}
        </div>
      </div>
      <div class="figure-grid">
        {figure_card("Out-of-fold discrimination", "Directional AUROC for each score. Dashed line marks chance.", perf_plot_uri, "Score AUROC performance")}
        {figure_card("Score quartile event rates", f"Observed {REPORT_OUTCOME_NAME} rate in low-score and high-score quartiles.", event_rate_uri, "Event rates by score quartile")}
        {figure_card("Feature burden by domain", "Candidates, selected inputs, elastic-net features, and correlated drops.", domain_burden_uri, "Feature burden by domain")}
        {figure_card("Score correlation", "Spearman correlation between derived score columns.", score_corr_uri, "Score correlation heatmap")}
      </div>
    </section>
    <section class="tab-panel" id="tab-winner">
      <div class="two-column">
        <div class="narrative">
          <p><span class="newthought">Winner score.</span> The highest out-of-fold discrimination in this run is
          <strong>{esc(top_score_label)}</strong> ({esc(score_type_label(top_score_name))}). The feature panel lists the
          fold-selected inputs behind this score after relevance screening and collinearity pruning. {calibration_note}</p>
        </div>
        <div class="summary-box">
          <h2>Calibration summary</h2>
          {winner_calibration_summary_html(winner_calibration_metrics)}
        </div>
      </div>
      <div class="figure-grid">
        {figure_card("Probability calibration", "Out-of-fold probability bins. Points on the diagonal are ideally calibrated.", winner_calibration_uri, "Winner score calibration plot")}
        {figure_card(winner_feature_title, winner_feature_caption, winner_feature_plot_uri, "Winner score feature contributions")}
      </div>
      <h2>Calibration bins</h2>
      {winner_calibration_table_html(winner_calibration)}
      <h2>Winner score features</h2>
      {winner_feature_table_html(selected_features, stability_selection, top_score_name)}
    </section>
    <section class="tab-panel" id="tab-scores">
      <div class="control-row">
        <label for="score-case-select">Score</label>
        <select id="score-case-select">{score_options}</select>
        <input type="search" data-filter-table="performance-table" placeholder="Filter performance table">
      </div>
      <div class="figure-grid single">
        {figure_card(f"Score distributions by {REPORT_OUTCOME_NAME}", "Seaborn violins use out-of-fold score values only.", violin_uri, f"Score violin plots by {REPORT_OUTCOME_NAME}")}
      </div>
      <h2>Performance</h2>
      {performance_table_html(performance)}
      <h2>Case score tails</h2>
      {score_case_tables}
    </section>
    <section class="tab-panel" id="tab-features">
      <div class="control-row">
        <input type="search" data-filter-table="selected-features-table" placeholder="Filter selected features">
        <input type="search" data-filter-table="domain-table" placeholder="Filter domain counts">
      </div>
      <div class="figure-grid">
        {figure_card("Feature selection frequency", "Unique selected features by domain.", selected_plot_uri, "Selected feature counts by domain")}
        {figure_card("Feature burden by domain", "Fold-aware feature counts after screening and elastic-net.", domain_burden_uri, "Feature burden by domain")}
      </div>
      <h2>Domain counts</h2>
      {domain_variable_count_table_html(summary["domain_variable_counts"])}
      <h2>Selected features</h2>
      {selected_features_table_html(selected_features)}
    </section>
    <section class="tab-panel" id="tab-stability">
      <div class="control-row">
        <input type="search" data-filter-table="stability-table" placeholder="Filter stability rows">
      </div>
      <div class="figure-grid">
        {figure_card("Stability probabilities", "Repeated stratified resamples; stable means selection probability >= threshold.", stability_plot_uri, "Stability selection probabilities")}
        {figure_card("Stable features by domain", "Stable rows compared with all stability rows.", stable_domain_uri, "Stable feature counts by domain")}
      </div>
      {stability_selection_table_html(stability_selection)}
    </section>
    <section class="tab-panel" id="tab-collinearity">
      <div class="control-row">
        <input type="search" data-filter-table="collinearity-table" placeholder="Filter collinearity table">
      </div>
      <div class="figure-grid single">
        {figure_card("Collinearity pruning", f"Dropped candidates were highly correlated with a more {REPORT_OUTCOME_NAME}-relevant retained feature in the same training fold.", collinearity_plot_uri, "Collinear feature drops by domain")}
      </div>
      {collinearity_table_html(collinearity_drops)}
    </section>
    <section class="tab-panel" id="tab-data">
      <div class="control-row">
        <input type="search" data-filter-table="ratio-table" placeholder="Filter ratio feature groups">
      </div>
      <h2>Aorta-size ratio features</h2>
      {ratio_features_table_html(summary["ratio_features"])}
      <h2>Score columns</h2>
      {score_column_table_html(scores)}
    </section>
    <section class="tab-panel" id="tab-files">
      <h2>Generated files</h2>
      <p>Scores CSV: <code>{esc(summary["scores_csv"])}</code><br>
      Performance CSV: <code>{esc(summary["performance_csv"])}</code><br>
      Selected features CSV: <code>{esc(summary["selected_features_csv"])}</code><br>
      Collinearity CSV: <code>{esc(summary["collinearity_report_csv"])}</code><br>
      Stability CSV: <code>{esc(summary["stability_selection_csv"])}</code><br>
      Report HTML: <code>{esc(summary["report_html"])}</code></p>
    </section>
    """
    return html_page(body)


def tab_button(tab_id: str, label: str, *, active: bool = False) -> str:
    selected = "true" if active else "false"
    active_class = " active" if active else ""
    return (
        f'<button class="tab-button{active_class}" type="button" role="tab" '
        f'aria-selected="{selected}" data-tab-target="{esc(tab_id)}">{esc(label)}</button>'
    )


def figure_card(title: str, caption: str, data_uri: str, alt: str) -> str:
    return (
        '<figure class="figure-panel">'
        f"<figcaption><strong>{esc(title)}</strong><span>{esc(caption)}</span></figcaption>"
        f"{embedded_image(data_uri, alt)}"
        "</figure>"
    )


def extra_note_html(note: str) -> str:
    if not note:
        return ""
    return f'<p class="subtitle">{esc(note)}</p>'


def score_options_html(performance: pd.DataFrame) -> str:
    if performance.empty:
        return '<option value="">No valid score</option>'
    options = []
    for _, row in performance.iterrows():
        score_name = str(row["score_name"])
        label = f"{score_label(score_name)} | AUROC {float(row['directional_auc']):.3f}"
        options.append(f'<option value="{esc(score_name)}">{esc(label)}</option>')
    return "".join(options)


def calibration_score_for_report(performance: pd.DataFrame, winner_score_name: str) -> str:
    if not winner_score_name:
        return ""
    if performance.empty or "score_name" not in performance.columns:
        return winner_score_name
    score_names = set(performance["score_name"].astype(str))
    if is_probability_score(winner_score_name) and not is_platt_score(winner_score_name):
        platt_candidate = platt_score_name(winner_score_name)
        if platt_candidate in score_names:
            return platt_candidate
    if is_probability_score(winner_score_name):
        return winner_score_name
    for score_name in performance["score_name"].astype(str):
        if is_platt_score(score_name):
            return score_name
    for score_name in performance["score_name"].astype(str):
        if is_probability_score(score_name):
            return score_name
    return winner_score_name


def is_probability_score(score_name: str) -> bool:
    return "probability" in score_name


def is_platt_score(score_name: str) -> bool:
    return score_name.endswith("__platt_cv")


def platt_score_name(score_name: str) -> str:
    return score_name if is_platt_score(score_name) else f"{score_name}__platt_cv"


def platt_base_score_name(score_name: str) -> str:
    return score_name.removesuffix("__platt_cv")


def score_type_label(score_name: str) -> str:
    if not score_name:
        return "no valid score"
    labels = {
        "cv_elastic_net_probability": "elastic-net probability",
        "cv_elastic_net_platt_probability": "elastic-net Platt probability",
        "cv_logistic_probability": "logistic probability",
        "cv_platt_probability": "Platt probability",
        "cv_signed_zsum": "signed z-score composite",
    }
    return labels.get(score_type(score_name), score_type(score_name))


def winner_calibration_note(winner_score_name: str, calibration_score_name: str) -> str:
    if not calibration_score_name:
        return "No probability calibration score was available."
    if calibration_score_name == winner_score_name:
        return (
            "The calibration plot uses the winner itself because it is an out-of-fold probability score, "
            f"and compares predicted probability with observed {esc(REPORT_OUTCOME_NAME)} rate across decile-like bins."
        )
    if is_platt_score(calibration_score_name) and platt_base_score_name(calibration_score_name) == winner_score_name:
        return (
            "The calibration plot uses the cross-fitted Platt recalibration of the winner, fitted as "
            "outcome ~ logit(raw out-of-fold probability) on training calibration folds."
        )
    return (
        f"The winner is not a probability, so true calibration uses the best-ranked probability score: "
        f"<strong>{esc(score_label(calibration_score_name))}</strong>. The winner remains the score ranked first by discrimination."
    )


def score_case_tables_html(scores: pd.DataFrame, performance: pd.DataFrame) -> str:
    if scores.empty or performance.empty:
        return '<p class="caption">No case-level score table is available.</p>'
    panels = []
    for index, row in performance.iterrows():
        score_name = str(row["score_name"])
        if score_name not in scores.columns:
            continue
        values = pd.to_numeric(scores[score_name], errors="coerce")
        data = scores[[CASE_ID, OUTCOME, "study_arm", score_name]].copy()
        data["_score"] = values
        data = data.dropna(subset=["_score"])
        if data.empty:
            continue
        low = data.nsmallest(12, "_score").assign(tail="lowest")
        high = data.nlargest(12, "_score").assign(tail="highest")
        tails = pd.concat([high, low], ignore_index=True)
        rows = []
        for _, case in tails.iterrows():
            rows.append(
                [
                    esc(case["tail"]),
                    esc(case[CASE_ID]),
                    REPORT_POSITIVE_LABEL if str(case[OUTCOME]) == "1" else REPORT_NEGATIVE_LABEL,
                    esc(str(case.get("study_arm", ""))),
                    f"{float(case['_score']):.4f}",
                ]
            )
        panel_class = "score-case-panel active" if index == 0 else "score-case-panel"
        panels.append(
            f'<div class="{panel_class}" data-score-panel="{esc(score_name)}">'
            f"<h3>{esc(score_label(score_name))}</h3>"
            + table_html(
                ["Tail", "Case", "Outcome", CASE_GROUP_LABEL, "Score"],
                rows,
                raw=True,
                numeric_columns={4},
                table_id=f"score-cases-{index}",
                classes="data-table sortable compact-table",
            )
            + "</div>"
        )
    return "".join(panels) if panels else '<p class="caption">No case-level score table is available.</p>'


def winner_calibration_frame(scores: pd.DataFrame, score_name: str, bins: int = 10) -> pd.DataFrame:
    columns = [
        "bin",
        "n",
        "events",
        "mean_predicted_probability",
        "observed_event_rate",
        "predicted_min",
        "predicted_max",
        "absolute_error",
        "observed_expected_ratio",
    ]
    if not score_name or score_name not in scores.columns or OUTCOME not in scores.columns:
        return pd.DataFrame(columns=columns)
    probabilities = pd.to_numeric(scores[score_name], errors="coerce").clip(1e-6, 1 - 1e-6)
    outcome = pd.to_numeric(scores[OUTCOME], errors="coerce")
    valid = probabilities.notna() & outcome.isin([0, 1])
    if valid.sum() < 4 or outcome.loc[valid].nunique() < 2:
        return pd.DataFrame(columns=columns)
    data = pd.DataFrame({"probability": probabilities.loc[valid], "outcome": outcome.loc[valid].astype(int)})
    q = min(bins, len(data))
    data["bin_id"] = pd.qcut(data["probability"].rank(method="first"), q=q, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in data.groupby("bin_id", sort=True):
        n = int(len(group))
        events = int(group["outcome"].sum())
        predicted_mean = float(group["probability"].mean())
        observed = float(group["outcome"].mean())
        rows.append(
            {
                "bin": int(bin_id) + 1,
                "n": n,
                "events": events,
                "mean_predicted_probability": predicted_mean,
                "observed_event_rate": observed,
                "predicted_min": float(group["probability"].min()),
                "predicted_max": float(group["probability"].max()),
                "absolute_error": abs(observed - predicted_mean),
                "observed_expected_ratio": observed / predicted_mean if predicted_mean > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def winner_calibration_summary(scores: pd.DataFrame, score_name: str) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "score_name": score_name,
        "n": 0,
        "events": 0,
        "observed_event_rate": math.nan,
        "mean_predicted_probability": math.nan,
        "brier_score": math.nan,
        "expected_calibration_error": math.nan,
        "calibration_intercept": math.nan,
        "calibration_slope": math.nan,
    }
    if not score_name or score_name not in scores.columns or OUTCOME not in scores.columns:
        return summary
    probabilities = pd.to_numeric(scores[score_name], errors="coerce").clip(1e-6, 1 - 1e-6)
    outcome = pd.to_numeric(scores[OUTCOME], errors="coerce")
    valid = probabilities.notna() & outcome.isin([0, 1])
    if valid.sum() == 0:
        return summary
    p = probabilities.loc[valid].to_numpy(dtype=float)
    y = outcome.loc[valid].astype(int).to_numpy()
    summary["n"] = int(len(y))
    summary["events"] = int(y.sum())
    summary["observed_event_rate"] = float(y.mean())
    summary["mean_predicted_probability"] = float(p.mean())
    summary["brier_score"] = float(brier_score_loss(y, p))
    calibration = winner_calibration_frame(scores, score_name)
    if not calibration.empty:
        weights = calibration["n"].to_numpy(dtype=float)
        errors = calibration["absolute_error"].to_numpy(dtype=float)
        summary["expected_calibration_error"] = float(np.sum(weights * errors) / np.sum(weights))
    if len(np.unique(y)) >= 2:
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        try:
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            model.fit(logit, y)
            summary["calibration_intercept"] = float(model.intercept_[0])
            summary["calibration_slope"] = float(model.coef_.ravel()[0])
        except Exception:
            pass
    return summary


def winner_calibration_summary_html(metrics: dict[str, float | int | str]) -> str:
    rows = [
        ["Calibration score", esc(score_label(str(metrics.get("score_name", ""))))],
        ["Rows", fmt_int(metrics.get("n", 0))],
        ["Events", fmt_int(metrics.get("events", 0))],
        ["Observed event rate", fmt_float(metrics.get("observed_event_rate"), digits=3)],
        ["Mean predicted probability", fmt_float(metrics.get("mean_predicted_probability"), digits=3)],
        ["Brier score", fmt_float(metrics.get("brier_score"), digits=3)],
        ["Expected calibration error", fmt_float(metrics.get("expected_calibration_error"), digits=3)],
        ["Calibration intercept", fmt_float(metrics.get("calibration_intercept"), digits=3)],
        ["Calibration slope", fmt_float(metrics.get("calibration_slope"), digits=3)],
    ]
    return table_html(["Metric", "Value"], rows, raw=True, table_id="winner-calibration-summary", classes="data-table compact-table")


def winner_calibration_table_html(calibration: pd.DataFrame) -> str:
    if calibration.empty:
        return '<p class="caption">Calibration bins could not be computed for the probability score.</p>'
    rows = []
    for _, row in calibration.iterrows():
        rows.append(
            [
                fmt_int(row["bin"]),
                fmt_int(row["n"]),
                fmt_int(row["events"]),
                fmt_float(row["mean_predicted_probability"], digits=3),
                fmt_float(row["observed_event_rate"], digits=3),
                fmt_float(row["predicted_min"], digits=3),
                fmt_float(row["predicted_max"], digits=3),
                fmt_float(row["absolute_error"], digits=3),
                fmt_float(row["observed_expected_ratio"], digits=2),
            ]
        )
    return table_html(
        [
            "Bin",
            "n",
            "Events",
            "Mean predicted",
            "Observed rate",
            "Pred min",
            "Pred max",
            "Abs error",
            "O/E",
        ],
        rows,
        raw=True,
        numeric_columns={0, 1, 2, 3, 4, 5, 6, 7, 8},
        table_id="winner-calibration-table",
    )


def winner_calibration_plot(
    calibration: pd.DataFrame,
    metrics: dict[str, float | int | str],
    score_name: str,
    output_path: Path,
) -> str:
    if calibration.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    max_axis = max(
        0.36,
        float(calibration["mean_predicted_probability"].max()) + 0.04,
        float(calibration["observed_event_rate"].max()) + 0.04,
    )
    ax.plot([0, max_axis], [0, max_axis], color="#c7bea9", linestyle=(0, (2, 4)), linewidth=1.1)
    sizes = np.clip(calibration["n"].to_numpy(dtype=float), 8, None) * 9.0
    ax.scatter(
        calibration["mean_predicted_probability"],
        calibration["observed_event_rate"],
        s=sizes,
        color="#335f86",
        alpha=0.86,
        edgecolor="#151515",
        linewidth=0.4,
    )
    for _, row in calibration.iterrows():
        ax.text(
            float(row["mean_predicted_probability"]),
            float(row["observed_event_rate"]) + 0.012,
            str(int(row["bin"])),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#151515",
        )
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel(f"Observed {REPORT_OUTCOME_NAME} rate")
    ax.set_title(score_name, fontweight="normal", fontsize=10)
    brier = metrics.get("brier_score", math.nan)
    ece = metrics.get("expected_calibration_error", math.nan)
    ax.text(
        0.02,
        0.98,
        f"Brier {fmt_float(brier, digits=3)}\nECE {fmt_float(ece, digits=3)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#666666",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def winner_feature_summary(
    selected_features: pd.DataFrame,
    stability_selection: pd.DataFrame,
    winner_score_name: str,
) -> pd.DataFrame:
    columns = [
        "feature",
        "feature_label",
        "feature_description",
        "feature_domain",
        "selected_rows",
        "mean_rank",
        "mean_coefficient",
        "mean_abs_coefficient",
        "positive_rows",
        "negative_rows",
        "selection_probability",
        "stable",
    ]
    if selected_features.empty or not winner_score_name:
        return pd.DataFrame(columns=columns)
    winner_domains = winner_feature_domains(winner_score_name)
    winner_method = winner_feature_selection_method(winner_score_name)
    if not winner_domains:
        return pd.DataFrame(columns=columns)
    data = selected_features.copy()
    data = data[
        data["base_domain"].isin(winner_domains)
        & data["selection_method"].eq(winner_method)
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        data.groupby(["feature", "feature_label", "feature_description"], dropna=False)
        .agg(
            selected_rows=("feature", "size"),
            mean_rank=("rank", "mean"),
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("coefficient", mean_abs_or_nan),
            positive_rows=("coefficient", lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())),
            negative_rows=("coefficient", lambda values: int((pd.to_numeric(values, errors="coerce") < 0).sum())),
        )
        .reset_index()
    )
    grouped["feature_domain"] = grouped["feature"].map(lambda value: DOMAIN_LABELS.get(feature_domain(value), "Other"))
    if not stability_selection.empty:
        stability = stability_selection[
            stability_selection["domain"].isin(winner_domains)
        ][["feature", "selection_probability", "stable"]]
        grouped = grouped.merge(stability, on="feature", how="left")
    else:
        grouped["selection_probability"] = math.nan
        grouped["stable"] = False
    grouped["stable"] = grouped["stable"].map(lambda value: bool(value) if pd.notna(value) else False)
    grouped = grouped.sort_values(
        ["selected_rows", "selection_probability", "mean_abs_coefficient"],
        ascending=[False, False, False],
    )
    return grouped[columns]


def winner_feature_table_html(
    selected_features: pd.DataFrame,
    stability_selection: pd.DataFrame,
    winner_score_name: str,
) -> str:
    summary = winner_feature_summary(selected_features, stability_selection, winner_score_name)
    if summary.empty:
        return '<p class="caption">No feature rows were available for the winner score.</p>'
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                esc(row["feature_domain"]),
                esc(row["feature_label"]),
                esc(row["feature_description"]),
                fmt_int(row["selected_rows"]),
                fmt_float(row["mean_rank"], digits=1),
                fmt_float(row["mean_coefficient"], digits=3),
                fmt_float(row["mean_abs_coefficient"], digits=3),
                fmt_int(row["positive_rows"]),
                fmt_int(row["negative_rows"]),
                fmt_float(row["selection_probability"], digits=2),
                "yes" if bool(row["stable"]) else "",
                esc(row["feature"]),
            ]
        )
    return table_html(
        [
            "Domain",
            "Feature",
            "Description",
            "Rows",
            "Mean rank",
            "Mean coef",
            "Abs coef",
            "Positive",
            "Negative",
            "Stability",
            "Stable",
            "Raw feature name",
        ],
        rows,
        raw=True,
        numeric_columns={3, 4, 5, 6, 7, 8, 9},
        table_id="winner-features-table",
    )


def winner_feature_plot(selected_features: pd.DataFrame, winner_score_name: str, output_path: Path) -> str:
    summary = winner_feature_summary(selected_features, pd.DataFrame(), winner_score_name)
    if summary.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    data = summary.head(18).sort_values(["selected_rows", "mean_abs_coefficient"], ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, max(4.0, 0.34 * len(data))))
    color_lookup = {
        "Calcium": DOMAIN_COLORS["calcium"],
        "Periaortic fat": DOMAIN_COLORS["fat"],
        "Wall-from-fat": DOMAIN_COLORS["wall_from_fat"],
        "Wall thickness": DOMAIN_COLORS["wall_thickness"],
    }
    colors = [color_lookup.get(domain, "#555555") for domain in data["feature_domain"]]
    coefficients = pd.to_numeric(data["mean_coefficient"], errors="coerce")
    if coefficients.notna().any():
        bar_values = coefficients.fillna(0).to_numpy(dtype=float)
        xlabel = "Mean elastic-net coefficient across selected folds"
        ax.axvline(0, color="#c7bea9", linewidth=1)
    else:
        bar_values = data["selected_rows"].to_numpy(dtype=float)
        xlabel = "Selected fold-level input rows"
    ax.barh(data["feature_label"], bar_values, color=colors, height=0.62)
    for y_pos, row in enumerate(data.itertuples(index=False)):
        x = float(bar_values[y_pos])
        offset = 0.01 if x >= 0 else -0.01
        ha = "left" if x >= 0 else "right"
        ax.text(x + offset, y_pos, f"n={int(row.selected_rows)}", va="center", ha=ha, fontsize=7.5)
    if coefficients.notna().any():
        limit = max(0.12, float(np.nanmax(np.abs(bar_values))) + 0.08)
        ax.set_xlim(-limit, limit)
    else:
        ax.set_xlim(0, max(5.0, float(np.nanmax(bar_values)) + 1.0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def winner_feature_domains(score_name: str) -> list[str]:
    if not score_name:
        return []
    domain = score_domain(score_name)
    if domain == "domain_sum":
        return ["calcium", "fat", "wall_from_fat", "wall_thickness"]
    return [domain]


def winner_feature_selection_method(score_name: str) -> str:
    if "elastic_net" in score_name:
        return "elastic_net_nonzero"
    return "relevance_collinearity"


def mean_abs_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return math.nan
    return float(np.nanmean(np.abs(numeric)))


def score_columns(scores: pd.DataFrame) -> list[str]:
    return [
        column
        for column in scores.columns
        if column.endswith("__signed_z_cv")
        or column.endswith("__probability_cv")
        or column.endswith("__elastic_net_probability_cv")
        or column.endswith("__platt_cv")
        or column == "domain_sum__probability_mean_cv"
        or column == "domain_sum__elastic_net_probability_mean_cv"
    ]


def score_long_frame(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in score_columns(scores):
        values = pd.to_numeric(scores[column], errors="coerce")
        for case_id, outcome, value in zip(scores[CASE_ID], scores[OUTCOME], values):
            if pd.isna(value) or outcome not in {"0", "1", 0, 1}:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "outcome": REPORT_POSITIVE_LABEL if str(outcome) == "1" else REPORT_NEGATIVE_LABEL,
                    "score": float(value),
                    "score_name": score_label(column),
                    "domain": score_domain(column),
                    "score_type": score_type(column),
                }
            )
    return pd.DataFrame(rows)


def score_violin_plot(frame: pd.DataFrame, output_path: Path) -> str:
    if frame.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    order = [REPORT_NEGATIVE_LABEL, REPORT_POSITIVE_LABEL]
    palette = {REPORT_NEGATIVE_LABEL: "#b9ae98", REPORT_POSITIVE_LABEL: "#335f86"}
    score_order = sorted(frame["score_name"].unique())
    grid = sns.FacetGrid(
        frame,
        col="score_name",
        col_wrap=3,
        sharey=False,
        height=2.55,
        aspect=1.18,
        col_order=score_order,
        despine=True,
    )

    def draw(data: pd.DataFrame, **_: object) -> None:
        ax = plt.gca()
        sns.violinplot(
            data=data,
            x="outcome",
            y="score",
            hue="outcome",
            order=order,
            hue_order=order,
            palette=palette,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            saturation=0.9,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="outcome",
            y="score",
            order=order,
            color="#111111",
            alpha=0.20,
            size=1.8,
            jitter=0.22,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    grid.map_dataframe(draw)
    grid.set_titles("{col_name}", size=8.6)
    for ax in grid.axes.flat:
        ax.grid(False)
        ax.spines["left"].set_color("#d8d0bc")
        ax.spines["bottom"].set_color("#d8d0bc")
    grid.fig.subplots_adjust(top=0.96, hspace=0.48, wspace=0.22)
    return save_figure_data_uri(grid.fig, output_path)


def performance_plot(performance: pd.DataFrame, output_path: Path) -> str:
    if performance.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    plot_data = performance.copy().sort_values("directional_auc", ascending=True)
    plot_data["label"] = plot_data["score_name"].map(score_label)
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.38 * len(plot_data))))
    colors = [DOMAIN_COLORS.get(domain, "#555") for domain in plot_data["domain"]]
    ax.barh(plot_data["label"], plot_data["directional_auc"], color=colors, height=0.62)
    ax.axvline(0.5, color="#c7bea9", linestyle=(0, (2, 4)), linewidth=1)
    ax.set_xlim(0.45, max(0.7, float(plot_data["directional_auc"].max()) + 0.03))
    ax.set_xlabel("Directional AUROC")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def event_rate_quartile_plot(performance: pd.DataFrame, output_path: Path) -> str:
    if performance.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    data = performance.copy().sort_values("directional_auc", ascending=False).head(12)
    data["label"] = data["score_name"].map(score_label)
    long = []
    for _, row in data.iterrows():
        long.append(
            {
                "label": row["label"],
                "quartile": "Low score",
                "event_rate": float(row["event_rate_low_quartile"]),
            }
        )
        long.append(
            {
                "label": row["label"],
                "quartile": "High score",
                "event_rate": float(row["event_rate_high_quartile"]),
            }
        )
    plot_data = pd.DataFrame(long)
    fig, ax = plt.subplots(figsize=(8.8, max(3.4, 0.34 * len(data))))
    sns.barplot(
        data=plot_data,
        y="label",
        x="event_rate",
        hue="quartile",
        palette={"Low score": "#b9ae98", "High score": "#335f86"},
        ax=ax,
    )
    ax.set_xlabel(f"Observed {REPORT_OUTCOME_NAME} rate")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.42, float(plot_data["event_rate"].max()) + 0.06))
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def selected_feature_count_plot(selected_features: pd.DataFrame, output_path: Path) -> str:
    if selected_features.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    data = selected_features.copy()
    data["base_domain"] = data["domain"].str.split(":").str[0]
    if "selection_method" in data.columns:
        data = data[data["selection_method"] == "relevance_collinearity"]
    if data.empty:
        return ""
    counts = data.groupby("base_domain")["feature"].nunique().reset_index(name="unique_selected_features")
    counts = counts.sort_values("unique_selected_features", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = [DOMAIN_COLORS.get(domain, "#555") for domain in counts["base_domain"]]
    ax.bar(counts["base_domain"].map(lambda value: DOMAIN_LABELS.get(value, value)), counts["unique_selected_features"], color=colors)
    ax.set_ylabel("Unique selected features")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def domain_burden_plot(domain_counts: list[dict[str, object]], output_path: Path) -> str:
    if not domain_counts:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    frame = pd.DataFrame(domain_counts)
    frame = frame[frame["domain"].ne("domain_sum")].copy()
    if frame.empty:
        return ""
    metrics = [
        ("candidate_count", "Candidates"),
        ("unique_selected_features", "Screened unique"),
        ("unique_elastic_net_features", "Elastic-net unique"),
        ("collinearity_drops", "Corr drops"),
    ]
    rows = []
    for _, row in frame.iterrows():
        for column, label in metrics:
            rows.append(
                {
                    "domain": row["domain_label"],
                    "metric": label,
                    "count": float(row[column]),
                    "base_domain": row["domain"],
                }
            )
    plot_data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    sns.scatterplot(
        data=plot_data,
        x="count",
        y="domain",
        hue="metric",
        style="metric",
        s=90,
        palette=["#333333", "#335f86", "#8c3d3d", "#8a5a2b"],
        ax=ax,
    )
    for domain, group in plot_data.groupby("domain"):
        ax.hlines(domain, xmin=0, xmax=float(group["count"].max()), color="#d8d0bc", linewidth=0.8, zorder=0)
    ax.set_xlabel("Feature rows or counts")
    ax.set_ylabel("")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def collinearity_drop_plot(collinearity_drops: pd.DataFrame, output_path: Path) -> str:
    if collinearity_drops.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    data = collinearity_drops.copy()
    data["base_domain"] = data["domain"].str.split(":").str[0]
    counts = data.groupby("base_domain").size().reset_index(name="dropped_features")
    counts = counts.sort_values("dropped_features", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = [DOMAIN_COLORS.get(domain, "#555") for domain in counts["base_domain"]]
    ax.bar(counts["base_domain"].map(lambda value: DOMAIN_LABELS.get(value, value)), counts["dropped_features"], color=colors)
    ax.set_ylabel("Fold-level correlated drops")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def stable_domain_plot(stability_selection: pd.DataFrame, output_path: Path) -> str:
    if stability_selection.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    data = stability_selection.copy()
    totals = data.groupby(["domain", "domain_label"], dropna=False).size().reset_index(name="all_rows")
    stable = (
        data[data["stable"].astype(bool)]
        .groupby(["domain", "domain_label"], dropna=False)
        .size()
        .reset_index(name="stable_rows")
    )
    plot_data = totals.merge(stable, on=["domain", "domain_label"], how="left").fillna({"stable_rows": 0})
    plot_data = plot_data.sort_values("stable_rows", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = [DOMAIN_COLORS.get(domain, "#555") for domain in plot_data["domain"]]
    ax.barh(plot_data["domain_label"], plot_data["all_rows"], color="#e5decb", label="All nonzero rows", height=0.68)
    ax.barh(plot_data["domain_label"], plot_data["stable_rows"], color=colors, label="Stable rows", height=0.68)
    ax.set_xlabel("Stability-selection feature rows")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def score_correlation_plot(scores: pd.DataFrame, output_path: Path) -> str:
    columns = score_columns(scores)
    if len(columns) < 2:
        return ""
    values = scores[columns].apply(pd.to_numeric, errors="coerce")
    corr = values.corr(method="spearman")
    if corr.empty:
        return ""
    labels = [score_label(column) for column in corr.columns]
    corr.index = labels
    corr.columns = labels
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.25,
        linecolor="#fffff8",
        square=False,
        cbar_kws={"label": "Spearman rho", "shrink": 0.72},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7.3)
    ax.tick_params(axis="y", labelsize=7.3)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def stability_selection_plot(stability_selection: pd.DataFrame, output_path: Path) -> str:
    if stability_selection.empty:
        return ""
    sns.set_theme(style="white", context="paper", font="serif", rc=plot_rc())
    data = stability_selection.copy()
    data = data.sort_values(["domain", "selection_probability", "mean_abs_coefficient"], ascending=[True, False, False])
    data = data.groupby("domain", group_keys=False).head(6)
    data["label"] = data["feature_label"]
    fig, ax = plt.subplots(figsize=(8.8, max(3.4, 0.28 * len(data))))
    colors = [DOMAIN_COLORS.get(domain, "#555") for domain in data["domain"]]
    ax.barh(data["domain_label"] + " - " + data["label"], data["selection_probability"], color=colors, height=0.62)
    ax.axvline(0.5, color="#c7bea9", linestyle=(0, (2, 4)), linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection probability")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure_data_uri(fig, output_path)


def enrich_selected_features(selected_features: pd.DataFrame) -> pd.DataFrame:
    if selected_features.empty:
        return selected_features
    enriched = selected_features.copy()
    enriched["base_domain"] = enriched["domain"].str.split(":").str[0]
    enriched["domain_label"] = enriched["base_domain"].map(lambda value: DOMAIN_LABELS.get(value, value))
    enriched["feature_label"] = enriched["feature"].map(feature_label)
    enriched["feature_description"] = enriched["feature"].map(feature_description)
    enriched["selection_method_label"] = enriched["selection_method"].map(selection_method_label)
    return enriched


def enrich_collinearity_drops(collinearity_drops: pd.DataFrame) -> pd.DataFrame:
    if collinearity_drops.empty:
        return collinearity_drops
    enriched = collinearity_drops.copy()
    enriched["base_domain"] = enriched["domain"].str.split(":").str[0]
    enriched["domain_label"] = enriched["base_domain"].map(lambda value: DOMAIN_LABELS.get(value, value))
    enriched["feature_label"] = enriched["feature"].map(feature_label)
    enriched["correlated_with_label"] = enriched["correlated_with"].map(feature_label)
    enriched["feature_description"] = enriched["feature"].map(feature_description)
    enriched["correlated_with_description"] = enriched["correlated_with"].map(feature_description)
    return enriched


def save_figure_data_uri(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="#fffff8")
    plt.close(fig)
    image_bytes = buffer.getvalue()
    output_path.write_bytes(image_bytes)
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def plot_rc() -> dict[str, str]:
    return {
        "figure.facecolor": "#fffff8",
        "axes.facecolor": "#fffff8",
        "axes.edgecolor": "#d8d0bc",
        "axes.labelcolor": "#151515",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
    }


def score_label(column: str) -> str:
    if is_platt_score(column):
        return f"{score_label(platt_base_score_name(column))} Platt"
    if column == "domain_sum__signed_z_cv":
        return "Domain sum signed z"
    if column == "domain_sum__probability_mean_cv":
        return "Domain sum probability"
    if column == "domain_sum__elastic_net_probability_mean_cv":
        return "Domain sum elastic-net"
    domain = score_domain(column)
    if "elastic_net" in column:
        kind = "elastic-net"
    elif "probability" in column:
        kind = "probability"
    else:
        kind = "signed z"
    return f"{DOMAIN_LABELS.get(domain, domain)} {kind}"


def feature_label(feature: str) -> str:
    suffix = ""
    base = feature
    if feature.startswith("ratio_aorta_size__"):
        base = feature.removeprefix("ratio_aorta_size__")
        suffix_map = [
            ("__per_equiv_aortic_radius_mm", " / aortic radius"),
            ("__per_equiv_aortic_diameter_mm", " / aortic diameter"),
            ("__per_aortic_length_cm", " / aortic cm"),
            ("__per_aorta_volume", " / aorta volume"),
            ("__per_wall_volume", " / wall volume"),
        ]
        for ending, label_suffix in suffix_map:
            if base.endswith(ending):
                base = base[: -len(ending)]
                suffix = label_suffix
                break
    lower = base.lower()

    if "calcium_per_cm" in lower:
        label = "Calcium burden per cm"
    elif "calcium_mass_proxy_per_cm" in lower:
        label = "Calcium mass per cm"
    elif "agatston" in lower:
        label = "Agatston-like calcium"
    elif "circumferential_arc_mean" in lower:
        label = "Mean calcium arc"
    elif "circumferential_arc_max" in lower:
        label = "Max calcium arc"
    elif "log1p_num_lesions" in lower or "num_lesions" in lower:
        label = "Calcium lesion count"
    elif "hu_gt_1000_fraction" in lower:
        label = "Very dense calcium fraction"
    elif "hu_gt_1000_volume" in lower:
        label = "Very dense calcium volume"
    elif "calcium_mean_hu" in lower:
        label = "Mean calcium HU"
    elif "calcium_max_hu" in lower:
        label = "Max calcium HU"
    elif "calcium_volume" in lower:
        label = "Calcium volume"
    elif "top_bottom_distance" in lower:
        label = "Calcium span"
    elif "external_contrast_touching_rejected_voxel_count" in lower:
        label = "Rejected external contrast voxels"
    elif "high_confidence_seed_voxel_count" in lower:
        label = "High-confidence calcium seeds"
    elif "dynamic_threshold_hu_min" in lower:
        label = "Dynamic calcium threshold min"
    elif "dynamic_threshold_hu_median" in lower:
        label = "Dynamic calcium threshold median"
    elif "dynamic_threshold_hu_max" in lower:
        label = "Dynamic calcium threshold max"
    elif "graylevelnonuniformitynormalized" in lower:
        label = "Fat texture nonuniformity"
    elif "smalldependencehighgraylevelemphasis" in lower:
        label = "Fat fine high-HU texture"
    elif "10percentile" in lower:
        label = "Fat HU 10th percentile"
    elif "flatness" in lower:
        label = "Fat shape flatness"
    elif "skewness" in lower:
        label = "Fat HU skewness"
    elif "highgraylevelzoneemphasis" in lower:
        label = "Fat high-HU zones"
    elif "largedependencehighgraylevelemphasis" in lower:
        label = "Fat coarse high-HU texture"
    elif "largeareahighgraylevelemphasis" in lower:
        label = "Fat large high-HU zones"
    elif "runentropy" in lower:
        label = "Fat run entropy"
    elif "kurtosis" in lower:
        label = "Fat HU kurtosis"
    elif "mean_hu_0_2mm" in lower:
        label = "Near-wall fat mean HU"
    elif "periaortic_fat_volume" in lower:
        label = "Periaortic fat volume"
    elif "hu_refined_aorta_added_volume" in lower:
        label = "HU-refined added aorta volume"
    elif "fat_support_0_5mm_volume" in lower:
        label = "Fat-supported wall volume"
    elif "contrast_lumen_volume" in lower:
        label = "Contrast lumen volume"
    elif "closed_outer_envelope_volume" in lower:
        label = "Outer envelope volume"
    elif "wall_candidate_mean_hu" in lower:
        label = "Candidate wall mean HU"
    elif "wall_candidate_volume" in lower:
        label = "Candidate wall volume"
    elif "gt4mm_volume" in lower or "wall_thickness_gt4mm_volume" in lower:
        label = "Wall >4 mm volume"
    elif "wall_volume" in lower:
        label = "Wall volume"
    elif "outer_surface" in lower:
        label = "Outer surface " + statistic_label(lower)
    elif "wall_thickness" in lower or lower.startswith("aortic_wall_wall_thickness"):
        label = "Wall thickness " + statistic_label(lower)
    else:
        label = fallback_feature_label(base)
    return label + suffix


def feature_description(feature: str) -> str:
    domain = feature_domain(feature)
    if feature.startswith("ratio_aorta_size__"):
        prefix = "Aorta-size-normalized "
    else:
        prefix = ""
    if domain == "calcium":
        return prefix + "calcification burden, density, or distribution metric."
    if domain == "fat":
        return prefix + "periaortic fat volume, HU, or texture metric."
    if domain == "wall_from_fat":
        return prefix + "experimental wall or lumen metric derived from fat/lumen support."
    if domain == "wall_thickness":
        return prefix + "aortic wall thickness or thick-wall burden metric."
    return prefix + "radiomics feature."


def selection_method_label(value: str) -> str:
    if value == "elastic_net_nonzero":
        return "Elastic-net nonzero"
    if value == "relevance_collinearity":
        return "Screened input"
    return value.replace("_", " ")


def statistic_label(lower: str) -> str:
    stats = [
        ("p05", "p05"),
        ("p25", "p25"),
        ("p75", "p75"),
        ("p95", "p95"),
        ("mean", "mean"),
        ("median", "median"),
        ("max", "max"),
        ("min", "min"),
        ("std", "SD"),
    ]
    for token, label in stats:
        if token in lower:
            return label
    return "metric"


def fallback_feature_label(feature: str) -> str:
    text = feature
    prefixes = [
        "aorta__calcium_omics__",
        "aorta_wall_band__calcification__",
        "aorta_wall_dynamic__calcification__",
        "aorta_wall_dynamic__calcification_dynamic_threshold__",
        "aorta_wall_from_fat__experimental_wall_from_fat_lumen__",
        "aortic_wall__wall_thickness_threshold__",
        "aortic_wall__wall_thickness__",
        "aorta_segment:whole_aorta__fat_omics__",
        "periaortic_fat__fat_omics__",
        "periaortic_fat__radiomics_firstorder__original_firstorder_",
        "periaortic_fat__radiomics_glrlm__original_glrlm_",
        "periaortic_fat__radiomics_glszm__original_glszm_",
        "periaortic_fat__radiomics_gldm__original_gldm_",
        "periaortic_fat__radiomics_shape__original_shape_",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("__thr_dynamic_lumen_referenced_seed500hu", "")
    text = text.replace("__thr_dynamic_lumen_referenced_seed500HU", "")
    text = text.replace("__thr_", " ")
    text = text.replace("_", " ")
    return " ".join(word.capitalize() if word.islower() else word for word in text.split())


def short_feature_name(feature: str, max_len: int = 72) -> str:
    text = feature_label(feature)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def performance_table_html(performance: pd.DataFrame) -> str:
    rows = []
    for _, row in performance.iterrows():
        lift = float(row["event_rate_high_quartile"]) - float(row["event_rate_low_quartile"])
        rows.append(
            [
                esc(score_label(str(row["score_name"]))),
                esc(DOMAIN_LABELS.get(str(row["domain"]), str(row["domain"]))),
                esc(str(row["score_type"])),
                f"{float(row['directional_auc']):.3f}",
                f"{float(row['average_precision']):.3f}",
                f"{float(row['event_rate_low_quartile']):.3f}",
                f"{float(row['event_rate_high_quartile']):.3f}",
                f"{lift:.3f}",
            ]
        )
    return table_html(
        [
            "Score",
            "Domain",
            "Type",
            "Directional AUROC",
            "Avg precision",
            "Low-Q event rate",
            "High-Q event rate",
            "Q4-Q1 lift",
        ],
        rows,
        raw=True,
        numeric_columns={3, 4, 5, 6, 7},
        table_id="performance-table",
    )


def pipeline_table_html(pipeline: list[dict[str, object]]) -> str:
    rows = []
    for row in pipeline:
        rows.append(
            [
                esc(row["step"]),
                esc(row["status"]),
                esc(row["implementation"]),
            ]
        )
    return table_html(["Step", "Status", "Implementation"], rows, raw=True, table_id="pipeline-table")


def domain_variable_count_table_html(domain_counts: list[dict[str, object]]) -> str:
    rows = []
    for row in domain_counts:
        rows.append(
            [
                esc(row["domain_label"]),
                fmt_int(row["candidate_count"]),
                fmt_int(row["target_inputs_per_fold"]),
                fmt_int(row["unique_selected_features"]),
                f"{float(row['elastic_net_mean_inputs_per_fold']):.1f}",
                fmt_int(row["unique_elastic_net_features"]),
                fmt_int(row["collinearity_drops"]),
                esc(row["note"]),
            ]
        )
    return table_html(
        [
            "Domain",
            "Candidates",
            "Inputs/fold",
            "Unique screened",
            "Elastic-net/fold",
            "Unique elastic-net",
            "Corr drops",
            "Meaning",
        ],
        rows,
        raw=True,
        numeric_columns={1, 2, 3, 4, 5, 6},
        table_id="domain-table",
    )


def stability_selection_table_html(stability_selection: pd.DataFrame) -> str:
    if stability_selection.empty:
        return '<p class="caption">No stability-selection rows were generated.</p>'
    data = stability_selection.copy()
    data = data.sort_values(
        ["stable", "domain", "selection_probability", "mean_abs_coefficient"],
        ascending=[False, True, False, False],
    )
    rows = []
    for _, row in data.iterrows():
        rows.append(
            [
                esc(str(row["domain_label"])),
                esc(str(row["feature_label"])),
                esc(str(row["feature_description"])),
                esc(str(row["feature"])),
                f"{float(row['selection_probability']):.2f}",
                fmt_int(row["selected_count"]),
                fmt_int(row["eligible_resamples"]),
                f"{float(row['mean_abs_coefficient']):.3f}",
                "yes" if bool(row["stable"]) else "",
            ]
        )
    return table_html(
        [
            "Domain",
            "Short label",
            "Description",
            "Feature name",
            "Selection prob",
            "Selected",
            "Resamples",
            "Abs coef",
            "Stable",
        ],
        rows,
        raw=True,
        numeric_columns={4, 5, 6, 7},
        table_id="stability-table",
    )


def selected_features_table_html(selected_features: pd.DataFrame) -> str:
    if selected_features.empty:
        return '<p class="caption">No selected features were recorded.</p>'
    data = selected_features.copy()
    grouped = (
        data.groupby(["base_domain", "domain_label", "selection_method", "selection_method_label", "feature", "feature_label", "feature_description"], dropna=False)
        .agg(
            times_selected=("feature", "size"),
            mean_rank=("rank", "mean"),
            mean_auc=("train_auc", "mean"),
            mean_abs_coef=("coefficient", lambda values: float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else math.nan),
        )
        .reset_index()
    )
    method_order = {"relevance_collinearity": 0, "elastic_net_nonzero": 1}
    domain_order = {"calcium": 0, "fat": 1, "wall_from_fat": 2, "wall_thickness": 3, "all_imaging": 4}
    grouped["method_order"] = grouped["selection_method"].map(lambda value: method_order.get(value, 99))
    grouped["domain_order"] = grouped["base_domain"].map(lambda value: domain_order.get(value, 99))
    grouped = grouped.sort_values(["domain_order", "method_order", "times_selected", "mean_rank"], ascending=[True, True, False, True])
    grouped = grouped.groupby(["base_domain", "selection_method"], group_keys=False).head(16)
    rows = []
    for _, row in grouped.head(180).iterrows():
        coef = "" if math.isnan(float(row["mean_abs_coef"])) else f"{float(row['mean_abs_coef']):.3f}"
        rows.append(
            [
                esc(str(row["domain_label"])),
                esc(str(row["selection_method_label"])),
                esc(str(row["feature_label"])),
                esc(str(row["feature_description"])),
                esc(str(row["feature"])),
                fmt_int(row["times_selected"]),
                f"{float(row['mean_rank']):.1f}",
                f"{float(row['mean_auc']):.3f}",
                coef,
            ]
        )
    return table_html(
        ["Domain", "Use", "Short label", "Description", "Feature name", "Rows", "Mean rank", "Train AUC", "Abs coef"],
        rows,
        raw=True,
        numeric_columns={5, 6, 7, 8},
        table_id="selected-features-table",
    )


def collinearity_table_html(collinearity_drops: pd.DataFrame) -> str:
    if collinearity_drops.empty:
        return '<p class="caption">No features exceeded the collinearity threshold.</p>'
    data = collinearity_drops.copy()
    data["base_domain"] = data["domain"].str.split(":").str[0]
    data = data.sort_values(["abs_correlation", "train_auc"], ascending=False).head(200)
    rows = []
    for _, row in data.iterrows():
        rows.append(
            [
                esc(str(row.get("domain_label", DOMAIN_LABELS.get(str(row["base_domain"]), str(row["base_domain"]))))),
                esc(str(row.get("feature_label", short_feature_name(str(row["feature"]))))),
                esc(str(row.get("correlated_with_label", short_feature_name(str(row["correlated_with"]))))),
                f"{float(row['abs_correlation']):.3f}",
                fmt_int(row["n_pair"]),
                f"{float(row['train_auc']):.3f}",
                f"{float(row['kept_train_auc']):.3f}",
            ]
        )
    return table_html(
        ["Domain", "Dropped", "Kept instead", "Abs corr", "Pair n", "Dropped AUC", "Kept AUC"],
        rows,
        raw=True,
        numeric_columns={3, 4, 5, 6},
        table_id="collinearity-table",
    )


def ratio_features_table_html(ratio_features: list[str]) -> str:
    if not ratio_features:
        return '<p class="caption">No aorta-size ratio features were generated.</p>'
    rows_by_key: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}
    for feature in ratio_features:
        domain = DOMAIN_LABELS.get(feature_domain(feature), feature_domain(feature) or "Other")
        denominator = ratio_denominator_label(feature)
        key = (domain, denominator)
        rows_by_key[key] += 1
        examples.setdefault(key, feature_label(feature))
    rows = []
    for (domain, denominator), count in sorted(rows_by_key.items()):
        rows.append([esc(domain), esc(denominator), fmt_int(count), esc(examples[(domain, denominator)])])
    return table_html(
        ["Domain", "Denominator", "Features", "Example label"],
        rows,
        raw=True,
        numeric_columns={2},
        table_id="ratio-table",
    )


def ratio_denominator_label(feature: str) -> str:
    if feature.endswith("__per_aortic_length_cm"):
        return "Aortic length"
    if feature.endswith("__per_aorta_volume"):
        return "Aorta volume"
    if feature.endswith("__per_equiv_aortic_radius_mm"):
        return "Equivalent aortic radius"
    if feature.endswith("__per_equiv_aortic_diameter_mm"):
        return "Equivalent aortic diameter"
    if feature.endswith("__per_wall_volume"):
        return "Wall volume"
    return "Other"


def score_column_table_html(scores: pd.DataFrame) -> str:
    columns = score_columns(scores)
    rows = []
    for column in columns:
        values = pd.to_numeric(scores[column], errors="coerce")
        rows.append(
            [
                esc(score_label(column)),
                esc(DOMAIN_LABELS.get(score_domain(column), score_domain(column))),
                esc(score_type(column)),
                fmt_int(values.notna().sum()),
                f"{float(values.mean()):.4f}" if values.notna().any() else "",
                f"{float(values.std()):.4f}" if values.notna().sum() > 1 else "",
                esc(column),
            ]
        )
    return table_html(
        ["Score", "Domain", "Type", "Nonmissing", "Mean", "SD", "Column"],
        rows,
        raw=True,
        numeric_columns={3, 4, 5},
        table_id="score-column-table",
    )


def table_html(
    headers: list[str],
    rows: list[list[str]],
    *,
    raw: bool = False,
    numeric_columns: set[int] | None = None,
    table_id: str | None = None,
    classes: str = "data-table sortable",
) -> str:
    numeric_columns = numeric_columns or set()
    table_attrs = f' class="{esc(classes)}"'
    if table_id:
        table_attrs += f' id="{esc(table_id)}"'
    parts = [f'<div class="table-wrap"><table{table_attrs}><thead><tr>']
    for i, header in enumerate(headers):
        cls = ' class="num"' if i in numeric_columns else ""
        parts.append(f"<th{cls}>{esc(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for i, value in enumerate(row):
            cls = ' class="num"' if i in numeric_columns else ""
            cell = str(value) if raw else esc(value)
            parts.append(f"<td{cls}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def html_page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(REPORT_PAGE_TITLE)}</title>
<style>
{css()}
</style>
</head>
<body>
<main>
{body}
</main>
<script>
{javascript()}
</script>
</body>
</html>
"""


def css() -> str:
    return """
:root { --paper: #fffff8; --ink: #151515; --muted: #666; --rule: #d8d0bc; --faint: #efeadc; --blue: #335f86; --red: #8c3d3d; --brown: #8a5a2b; --green: #4c7f73; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: Georgia, "Times New Roman", serif; line-height: 1.48; }
main { max-width: 1180px; margin: 0 auto; padding: 42px 34px 70px; }
header.hero { border-bottom: 1px solid var(--rule); padding-bottom: 18px; margin-bottom: 26px; }
.eyebrow { margin: 0 0 10px; color: var(--muted); font-size: 13px; letter-spacing: .08em; text-transform: uppercase; }
h1 { font-size: clamp(36px, 6vw, 66px); line-height: .98; font-weight: 400; margin: 0 0 14px; }
h2 { font-size: 24px; line-height: 1.1; font-weight: 400; margin: 0 0 12px; }
h3 { font-size: 18px; line-height: 1.15; font-weight: 400; margin: 18px 0 10px; }
.subtitle, .caption { color: var(--muted); }
.narrative { max-width: 820px; font-size: 19px; }
.newthought { font-variant: small-caps; letter-spacing: .03em; }
.stat-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 22px 0 28px; }
.stat { padding: 18px 18px 15px 0; border-right: 1px solid var(--rule); }
.stat:last-child { border-right: 0; }
.stat .value { font-size: 42px; line-height: 1; font-variant-numeric: tabular-nums; }
.stat .label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .06em; margin-top: 7px; }
.tabs { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 6px; background: color-mix(in srgb, var(--paper) 94%, white); border-bottom: 1px solid var(--rule); padding: 10px 0; margin: 18px 0 24px; }
.tab-button { border: 1px solid var(--rule); background: transparent; color: var(--ink); font: 14px Georgia, "Times New Roman", serif; padding: 7px 10px; cursor: pointer; }
.tab-button.active { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.tab-panel { display: none; margin: 28px 0 44px; }
.tab-panel.active { display: block; }
.two-column { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr); gap: 30px; align-items: start; }
.summary-box { border-left: 1px solid var(--rule); padding-left: 20px; }
.figure-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; margin: 22px 0 28px; }
.figure-grid.single { grid-template-columns: 1fr; }
.figure-panel { margin: 0; min-width: 0; }
.figure-panel figcaption { border-top: 1px solid var(--rule); padding-top: 9px; margin-bottom: 7px; color: var(--muted); display: grid; gap: 2px; font-size: 13px; }
.figure-panel figcaption strong { color: var(--ink); font-weight: 400; font-size: 16px; }
.figure-img { width: 100%; display: block; border-bottom: 1px solid var(--rule); padding: 8px 0 10px; }
.control-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 12px 0 18px; }
.control-row label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }
input[type="search"], select { border: 1px solid var(--rule); background: #fffdf2; color: var(--ink); font: 14px Georgia, "Times New Roman", serif; min-height: 34px; padding: 6px 9px; max-width: 100%; }
input[type="search"] { min-width: min(280px, 100%); }
select { min-width: min(420px, 100%); }
.table-wrap { overflow-x: auto; border-top: 1px solid var(--rule); margin: 12px 0 26px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 400; color: var(--muted); border-bottom: 1px solid var(--rule); padding: 7px 8px 7px 0; white-space: nowrap; cursor: pointer; }
td { border-bottom: 1px solid var(--faint); padding: 7px 8px 7px 0; vertical-align: top; }
.compact-table { font-size: 13px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .9em; background: #f5f1e4; padding: 1px 4px; }
.score-case-panel { display: none; }
.score-case-panel.active { display: block; }
@media (max-width: 900px) {
  main { padding: 28px 18px 54px; }
  .stat-strip { grid-template-columns: 1fr; }
  .stat { border-right: 0; border-bottom: 1px solid var(--rule); }
  .two-column, .figure-grid { grid-template-columns: 1fr; }
  .summary-box { border-left: 0; padding-left: 0; }
  .tabs { position: static; }
}
"""


def javascript() -> str:
    return """
(() => {
  const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));
  function activateTab(name) {
    tabButtons.forEach((button) => {
      const active = button.dataset.tabTarget === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    panels.forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
    if (history.replaceState) history.replaceState(null, "", `#${name}`);
  }
  tabButtons.forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tabTarget)));
  const initial = location.hash ? location.hash.slice(1) : "overview";
  if (tabButtons.some((button) => button.dataset.tabTarget === initial)) activateTab(initial);

  document.querySelectorAll("[data-filter-table]").forEach((input) => {
    input.addEventListener("input", () => {
      const table = document.getElementById(input.dataset.filterTable);
      if (!table) return;
      const needle = input.value.trim().toLowerCase();
      table.querySelectorAll("tbody tr").forEach((row) => {
        row.style.display = row.textContent.toLowerCase().includes(needle) ? "" : "none";
      });
    });
  });

  document.querySelectorAll("table.sortable th").forEach((header, columnIndex) => {
    header.addEventListener("click", () => {
      const table = header.closest("table");
      const tbody = table.querySelector("tbody");
      const current = header.getAttribute("data-sort") || "none";
      const direction = current === "asc" ? "desc" : "asc";
      table.querySelectorAll("th").forEach((cell) => cell.removeAttribute("data-sort"));
      header.setAttribute("data-sort", direction);
      const rows = Array.from(tbody.querySelectorAll("tr"));
      rows.sort((a, b) => {
        const av = a.children[columnIndex]?.textContent.trim() || "";
        const bv = b.children[columnIndex]?.textContent.trim() || "";
        const an = Number(av.replace(/,/g, ""));
        const bn = Number(bv.replace(/,/g, ""));
        const comparison = Number.isFinite(an) && Number.isFinite(bn)
          ? an - bn
          : av.localeCompare(bv, undefined, {numeric: true, sensitivity: "base"});
        return direction === "asc" ? comparison : -comparison;
      });
      rows.forEach((row) => tbody.appendChild(row));
    });
  });

  const selector = document.getElementById("score-case-select");
  if (selector) {
    const syncScorePanel = () => {
      document.querySelectorAll("[data-score-panel]").forEach((panel) => {
        panel.classList.toggle("active", panel.dataset.scorePanel === selector.value);
      });
    };
    selector.addEventListener("change", syncScorePanel);
    syncScorePanel();
  }
})();
"""


def stat_card(label: str, value: str) -> str:
    return f'<div class="stat"><div class="value">{esc(value)}</div><div class="label">{esc(label)}</div></div>'


def embedded_image(data_uri: str, alt: str) -> str:
    if not data_uri:
        return '<p class="caption">No figure could be generated.</p>'
    return f'<img class="figure-img" src="{esc(data_uri)}" alt="{esc(alt)}">'


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_float(value: object, *, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


if __name__ == "__main__":
    main()
