"""Core leakage-safe feature selection engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import RunConfig
from .projections import build_final_projection, build_projection, fit_projection_transform


BLOCKED_RUNTIME_DEPENDENCIES = {
    "anthropic",
    "cohere",
    "google-generativeai",
    "langchain",
    "llama-index",
    "openai",
    "tiktoken",
}


@dataclass(slots=True)
class RobustnessSummary:
    status: str
    path: str | None
    kept_features: int
    rejected_features: int
    details: str


@dataclass(slots=True)
class FeatureMetadataSummary:
    status: str
    path: str | None
    audited_features: int
    compliant_features: int
    rejected_features: int
    details: str


@dataclass(slots=True)
class SelectionRow:
    modality: str
    fold: str
    feature: str
    rank: int
    relevance: float
    sign: int
    missing_fraction: float
    coefficient: float
    stage: str


@dataclass(slots=True)
class DropRow:
    modality: str
    fold: str
    feature: str
    reason: str
    value: float
    compared_with: str = ""


@dataclass(slots=True)
class CorrelationRow:
    modality: str
    fold: str
    kept_feature: str
    dropped_feature: str
    abs_correlation: float
    threshold: float
    method: str
    kept_relevance: float
    dropped_relevance: float
    decision: str


@dataclass(slots=True)
class TuningRow:
    modality: str
    outer_fold: str
    candidate: str
    elastic_net_c: float
    elastic_net_alpha: float
    elastic_net_l1_ratio: float
    mean_inner_score: float
    selected: bool
    metric: str


@dataclass(slots=True)
class RadselectResult:
    selected_features: pd.DataFrame
    dropped_features: pd.DataFrame
    stability_selection: pd.DataFrame
    stability_resamples: pd.DataFrame
    performance: pd.DataFrame
    predictions: pd.DataFrame
    composite_scores: pd.DataFrame
    final_signature: pd.DataFrame
    final_signature_parameters: pd.DataFrame
    final_composite_scores: pd.DataFrame
    projection_performance: pd.DataFrame
    projection_predictions: pd.DataFrame
    projection_scores: pd.DataFrame
    projection_loadings: pd.DataFrame
    final_projection_scores: pd.DataFrame
    final_projection_parameters: pd.DataFrame
    tuning_summary: pd.DataFrame
    schema_audit: pd.DataFrame
    column_audit: pd.DataFrame
    modality_audit: pd.DataFrame
    correlation_audit: pd.DataFrame
    dependency_audit: pd.DataFrame
    robustness_audit: pd.DataFrame
    validation_splits: pd.DataFrame
    quality_checks: pd.DataFrame
    feature_metadata_audit: pd.DataFrame
    sample_audit: pd.DataFrame
    manifest: dict

    def write(self, outdir: str | Path) -> None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.selected_features.to_csv(out / "selected_features.csv", index=False)
        self.dropped_features.to_csv(out / "dropped_features.csv", index=False)
        self.stability_selection.to_csv(out / "stability_selection.csv", index=False)
        self.stability_resamples.to_csv(out / "stability_resamples.csv", index=False)
        self.performance.to_csv(out / "performance.csv", index=False)
        self.predictions.to_csv(out / "predictions.csv", index=False)
        self.composite_scores.to_csv(out / "composite_scores.csv", index=False)
        self.final_signature.to_csv(out / "final_signature.csv", index=False)
        self.final_signature_parameters.to_csv(out / "final_signature_parameters.csv", index=False)
        self.final_composite_scores.to_csv(out / "final_composite_scores.csv", index=False)
        projection_enabled = bool(
            self.manifest.get("projection_validation", {}).get("enabled")
            or self.manifest.get("final_projection", {}).get("enabled")
        )
        if projection_enabled or not self.projection_performance.empty:
            self.projection_performance.to_csv(out / "projection_performance.csv", index=False)
        if projection_enabled or not self.projection_predictions.empty:
            self.projection_predictions.to_csv(out / "projection_predictions.csv", index=False)
        self.column_audit.to_csv(out / "column_audit.csv", index=False)
        self.modality_audit.to_csv(out / "modality_audit.csv", index=False)
        self.correlation_audit.to_csv(out / "correlation_audit.csv", index=False)
        self.dependency_audit.to_csv(out / "dependency_audit.csv", index=False)
        self.robustness_audit.to_csv(out / "robustness_audit.csv", index=False)
        self.validation_splits.to_csv(out / "validation_splits.csv", index=False)
        self.quality_checks.to_csv(out / "quality_checks.csv", index=False)
        self.feature_metadata_audit.to_csv(out / "feature_metadata_audit.csv", index=False)
        self.sample_audit.to_csv(out / "sample_audit.csv", index=False)
        selected_frequency = selected_feature_frequency(self.selected_features, self.stability_selection)
        selected_frequency.to_csv(out / "selected_feature_frequency.csv", index=False)
        if projection_enabled or not self.projection_scores.empty:
            self.projection_scores.to_csv(out / "projection_scores.csv", index=False)
        if projection_enabled or not self.projection_loadings.empty:
            self.projection_loadings.to_csv(out / "projection_loadings.csv", index=False)
        if projection_enabled or not self.final_projection_scores.empty:
            self.final_projection_scores.to_csv(out / "final_projection_scores.csv", index=False)
        if projection_enabled or not self.final_projection_parameters.empty:
            self.final_projection_parameters.to_csv(out / "final_projection_parameters.csv", index=False)
        self.tuning_summary.to_csv(out / "tuning_summary.csv", index=False)
        self.schema_audit.to_csv(out / "schema_audit.csv", index=False)
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        (out / "provenance.json").write_text(json.dumps(build_provenance(self.manifest), indent=2), encoding="utf-8")
        write_output_manifest(out)


@dataclass(slots=True)
class FittedSelector:
    modality: str
    selected_features: list[str]
    rows: list[SelectionRow]
    drops: list[DropRow]
    tuning_rows: list[TuningRow]
    correlation_rows: list[CorrelationRow]
    model_status: str


def run_selection(
    data: pd.DataFrame,
    config: RunConfig,
    *,
    external_data: pd.DataFrame | None = None,
) -> RadselectResult:
    """Run leakage-safe feature selection and validation on tabular features."""

    config.validate()
    raw_development, holdout_data, holdout_summary = split_holdout_groups(data.copy(), config)
    if external_data is not None and holdout_data is not None:
        raise ValueError("Use either external_data or holdout_groups, not both in the same run.")
    if holdout_data is not None:
        external_data = holdout_data
    frame, development_sample_audit = audit_and_filter_samples(raw_development, config, "development")
    if frame.empty:
        raise ValueError("No development rows remain after outcome/time/event validity filtering.")
    external_sample_audit = pd.DataFrame()
    if external_data is not None:
        external_data, external_sample_audit = audit_and_filter_samples(external_data.copy(), config, "external")
        if external_data.empty:
            raise ValueError("No external rows remain after outcome/time/event validity filtering.")
    sample_audit = pd.concat([development_sample_audit, external_sample_audit], ignore_index=True)
    metadata_allowed, metadata_rejected, feature_metadata_summary, feature_metadata_audit = load_feature_metadata_filter(
        config
    )
    robustness_allowed, robustness_rejected, robustness_summary, robustness_audit = load_robustness_filter(config)
    allowed = combine_allowed_sets(metadata_allowed, robustness_allowed)
    rejected = metadata_rejected | robustness_rejected
    modalities = build_modalities(frame, config, allowed, rejected)
    if not modalities:
        raise ValueError("No feature columns are available after applying feature and robustness filters.")
    modality_audit = build_modality_audit(frame, config, modalities, allowed, rejected)
    schema_audit = build_schema_audit(frame, external_data, config, modalities)
    validate_schema_audit(schema_audit)
    column_audit = build_column_audit(frame, config, modalities)
    dependency_audit = build_dependency_audit()

    y = make_target(frame, config)
    cv_rows: list[SelectionRow] = []
    drop_rows: list[DropRow] = []
    tuning_rows: list[TuningRow] = []
    correlation_rows: list[CorrelationRow] = []
    performance_rows: list[dict] = []
    prediction_rows: list[dict] = []
    composite_score_rows: list[dict] = []
    projection_performance_rows: list[dict] = []
    projection_prediction_rows: list[dict] = []
    outer_splits = list(iter_validation_splits(frame, y, config))
    validation_splits = build_validation_split_audit(frame, outer_splits, config, external_data)

    for modality, columns in modalities.items():
        for fold_name, train_idx, test_idx in outer_splits:
            fitted = fit_selector(
                frame.iloc[train_idx],
                y_train=target_subset(y, train_idx),
                columns=columns,
                modality=modality,
                fold=fold_name,
                config=config,
            )
            cv_rows.extend(fitted.rows)
            drop_rows.extend(fitted.drops)
            tuning_rows.extend(fitted.tuning_rows)
            correlation_rows.extend(fitted.correlation_rows)
            validation = evaluate_selected_model(
                train=frame.iloc[train_idx],
                test=frame.iloc[test_idx],
                y_train=target_subset(y, train_idx),
                y_test=target_subset(y, test_idx),
                features=fitted.selected_features,
                modality=modality,
                fold=fold_name,
                config=config,
                id_column=config.id_column,
            )
            performance_rows.extend(validation["performance"])
            prediction_rows.extend(validation["predictions"])
            composite_score_rows.extend(
                build_composite_scores(
                    train=frame.iloc[train_idx],
                    test=frame.iloc[test_idx],
                    y_test=target_subset(y, test_idx),
                    fitted=fitted,
                    fold=fold_name,
                    config=config,
                    id_column=config.id_column,
                )
            )
            projection_validation = evaluate_projection_model(
                train=frame.iloc[train_idx],
                test=frame.iloc[test_idx],
                y_train=target_subset(y, train_idx),
                y_test=target_subset(y, test_idx),
                columns=columns,
                modality=modality,
                fold=fold_name,
                config=config,
                id_column=config.id_column,
            )
            projection_performance_rows.extend(projection_validation["performance"])
            projection_prediction_rows.extend(projection_validation["predictions"])

        if external_data is not None:
            ext_y = make_target(external_data, config)
            fitted = fit_selector(
                frame,
                y_train=y,
                columns=columns,
                modality=modality,
                fold="external_fit",
                config=config,
            )
            cv_rows.extend(fitted.rows)
            drop_rows.extend(fitted.drops)
            tuning_rows.extend(fitted.tuning_rows)
            correlation_rows.extend(fitted.correlation_rows)
            validation = evaluate_selected_model(
                train=frame,
                test=external_data,
                y_train=y,
                y_test=ext_y,
                features=fitted.selected_features,
                modality=modality,
                fold="external",
                config=config,
                id_column=config.id_column,
            )
            performance_rows.extend(validation["performance"])
            prediction_rows.extend(validation["predictions"])
            composite_score_rows.extend(
                build_composite_scores(
                    train=frame,
                    test=external_data,
                    y_test=ext_y,
                    fitted=fitted,
                    fold="external",
                    config=config,
                    id_column=config.id_column,
                )
            )
            projection_validation = evaluate_projection_model(
                train=frame,
                test=external_data,
                y_train=y,
                y_test=ext_y,
                columns=columns,
                modality=modality,
                fold="external",
                config=config,
                id_column=config.id_column,
            )
            projection_performance_rows.extend(projection_validation["performance"])
            projection_prediction_rows.extend(projection_validation["predictions"])

    stability, stability_resamples = run_stability_selection(frame, y, modalities, config)
    projection_scores, projection_loadings = build_projection(frame, y, modalities, config)
    final_projection_scores, final_projection_parameters = build_final_projection(
        frame,
        y,
        modalities,
        config,
        external_data=external_data,
    )
    (
        final_signature_df,
        final_signature_parameters_df,
        final_composite_scores_df,
        final_correlation_audit_df,
    ) = build_final_refit_outputs(
        frame=frame,
        y=y,
        modalities=modalities,
        stability=stability,
        config=config,
        external_data=external_data,
    )
    selected_df = selection_rows_to_frame(cv_rows)
    dropped_df = drop_rows_to_frame(drop_rows)
    correlation_df = pd.concat(
        [correlation_rows_to_frame(correlation_rows), final_correlation_audit_df],
        ignore_index=True,
    )
    performance_df = performance_rows_to_frame(performance_rows)
    prediction_df = prediction_rows_to_frame(prediction_rows)
    composite_scores_df = composite_score_rows_to_frame(composite_score_rows)
    projection_performance_df = projection_performance_rows_to_frame(projection_performance_rows)
    projection_prediction_df = projection_prediction_rows_to_frame(projection_prediction_rows)
    tuning_df = tuning_rows_to_frame(tuning_rows)
    quality_checks = build_quality_checks(
        config=config,
        schema_audit=schema_audit,
        column_audit=column_audit,
        modality_audit=modality_audit,
        correlation_audit=correlation_df,
        dependency_audit=dependency_audit,
        validation_splits=validation_splits,
        final_signature_parameters=final_signature_parameters_df,
        final_projection_parameters=final_projection_parameters,
        projection_performance=projection_performance_df,
        tuning_summary=tuning_df,
        feature_metadata_summary=feature_metadata_summary,
        robustness_summary=robustness_summary,
        stability_resamples=stability_resamples,
    )
    manifest = {
        "package": "radselect",
        "task": config.task,
        "n_rows_raw": int(len(data)),
        "n_rows": int(len(frame)),
        "holdout_validation": holdout_summary,
        "sample_audit": {
            "rows": int(len(sample_audit)),
            "development_rows_retained": int(
                ((sample_audit["dataset"] == "development") & (sample_audit["status"] == "retained")).sum()
            ),
            "development_rows_dropped": int(
                ((sample_audit["dataset"] == "development") & (sample_audit["status"] == "dropped")).sum()
            ),
            "external_rows_retained": int(
                ((sample_audit["dataset"] == "external") & (sample_audit["status"] == "retained")).sum()
            ),
            "external_rows_dropped": int(
                ((sample_audit["dataset"] == "external") & (sample_audit["status"] == "dropped")).sum()
            ),
        },
        "modalities": {name: len(cols) for name, cols in modalities.items()},
        "modality_audit": {
            "rows": int(len(modality_audit)),
            "modalities": int(modality_audit["modality"].nunique()) if not modality_audit.empty else 0,
            "included_memberships": int(modality_audit["included_in_modality"].sum())
            if not modality_audit.empty and "included_in_modality" in modality_audit.columns
            else 0,
            "included_features": int(
                modality_audit.loc[modality_audit["included_in_modality"].astype(bool), "feature"].nunique()
            )
            if not modality_audit.empty and "included_in_modality" in modality_audit.columns
            else 0,
        },
        "correlation_audit": {
            "rows": int(len(correlation_df)),
            "redundant_features_dropped": int(correlation_df["dropped_feature"].nunique())
            if not correlation_df.empty and "dropped_feature" in correlation_df.columns
            else 0,
            "threshold": float(config.correlation_threshold),
            "method": config.correlation_method,
        },
        "dependency_audit": dependency_audit_summary(dependency_audit),
        "analysis_method": analysis_method_summary(config),
        "outcome_summary": outcome_summary(y, config),
        "schema_audit": {
            "rows": int(len(schema_audit)),
            "external_issues": int(
                schema_audit["issue"].astype(str).ne("").sum()
                if not schema_audit.empty and "issue" in schema_audit.columns
                else 0
            ),
        },
        "validation_splits": {
            "rows": int(len(validation_splits)),
            "folds": int(validation_splits["fold"].nunique()) if not validation_splits.empty else 0,
            "external_rows": int(
                (validation_splits["dataset"].astype(str).eq("external")).sum()
                if not validation_splits.empty and "dataset" in validation_splits.columns
                else 0
            ),
        },
        "composite_scores": {
            "rows": int(len(composite_scores_df)),
            "method": "training-standardized weighted sum of selected features",
        },
        "final_signature": {
            "rows": int(len(final_signature_df)),
            "parameter_rows": int(len(final_signature_parameters_df)),
            "score_rows": int(len(final_composite_scores_df)),
            "note": "Final refit is trained on retained development rows for downstream use after validation.",
        },
        "projection_validation": {
            "enabled": config.projection != "none",
            "rows": int(len(projection_performance_df)),
        },
        "final_projection": {
            "enabled": config.projection != "none",
            "score_rows": int(len(final_projection_scores)),
            "parameter_rows": int(len(final_projection_parameters)),
            "note": "Final projection is fit on retained development rows for downstream transformation after validation.",
        },
        "stability_analysis": {
            "resamples_requested": int(config.stability_resamples),
            "resample_rows": int(len(stability_resamples)),
            "selection_rows": int(len(stability)),
            "threshold": float(config.stability_threshold),
            "sampling_unit": "group" if config.group_column and config.group_column in frame.columns else "row",
            "group_column": config.group_column,
        },
        "nested_tuning": nested_tuning_summary(config, tuning_df),
        "quality_checks": quality_check_summary(quality_checks),
        "feature_metadata": asdict(feature_metadata_summary),
        "robustness": asdict(robustness_summary),
        "column_audit": {
            "columns": int(len(column_audit)),
            "candidate_features": int(column_audit["included_as_candidate"].sum()),
            "metadata_columns": int(column_audit["role"].isin(["id", "outcome", "time", "event", "group"]).sum()),
            "leakage_risk_flags": int(column_audit["leakage_risk"].sum()),
        },
        "selection_pipeline": [
            "already-extracted IBSI/provenance-compatible feature table",
            "IBSI/provenance feature metadata audit or filter when provided",
            "test-retest/segmentation/acquisition robustness filter when provided",
            "foldwise missingness and near-zero-variance removal",
            f"foldwise {config.screening_method} conventional feature screening",
            "foldwise correlation redundancy filtering",
            "inner-loop elastic-net hyperparameter tuning inside each outer training fold",
            "foldwise elastic-net model-based selection with tuned parameters",
            "repeated-resample stability selection",
            "outer validation plus optional center/group-held-out or external validation",
        ],
        "config": config_to_manifest(config),
    }
    return RadselectResult(
        selected_features=selected_df,
        dropped_features=dropped_df,
        stability_selection=stability,
        stability_resamples=stability_resamples,
        performance=performance_df,
        predictions=prediction_df,
        composite_scores=composite_scores_df,
        final_signature=final_signature_df,
        final_signature_parameters=final_signature_parameters_df,
        final_composite_scores=final_composite_scores_df,
        projection_performance=projection_performance_df,
        projection_predictions=projection_prediction_df,
        projection_scores=projection_scores,
        projection_loadings=projection_loadings,
        final_projection_scores=final_projection_scores,
        final_projection_parameters=final_projection_parameters,
        tuning_summary=tuning_df,
        schema_audit=schema_audit,
        column_audit=column_audit,
        modality_audit=modality_audit,
        correlation_audit=correlation_df,
        dependency_audit=dependency_audit,
        robustness_audit=robustness_audit,
        validation_splits=validation_splits,
        quality_checks=quality_checks,
        feature_metadata_audit=feature_metadata_audit,
        sample_audit=sample_audit,
        manifest=manifest,
    )


def build_validation_split_audit(
    development: pd.DataFrame,
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    config: RunConfig,
    external_data: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold_name, train_idx, test_idx in splits:
        rows.extend(split_rows(development, train_idx, fold_name, "development", "train", config))
        rows.extend(split_rows(development, test_idx, fold_name, "development", "test", config))
    if external_data is not None:
        rows.extend(
            split_rows(
                external_data,
                np.arange(len(external_data)),
                "external",
                "external",
                "external_validation",
                config,
            )
        )
    return pd.DataFrame(
        rows,
        columns=["fold", "dataset", "role", "row_index", "id", "group"],
    )


def split_rows(
    frame: pd.DataFrame,
    indices: np.ndarray,
    fold: str,
    dataset: str,
    role: str,
    config: RunConfig,
) -> list[dict]:
    rows = []
    for index in indices:
        position = int(index)
        row = frame.iloc[position]
        rows.append(
            {
                "fold": fold,
                "dataset": dataset,
                "role": role,
                "row_index": position,
                "id": row[config.id_column] if config.id_column and config.id_column in frame.columns else position,
                "group": row[config.group_column] if config.group_column and config.group_column in frame.columns else "",
            }
        )
    return rows


def build_quality_checks(
    *,
    config: RunConfig,
    schema_audit: pd.DataFrame,
    column_audit: pd.DataFrame,
    modality_audit: pd.DataFrame,
    correlation_audit: pd.DataFrame,
    dependency_audit: pd.DataFrame,
    validation_splits: pd.DataFrame,
    final_signature_parameters: pd.DataFrame,
    final_projection_parameters: pd.DataFrame,
    projection_performance: pd.DataFrame,
    tuning_summary: pd.DataFrame,
    feature_metadata_summary: FeatureMetadataSummary,
    robustness_summary: RobustnessSummary,
    stability_resamples: pd.DataFrame,
) -> pd.DataFrame:
    dependency_summary = dependency_audit_summary(dependency_audit)
    rows = [
        quality_check(
            "runtime_has_no_llm_dependency",
            "pass" if dependency_summary["blocked_runtime_dependencies"] == 0 else "fail",
            (
                f"{dependency_summary['blocked_runtime_dependencies']} blocked LLM/OpenAI package names found "
                "in radselect runtime requirements."
            ),
        ),
        quality_check(
            "validation_splits_recorded",
            "pass" if not validation_splits.empty else "warning",
            f"{len(validation_splits)} split-membership rows recorded.",
        ),
        quality_check(
            "external_schema_validated",
            "pass" if schema_external_issue_count(schema_audit) == 0 else "fail",
            f"{schema_external_issue_count(schema_audit)} external schema issues recorded.",
        ),
        quality_check(
            "metadata_columns_protected",
            "pass" if metadata_candidate_count(column_audit) == 0 else "fail",
            f"{metadata_candidate_count(column_audit)} ID/outcome/time/event/group columns included as candidates.",
        ),
    ]
    leakage_flags = int(column_audit["leakage_risk"].sum()) if not column_audit.empty else 0
    rows.append(
        quality_check(
            "outcome_like_candidate_names_reviewed",
            "pass" if leakage_flags == 0 else "warning",
            f"{leakage_flags} candidate feature names were flagged as outcome-like in column_audit.csv.",
        )
    )
    included_memberships = (
        int(modality_audit["included_in_modality"].sum())
        if not modality_audit.empty and "included_in_modality" in modality_audit.columns
        else 0
    )
    rows.append(
        quality_check(
            "modality_definitions_recorded",
            "pass" if included_memberships > 0 else "warning",
            f"{included_memberships} included modality/domain feature memberships recorded in modality_audit.csv.",
        )
    )
    rows.append(
        quality_check(
            "correlation_redundancy_audited",
            "pass",
            f"{len(correlation_audit)} correlation redundancy decisions recorded in correlation_audit.csv.",
        )
    )
    rows.append(
        quality_check(
            "feature_metadata_audited",
            "not_applicable" if feature_metadata_summary.status == "not_provided" else "pass",
            feature_metadata_summary.details,
        )
    )
    rows.append(
        quality_check(
            "robustness_screening_audited",
            "not_applicable" if robustness_summary.status == "not_provided" else "pass",
            robustness_summary.details,
        )
    )
    rows.append(
        quality_check(
            "stability_resamples_recorded",
            "pass" if not stability_resamples.empty else "not_applicable",
            f"{len(stability_resamples)} stability resample rows recorded.",
        )
    )
    tuning = nested_tuning_summary(config, tuning_summary)
    if not tuning["enabled"]:
        rows.append(
            quality_check(
                "nested_elastic_net_tuning_recorded",
                "not_applicable",
                "Elastic-net tuning was disabled for this run.",
            )
        )
    else:
        rows.append(
            quality_check(
                "nested_elastic_net_tuning_recorded",
                "pass" if tuning["selected_candidate_rows"] > 0 else "warning",
                (
                    f"{tuning['rows']} tuning rows recorded; "
                    f"{tuning['selected_candidate_rows']} selected candidate rows."
                ),
            )
        )
    rows.append(
        quality_check(
            "final_signature_reproducible",
            "pass" if not final_signature_parameters.empty else "warning",
            f"{len(final_signature_parameters)} final signature parameter rows written.",
        )
    )
    if config.projection == "none":
        rows.append(
            quality_check(
                "final_projection_reproducible",
                "not_applicable",
                "Projection was disabled for this run.",
            )
        )
        rows.append(
            quality_check(
                "projection_validation_foldwise",
                "not_applicable",
                "Projection was disabled for this run.",
            )
        )
    else:
        rows.append(
            quality_check(
                "final_projection_reproducible",
                "pass" if not final_projection_parameters.empty else "warning",
                f"{len(final_projection_parameters)} final projection parameter rows written.",
            )
        )
        ok_rows = (
            projection_performance["status"].astype(str).eq("ok").sum()
            if not projection_performance.empty and "status" in projection_performance.columns
            else 0
        )
        rows.append(
            quality_check(
                "projection_validation_foldwise",
                "pass" if ok_rows > 0 else "warning",
                f"{ok_rows} projection validation rows reported status=ok.",
            )
        )
    return pd.DataFrame(rows, columns=["check", "status", "details"])


def quality_check(check: str, status: str, details: str) -> dict:
    return {"check": check, "status": status, "details": details}


def schema_external_issue_count(schema_audit: pd.DataFrame) -> int:
    if schema_audit.empty or "issue" not in schema_audit.columns:
        return 0
    return int(schema_audit["issue"].astype(str).ne("").sum())


def metadata_candidate_count(column_audit: pd.DataFrame) -> int:
    if column_audit.empty:
        return 0
    protected = column_audit["role"].isin(["id", "outcome", "time", "event", "group"])
    included = column_audit["included_as_candidate"].astype(bool)
    return int((protected & included).sum())


def quality_check_summary(quality_checks: pd.DataFrame) -> dict:
    if quality_checks.empty:
        return {"rows": 0, "status_counts": {}}
    counts = quality_checks["status"].astype(str).value_counts().to_dict()
    return {
        "rows": int(len(quality_checks)),
        "status_counts": {str(status): int(count) for status, count in counts.items()},
    }


def nested_tuning_summary(config: RunConfig, tuning_summary: pd.DataFrame) -> dict:
    selected = (
        tuning_summary["selected"].astype(bool)
        if not tuning_summary.empty and "selected" in tuning_summary.columns
        else pd.Series(dtype=bool)
    )
    return {
        "enabled": bool(config.tune_elastic_net),
        "rows": int(len(tuning_summary)),
        "selected_candidate_rows": int(selected.sum()) if len(selected) else 0,
        "outer_folds": int(tuning_summary["outer_fold"].nunique())
        if not tuning_summary.empty and "outer_fold" in tuning_summary.columns
        else 0,
        "inner_splits": int(config.inner_splits),
        "metric": sorted(tuning_summary["metric"].dropna().astype(str).unique().tolist())
        if not tuning_summary.empty and "metric" in tuning_summary.columns
        else [],
    }


def analysis_method_summary(config: RunConfig) -> dict:
    if config.task == "competing_risk":
        return {
            "task": config.task,
            "method": "cause_specific_cox_or_signed_score_fallback",
            "event_of_interest": str(config.competing_event_code),
            "competing_events_handling": "treated_as_censored_for_selection_and_c_index",
            "note": (
                "This is a cause-specific competing-risk analysis for the configured event code, "
                "not a Fine-Gray subdistribution hazard model."
            ),
        }
    if config.task == "survival":
        return {
            "task": config.task,
            "method": "cox_or_signed_score_fallback",
            "event_handling": "nonzero_or_true_event_values_are_events",
        }
    if config.task in {"binary", "multiclass"}:
        return {
            "task": config.task,
            "method": "elastic_net_logistic",
            "validation_metric_priority": validation_score_metric(config.task),
        }
    return {
        "task": config.task,
        "method": "elastic_net_regression",
        "validation_metric_priority": validation_score_metric(config.task),
    }


def outcome_summary(y: pd.Series | pd.DataFrame, config: RunConfig) -> dict:
    if config.task in {"binary", "multiclass"}:
        values = pd.Series(y).astype(str)
        return {
            "task": config.task,
            "rows": int(len(values)),
            "class_counts": {str(label): int(count) for label, count in values.value_counts(dropna=False).items()},
        }
    if config.task == "regression":
        values = pd.to_numeric(pd.Series(y), errors="coerce")
        return {
            "task": config.task,
            "rows": int(len(values)),
            "non_missing": int(values.notna().sum()),
            "mean": finite_float(values.mean(), math.nan),
            "std": finite_float(values.std(), math.nan),
            "min": finite_float(values.min(), math.nan),
            "max": finite_float(values.max(), math.nan),
        }

    target = pd.DataFrame(y)
    time = pd.to_numeric(target["time"], errors="coerce")
    events = target["event"]
    event_flag = event_indicator(events, config)
    summary = {
        "task": config.task,
        "rows": int(len(target)),
        "time_min": finite_float(time.min(), math.nan),
        "time_median": finite_float(time.median(), math.nan),
        "time_max": finite_float(time.max(), math.nan),
        "events": int(event_flag.sum()),
        "censored": int((~event_flag).sum()),
    }
    if config.task == "competing_risk":
        censored = censored_event_mask(events)
        event_of_interest = events.astype(str).eq(str(config.competing_event_code))
        competing = ~(censored | event_of_interest)
        summary.update(
            {
                "event_of_interest": str(config.competing_event_code),
                "event_of_interest_count": int(event_of_interest.sum()),
                "competing_event_count": int(competing.sum()),
                "censored_count": int(censored.sum()),
                "event_code_counts": {
                    str(label): int(count) for label, count in events.astype(str).value_counts(dropna=False).items()
                },
            }
        )
    return summary


def censored_event_mask(event: pd.Series) -> pd.Series:
    text = event.astype(str).str.strip().str.lower()
    censored = text.isin({"", "0", "false", "no", "none", "nan", "censor", "censored"})
    numeric = pd.to_numeric(event, errors="coerce")
    if numeric.notna().any():
        censored = censored | numeric.fillna(0).eq(0)
    return censored


def config_to_manifest(config: RunConfig) -> dict:
    result = asdict(config)
    if config.feature_metadata_csv is not None:
        result["feature_metadata_csv"] = str(config.feature_metadata_csv)
    if config.robustness_csv is not None:
        result["robustness_csv"] = str(config.robustness_csv)
    return result


def rows_to_frame(rows: Iterable[object]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def selection_rows_to_frame(rows: Iterable[SelectionRow]) -> pd.DataFrame:
    return rows_to_frame(rows).reindex(
        columns=[
            "modality",
            "fold",
            "feature",
            "rank",
            "relevance",
            "sign",
            "missing_fraction",
            "coefficient",
            "stage",
        ]
    )


def drop_rows_to_frame(rows: Iterable[DropRow]) -> pd.DataFrame:
    return rows_to_frame(rows).reindex(
        columns=[
            "modality",
            "fold",
            "feature",
            "reason",
            "value",
            "compared_with",
        ]
    )


def correlation_rows_to_frame(rows: Iterable[CorrelationRow]) -> pd.DataFrame:
    return rows_to_frame(rows).reindex(
        columns=[
            "modality",
            "fold",
            "kept_feature",
            "dropped_feature",
            "abs_correlation",
            "threshold",
            "method",
            "kept_relevance",
            "dropped_relevance",
            "decision",
        ]
    )


def tuning_rows_to_frame(rows: Iterable[TuningRow]) -> pd.DataFrame:
    return rows_to_frame(rows).reindex(
        columns=[
            "modality",
            "outer_fold",
            "candidate",
            "elastic_net_c",
            "elastic_net_alpha",
            "elastic_net_l1_ratio",
            "mean_inner_score",
            "selected",
            "metric",
        ]
    )


def composite_parameter_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(
        columns=[
            "fold",
            "modality",
            "task",
            "feature",
            "median",
            "mean",
            "std",
            "weight",
        ]
    )


def composite_score_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(
        columns=[
            "id",
            "fold",
            "modality",
            "task",
            "score_type",
            "composite_score",
            "raw_composite_score",
            "n_features",
            "weight_abs_sum",
            "features",
            "y_true",
            "time",
            "event",
        ]
    )


def projection_performance_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = [
        "fold",
        "modality",
        "task",
        "projection",
        "n_input_features",
        "n_components",
        "n_features",
        "n_test",
        "status",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "roc_auc_ovr",
        "r2",
        "mae",
        "rmse",
        "c_index",
    ]
    return frame.reindex(columns=ordered_existing_columns(frame, base_columns))


def performance_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = [
        "fold",
        "modality",
        "task",
        "n_features",
        "n_test",
        "status",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "roc_auc_ovr",
        "r2",
        "mae",
        "rmse",
        "c_index",
    ]
    return frame.reindex(columns=ordered_existing_columns(frame, base_columns))


def prediction_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = ["id", "fold", "modality", "y_true", "prediction", "risk", "time", "event"]
    return frame.reindex(columns=ordered_existing_columns(frame, base_columns))


def projection_prediction_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = ["id", "fold", "modality", "projection", "y_true", "prediction", "risk", "time", "event"]
    return frame.reindex(columns=ordered_existing_columns(frame, base_columns))


def projection_application_rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    base_columns = ["id", "fold", "signature_fold", "modality", "projection", "n_components", "features"]
    return frame.reindex(columns=ordered_existing_columns(frame, [*base_columns, *component_columns_from_frame(frame)]))


def component_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("component_")]
    return sorted(columns, key=component_column_sort_key) or ["component_1"]


def component_column_sort_key(column: str) -> tuple[int, str]:
    suffix = str(column).removeprefix("component_")
    try:
        return int(suffix), str(column)
    except ValueError:
        return 10**9, str(column)


def ordered_existing_columns(frame: pd.DataFrame, base_columns: list[str]) -> list[str]:
    extras = [column for column in frame.columns if column not in base_columns]
    return [*base_columns, *extras]


def build_schema_audit(
    development: pd.DataFrame,
    external: pd.DataFrame | None,
    config: RunConfig,
    modalities: dict[str, list[str]],
) -> pd.DataFrame:
    feature_modalities: dict[str, list[str]] = {}
    for modality, columns in modalities.items():
        for column in columns:
            feature_modalities.setdefault(column, []).append(modality)

    required_columns: dict[str, str] = {}
    for column in metadata_columns(config):
        required_columns[column] = column_role(column, config)
    for column in feature_modalities:
        required_columns[column] = "feature"

    rows = []
    for dataset, frame in [("development", development), ("external", external)]:
        if frame is None:
            continue
        for column, role in sorted(required_columns.items()):
            present = column in frame.columns
            numeric_usable = bool(pd.to_numeric(frame[column], errors="coerce").notna().any()) if present else False
            issue = ""
            if dataset == "external" and role == "feature" and not present:
                issue = "missing_external_feature"
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "role": role,
                    "modalities": ";".join(feature_modalities.get(column, [])),
                    "present": present,
                    "numeric_usable": numeric_usable,
                    "issue": issue,
                }
            )
    return projection_application_rows_to_frame(rows)


def validate_schema_audit(schema_audit: pd.DataFrame) -> None:
    if schema_audit.empty or "issue" not in schema_audit.columns:
        return
    issues = schema_audit[schema_audit["issue"].astype(str).ne("")]
    if issues.empty:
        return
    missing = issues.loc[issues["issue"].eq("missing_external_feature"), "column"].tolist()
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"External data is missing required feature columns: {preview}{suffix}")


def build_final_refit_outputs(
    *,
    frame: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    modalities: dict[str, list[str]],
    stability: pd.DataFrame,
    config: RunConfig,
    external_data: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signature_rows: list[dict] = []
    parameter_rows: list[dict] = []
    score_rows: list[dict] = []
    correlation_rows: list[CorrelationRow] = []
    stability_lookup = {}
    if not stability.empty:
        for row in stability.itertuples(index=False):
            stability_lookup[(row.modality, row.feature)] = {
                "selection_probability": finite_float(getattr(row, "selection_probability", math.nan), math.nan),
                "stable": bool(getattr(row, "stable", False)),
            }

    final_config = replace(config, tune_elastic_net=False)
    for modality, columns in modalities.items():
        fitted = fit_selector(
            frame,
            y_train=y,
            columns=columns,
            modality=modality,
            fold="final_refit",
            config=final_config,
        )
        correlation_rows.extend(fitted.correlation_rows)
        for row in fitted.rows:
            record = asdict(row)
            record["model_status"] = fitted.model_status
            stable_info = stability_lookup.get((modality, row.feature), {})
            record["selection_probability"] = stable_info.get("selection_probability", math.nan)
            record["stable"] = stable_info.get("stable", False)
            signature_rows.append(record)
        parameters = composite_score_parameters(frame, fitted, config)
        for parameter in parameters:
            parameter_rows.append({"fold": "final_refit", "modality": modality, "task": config.task, **parameter})
        score_rows.extend(
            build_composite_scores(
                train=frame,
                test=frame,
                y_test=y,
                fitted=fitted,
                fold="final_refit_development",
                config=config,
                id_column=config.id_column,
                parameters=parameters,
            )
        )
        if external_data is not None:
            external_y = make_target(external_data, config)
            score_rows.extend(
                build_composite_scores(
                    train=frame,
                    test=external_data,
                    y_test=external_y,
                    fitted=fitted,
                    fold="final_refit_external",
                    config=config,
                    id_column=config.id_column,
                    parameters=parameters,
                )
            )
    return (
        pd.DataFrame(signature_rows).reindex(
            columns=[
                "modality",
                "fold",
                "feature",
                "rank",
                "relevance",
                "sign",
                "missing_fraction",
                "coefficient",
                "stage",
                "model_status",
                "selection_probability",
                "stable",
            ]
        ),
        composite_parameter_rows_to_frame(parameter_rows),
        composite_score_rows_to_frame(score_rows),
        correlation_rows_to_frame(correlation_rows),
    )


def build_composite_scores(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    fitted: FittedSelector,
    fold: str,
    config: RunConfig,
    id_column: str | None,
    parameters: list[dict] | None = None,
) -> list[dict]:
    parameters = parameters or composite_score_parameters(train, fitted, config)
    parameters = [row for row in parameters if row["feature"] in test.columns]
    if not parameters:
        return []

    features = [row["feature"] for row in parameters]
    test_numeric = test[features].apply(pd.to_numeric, errors="coerce")
    z_columns = []
    weight_vector = []
    for row in parameters:
        feature = row["feature"]
        median = finite_float(row["median"], 0.0)
        mean = finite_float(row["mean"], 0.0)
        std = finite_float(row["std"], 1.0) or 1.0
        z_columns.append(((test_numeric[feature].fillna(median).fillna(0.0) - mean) / std).to_numpy(dtype=float))
        weight_vector.append(finite_float(row["weight"], 0.0))
    raw_scores = np.column_stack(z_columns).dot(np.asarray(weight_vector, dtype=float))
    denominator = float(np.sum(np.abs(weight_vector))) or 1.0
    scores = raw_scores / denominator
    ids = test[id_column].tolist() if id_column and id_column in test.columns else list(test.index)
    outcome_rows = composite_outcome_rows(y_test, config)

    rows = []
    for idx, (row_id, raw_score, score) in enumerate(zip(ids, raw_scores, scores, strict=False)):
        row = {
            "id": row_id,
            "fold": fold,
            "modality": fitted.modality,
            "task": config.task,
            "score_type": "selected_feature_composite",
            "composite_score": float(score),
            "raw_composite_score": float(raw_score),
            "n_features": len(features),
            "weight_abs_sum": denominator,
            "features": ";".join(features),
        }
        row.update(outcome_rows[idx])
        rows.append(row)
    return rows


def apply_composite_score_parameters(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    id_column: str | None = None,
    fold: str = "applied",
) -> pd.DataFrame:
    required = {"feature", "median", "mean", "std", "weight"}
    missing_columns = sorted(required - set(parameters.columns))
    if missing_columns:
        raise ValueError(f"Composite parameter table is missing required columns: {', '.join(missing_columns)}.")

    group_columns = [column for column in ["fold", "modality", "task"] if column in parameters.columns]
    if not group_columns:
        parameter_groups = [(("", "all", "unknown"), parameters)]
    else:
        parameter_groups = []
        for key, group in parameters.groupby(group_columns, dropna=False):
            values = key if isinstance(key, tuple) else (key,)
            lookup = dict(zip(group_columns, values, strict=False))
            parameter_groups.append(
                (
                    (
                        str(lookup.get("fold", "")),
                        str(lookup.get("modality", "all")),
                        str(lookup.get("task", "unknown")),
                    ),
                    group,
                )
            )

    missing_features = sorted(
        {
            str(feature)
            for _, group in parameter_groups
            for feature in group["feature"].astype(str)
            if str(feature) not in frame.columns
        }
    )
    if missing_features:
        preview = ", ".join(missing_features[:10])
        suffix = "..." if len(missing_features) > 10 else ""
        raise ValueError(f"Input data is missing features required by the signature: {preview}{suffix}")

    ids = frame[id_column].tolist() if id_column and id_column in frame.columns else list(frame.index)
    rows: list[dict] = []
    for (signature_fold, modality, task), group in parameter_groups:
        records = group.to_dict(orient="records")
        features = [str(row["feature"]) for row in records]
        numeric = frame[features].apply(pd.to_numeric, errors="coerce")
        z_columns = []
        weights = []
        for row in records:
            feature = str(row["feature"])
            median = finite_float(row["median"], 0.0)
            mean = finite_float(row["mean"], 0.0)
            std = finite_float(row["std"], 1.0) or 1.0
            weight = finite_float(row["weight"], 0.0)
            z_columns.append(((numeric[feature].fillna(median).fillna(0.0) - mean) / std).to_numpy(dtype=float))
            weights.append(weight)
        raw_scores = np.column_stack(z_columns).dot(np.asarray(weights, dtype=float))
        denominator = float(np.sum(np.abs(weights))) or 1.0
        scores = raw_scores / denominator
        for row_id, raw_score, score in zip(ids, raw_scores, scores, strict=False):
            rows.append(
                {
                    "id": row_id,
                    "fold": fold,
                    "signature_fold": signature_fold,
                    "modality": modality,
                    "task": task,
                    "score_type": "selected_feature_composite",
                    "composite_score": float(score),
                    "raw_composite_score": float(raw_score),
                    "n_features": len(features),
                    "weight_abs_sum": denominator,
                    "features": ";".join(features),
                }
            )
    return pd.DataFrame(rows).reindex(
        columns=[
            "id",
            "fold",
            "signature_fold",
            "modality",
            "task",
            "score_type",
            "composite_score",
            "raw_composite_score",
            "n_features",
            "weight_abs_sum",
            "features",
        ]
    )


def apply_projection_parameters(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    id_column: str | None = None,
    fold: str = "applied_projection",
) -> pd.DataFrame:
    required = {"feature", "component", "impute_median", "scale_mean", "scale_std", "weight"}
    missing_columns = sorted(required - set(parameters.columns))
    if missing_columns:
        raise ValueError(f"Projection parameter table is missing required columns: {', '.join(missing_columns)}.")

    group_columns = [column for column in ["fold", "modality", "projection"] if column in parameters.columns]
    if not group_columns:
        parameter_groups = [(("", "all", "projection"), parameters)]
    else:
        parameter_groups = []
        for key, group in parameters.groupby(group_columns, dropna=False):
            values = key if isinstance(key, tuple) else (key,)
            lookup = dict(zip(group_columns, values, strict=False))
            parameter_groups.append(
                (
                    (
                        str(lookup.get("fold", "")),
                        str(lookup.get("modality", "all")),
                        str(lookup.get("projection", "projection")),
                    ),
                    group,
                )
            )

    missing_features = sorted(
        {
            str(feature)
            for _, group in parameter_groups
            for feature in group["feature"].astype(str)
            if str(feature) not in frame.columns
        }
    )
    if missing_features:
        preview = ", ".join(missing_features[:10])
        suffix = "..." if len(missing_features) > 10 else ""
        raise ValueError(f"Input data is missing features required by the projection signature: {preview}{suffix}")

    ids = frame[id_column].tolist() if id_column and id_column in frame.columns else list(frame.index)
    rows: list[dict] = []
    for (signature_fold, modality, projection), group in parameter_groups:
        group = group.copy()
        group["component"] = pd.to_numeric(group["component"], errors="coerce").astype(int)
        components = sorted(group["component"].unique().tolist())
        features = list(dict.fromkeys(group["feature"].astype(str).tolist()))
        numeric = frame[features].apply(pd.to_numeric, errors="coerce")
        component_scores: dict[int, np.ndarray] = {}
        for component in components:
            component_rows = group[group["component"].eq(component)]
            z_columns = []
            weights = []
            for parameter in component_rows.to_dict(orient="records"):
                feature = str(parameter["feature"])
                median = finite_float(parameter["impute_median"], 0.0)
                mean = finite_float(parameter["scale_mean"], 0.0)
                std = finite_float(parameter["scale_std"], 1.0) or 1.0
                weight = finite_float(parameter["weight"], 0.0)
                z_columns.append(((numeric[feature].fillna(median).fillna(0.0) - mean) / std).to_numpy(dtype=float))
                weights.append(weight)
            component_scores[component] = np.column_stack(z_columns).dot(np.asarray(weights, dtype=float))

        for row_index, row_id in enumerate(ids):
            row = {
                "id": row_id,
                "fold": fold,
                "signature_fold": signature_fold,
                "modality": modality,
                "projection": projection,
                "n_components": len(components),
                "features": ";".join(features),
            }
            for component in components:
                row[f"component_{component}"] = float(component_scores[component][row_index])
            rows.append(row)
    return pd.DataFrame(rows)


def composite_score_parameters(train: pd.DataFrame, fitted: FittedSelector, config: RunConfig) -> list[dict]:
    features = [feature for feature in fitted.selected_features if feature in train.columns]
    if not features:
        return []
    weights = composite_feature_weights(fitted.rows, features, config)
    if not weights:
        return []
    train_numeric = train[features].apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_imputed = train_numeric.fillna(medians).fillna(0.0)
    means = train_imputed.mean(axis=0)
    stds = train_imputed.std(axis=0, ddof=0).replace(0, 1).fillna(1.0)
    parameters = []
    for feature in features:
        parameters.append(
            {
                "feature": feature,
                "median": finite_float(medians.get(feature), 0.0),
                "mean": finite_float(means.get(feature), 0.0),
                "std": finite_float(stds.get(feature), 1.0) or 1.0,
                "weight": finite_float(weights.get(feature), 0.0),
            }
        )
    return parameters


def composite_feature_weights(rows: list[SelectionRow], features: list[str], config: RunConfig) -> dict[str, float]:
    by_feature = {row.feature: row for row in rows}
    weights = {}
    for feature in features:
        row = by_feature.get(feature)
        if row is None:
            continue
        coefficient = finite_float(row.coefficient, math.nan)
        if math.isfinite(coefficient) and abs(coefficient) > 1e-12:
            if config.task in {"binary", "multiclass"}:
                weights[feature] = float((row.sign or 1) * abs(coefficient))
            else:
                weights[feature] = float(coefficient)
            continue
        relevance = finite_float(row.relevance, 0.0)
        fallback = relevance if relevance > 0 else 1.0
        weights[feature] = float((row.sign or 1) * fallback)
    return weights


def composite_outcome_rows(y: pd.Series | pd.DataFrame, config: RunConfig) -> list[dict]:
    if config.task in {"binary", "multiclass", "regression"}:
        values = pd.Series(y).reset_index(drop=True)
        return [{"y_true": value} for value in values]
    target = pd.DataFrame(y).reset_index(drop=True)
    return [
        {"time": row.time, "event": row.event}
        for row in target[["time", "event"]].itertuples(index=False)
    ]


def split_holdout_groups(
    frame: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    if not config.holdout_groups:
        return frame, None, {"enabled": False, "group_column": config.group_column, "groups": [], "rows": 0}
    assert config.group_column is not None
    if config.group_column not in frame.columns:
        raise ValueError(f"group_column={config.group_column} is not present in the input table.")

    requested = {str(value) for value in config.holdout_groups}
    groups = frame[config.group_column].astype(str)
    present = set(groups.dropna().unique())
    missing = sorted(requested - present)
    if missing:
        raise ValueError(f"holdout_groups not found in group_column={config.group_column}: {', '.join(missing)}.")

    holdout_mask = groups.isin(requested)
    holdout = frame.loc[holdout_mask].reset_index(drop=True)
    development = frame.loc[~holdout_mask].reset_index(drop=True)
    if development.empty:
        raise ValueError("holdout_groups removed all development rows.")
    if holdout.empty:
        raise ValueError("holdout_groups did not select any holdout rows.")
    summary = {
        "enabled": True,
        "group_column": config.group_column,
        "groups": sorted(requested),
        "rows": int(len(holdout)),
        "development_rows_raw": int(len(development)),
    }
    return development, holdout, summary


def audit_and_filter_samples(frame: pd.DataFrame, config: RunConfig, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    kept_indices = []
    id_values = frame[config.id_column] if config.id_column and config.id_column in frame.columns else None
    for row_number, (index, row) in enumerate(frame.iterrows()):
        reason = sample_exclusion_reason(row, config)
        retained = reason == ""
        if retained:
            kept_indices.append(index)
        rows.append(
            {
                "dataset": dataset,
                "row_number": row_number,
                "original_index": index,
                "id": id_values.loc[index] if id_values is not None else index,
                "status": "retained" if retained else "dropped",
                "reason": reason,
            }
        )
    return frame.loc[kept_indices].reset_index(drop=True), pd.DataFrame(rows)


def sample_exclusion_reason(row: pd.Series, config: RunConfig) -> str:
    if config.task in {"binary", "multiclass"}:
        assert config.target_column is not None
        value = row.get(config.target_column)
        if is_missing_scalar(value):
            return "missing_outcome"
        return ""
    if config.task == "regression":
        assert config.target_column is not None
        value = to_finite_number(row.get(config.target_column))
        if value is None:
            return "missing_or_non_numeric_outcome"
        return ""

    assert config.time_column is not None and config.event_column is not None
    time_value = to_finite_number(row.get(config.time_column))
    if time_value is None or time_value <= 0:
        return "missing_or_non_positive_time"
    if is_missing_scalar(row.get(config.event_column)):
        return "missing_event"
    return ""


def is_missing_scalar(value: object) -> bool:
    if pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def to_finite_number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    number = float(numeric)
    return number if math.isfinite(number) else None


def selected_feature_frequency(selected_features: pd.DataFrame, stability_selection: pd.DataFrame) -> pd.DataFrame:
    if selected_features.empty and stability_selection.empty:
        return pd.DataFrame(
            columns=[
                "modality",
                "feature",
                "selected_folds",
                "mean_abs_coefficient",
                "mean_relevance",
                "selection_probability",
                "stable",
            ]
        )
    rows = []
    if not selected_features.empty:
        grouped = selected_features.assign(abs_coefficient=selected_features["coefficient"].abs()).groupby(
            ["modality", "feature"], dropna=False
        )
        for (modality, feature), group in grouped:
            rows.append(
                {
                    "modality": modality,
                    "feature": feature,
                    "selected_folds": int(group["fold"].nunique()),
                    "mean_abs_coefficient": finite_float(group["abs_coefficient"].mean(), math.nan),
                    "mean_relevance": finite_float(group["relevance"].mean(), math.nan),
                }
            )
    frequency = pd.DataFrame(rows)
    if not stability_selection.empty:
        stability_cols = ["modality", "feature", "selection_probability", "stable"]
        stability = stability_selection[stability_cols].copy()
        if frequency.empty:
            frequency = stability.assign(
                selected_folds=0,
                mean_abs_coefficient=math.nan,
                mean_relevance=math.nan,
            )
        else:
            frequency = frequency.merge(stability, on=["modality", "feature"], how="outer")
    if "selected_folds" not in frequency.columns:
        frequency["selected_folds"] = 0
    if "selection_probability" not in frequency.columns:
        frequency["selection_probability"] = math.nan
    if "stable" not in frequency.columns:
        frequency["stable"] = False
    frequency["selected_folds"] = frequency["selected_folds"].fillna(0).astype(int)
    return frequency.sort_values(
        ["modality", "selection_probability", "selected_folds", "feature"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_dependency_audit() -> pd.DataFrame:
    rows = []
    required = package_requirements("radselect")
    if not required:
        required = ["numpy", "pandas", "scikit-learn"]
    for requirement in required:
        parsed = parse_requirement_record(requirement)
        name = parsed["package"]
        normalized = normalize_package_name(name)
        rows.append(
            {
                "requirement": requirement,
                "package": name,
                "normalized_package": normalized,
                "scope": parsed["scope"],
                "extra": parsed["extra"],
                "blocked_llm_or_openai_dependency": normalized in BLOCKED_RUNTIME_DEPENDENCIES,
            }
        )
    existing = {(row["normalized_package"], row["scope"]) for row in rows}
    for requirement, scope, extra in [
        ("matplotlib", "optional_extra", "reports"),
        ("seaborn", "optional_extra", "reports"),
        ("lifelines", "optional_extra", "survival"),
        ("pytest", "dev_extra", "dev"),
        ("ruff", "dev_extra", "dev"),
    ]:
        normalized = normalize_package_name(requirement)
        if (normalized, scope) in existing:
            continue
        rows.append(
            {
                "requirement": requirement,
                "package": requirement,
                "normalized_package": normalized,
                "scope": scope,
                "extra": extra,
                "blocked_llm_or_openai_dependency": normalized in BLOCKED_RUNTIME_DEPENDENCIES,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "requirement",
            "package",
            "normalized_package",
            "scope",
            "extra",
            "blocked_llm_or_openai_dependency",
        ],
    )


def package_requirements(package_name: str) -> list[str]:
    try:
        return list(importlib.metadata.requires(package_name) or [])
    except importlib.metadata.PackageNotFoundError:
        return []


def requirement_name(requirement: str) -> str:
    text = str(requirement).strip()
    for separator in [";", "[", " ", "<", ">", "=", "!", "~"]:
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.strip()


def parse_requirement_record(requirement: str) -> dict:
    text = str(requirement).strip()
    requirement_text, marker = split_requirement_marker(text)
    extra = requirement_extra(marker)
    if extra == "dev":
        scope = "dev_extra"
    elif extra:
        scope = "optional_extra"
    else:
        scope = "runtime_required"
    return {
        "package": requirement_name(requirement_text),
        "scope": scope,
        "extra": extra,
    }


def split_requirement_marker(requirement: str) -> tuple[str, str]:
    if ";" not in requirement:
        return requirement, ""
    left, right = requirement.split(";", 1)
    return left.strip(), right.strip()


def requirement_extra(marker: str) -> str:
    match = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", marker)
    return match.group(1) if match else ""


def normalize_package_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def dependency_audit_summary(dependency_audit: pd.DataFrame) -> dict:
    if dependency_audit.empty:
        return {
            "rows": 0,
            "runtime_required_dependencies": 0,
            "blocked_runtime_dependencies": 0,
            "blocked_dependencies": [],
        }
    blocked = dependency_audit["blocked_llm_or_openai_dependency"].astype(bool)
    runtime = dependency_audit["scope"].astype(str).eq("runtime_required")
    blocked_runtime = dependency_audit[blocked & runtime]
    return {
        "rows": int(len(dependency_audit)),
        "runtime_required_dependencies": int(runtime.sum()),
        "blocked_runtime_dependencies": int(len(blocked_runtime)),
        "blocked_dependencies": sorted(blocked_runtime["normalized_package"].astype(str).unique().tolist()),
    }


def build_provenance(manifest: dict) -> dict:
    dependency_audit = build_dependency_audit()
    return {
        "package": "radselect",
        "package_version": package_version("radselect"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            name: package_version(name)
            for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "lifelines"]
        },
        "dependency_audit": dependency_audit_summary(dependency_audit),
        "runtime_note": "radselect operates on already-extracted tabular features and has no LLM/OpenAI runtime dependency.",
        "manifest": manifest,
    }


def write_output_manifest(outdir: str | Path) -> Path:
    out = Path(outdir)
    manifest = {
        "schema_version": 1,
        "artifacts": output_artifacts(out),
    }
    path = out / "output_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def output_artifacts(outdir: Path) -> list[dict]:
    artifacts = []
    if not outdir.exists():
        return artifacts
    for path in sorted(outdir.rglob("*")):
        if path.is_dir() or path.name == "output_manifest.json":
            continue
        relative_path = path.relative_to(outdir).as_posix()
        record = {
            "path": relative_path,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() == ".csv":
            try:
                table = pd.read_csv(path)
            except Exception:
                table = None
            if table is not None:
                record["rows"] = int(len(table))
                record["columns"] = int(len(table.columns))
        artifacts.append(record)
    return artifacts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def metadata_columns(config: RunConfig) -> set[str]:
    return {
        column
        for column in [
            config.id_column,
            config.target_column,
            config.time_column,
            config.event_column,
            config.group_column,
        ]
        if column
    }


def build_column_audit(frame: pd.DataFrame, config: RunConfig, modalities: dict[str, list[str]]) -> pd.DataFrame:
    modality_by_column: dict[str, list[str]] = {}
    for modality, columns in modalities.items():
        for column in columns:
            modality_by_column.setdefault(column, []).append(modality)
    metadata = metadata_columns(config)
    rows = []
    for column in frame.columns:
        included = column in modality_by_column
        role = column_role(column, config)
        if role == "other" and included:
            role = "feature"
        leakage_risk, reason = leakage_risk_reason(column, role, included, config)
        rows.append(
            {
                "column": column,
                "role": role,
                "included_as_candidate": included,
                "modalities": ";".join(modality_by_column.get(column, [])),
                "numeric_usable": bool(pd.to_numeric(frame[column], errors="coerce").notna().any()),
                "metadata_protected": column in metadata,
                "leakage_risk": leakage_risk,
                "leakage_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def column_role(column: str, config: RunConfig) -> str:
    if column == config.id_column:
        return "id"
    if column == config.target_column:
        return "outcome"
    if column == config.time_column:
        return "time"
    if column == config.event_column:
        return "event"
    if column == config.group_column:
        return "group"
    return "feature" if column in (config.feature_columns or []) else "other"


def leakage_risk_reason(column: str, role: str, included: bool, config: RunConfig) -> tuple[bool, str]:
    if role in {"id", "outcome", "time", "event", "group"}:
        return False, "metadata column protected from feature use"
    if not included:
        return False, ""
    lower = column.lower()
    protected_tokens = [
        str(token).lower()
        for token in [
            config.target_column,
            config.time_column,
            config.event_column,
        ]
        if token
    ]
    generic_tokens = ["outcome", "label", "target", "event", "death", "mortality", "mace", "followup"]
    for token in [*protected_tokens, *generic_tokens]:
        if token and token in lower:
            return True, f"candidate feature name contains outcome-like token '{token}'"
    return False, ""


def combine_allowed_sets(*allowed_sets: set[str] | None) -> set[str] | None:
    active = [values for values in allowed_sets if values is not None]
    if not active:
        return None
    allowed = set(active[0])
    for values in active[1:]:
        allowed &= values
    return allowed


def load_feature_metadata_filter(
    config: RunConfig,
) -> tuple[set[str] | None, set[str], FeatureMetadataSummary, pd.DataFrame]:
    path = config.feature_metadata_csv
    audit_columns = [
        "feature",
        "ibsi_compliant",
        "filter_decision",
        "reason",
    ]
    if path is None:
        return (
            None,
            set(),
            FeatureMetadataSummary("not_provided", None, 0, 0, 0, "No feature metadata CSV was provided."),
            pd.DataFrame(columns=audit_columns),
        )

    table = pd.read_csv(path)
    feature_col = first_existing(table.columns, ["feature", "feature_name", "variable", "column", "name"])
    if feature_col is None:
        raise ValueError("feature metadata CSV needs a feature-name column.")
    ibsi_col = first_existing(
        table.columns,
        ["ibsi_compliant", "ibsi", "ibsi_compliance", "compliant", "radiomics_compliant"],
    )
    if ibsi_col is None and config.require_ibsi_compliant:
        raise ValueError("require_ibsi_compliant needs an IBSI compliance column in feature_metadata_csv.")

    features = table[feature_col].astype(str)
    if ibsi_col is None:
        audit = pd.DataFrame(
            {
                "feature": features,
                "ibsi_compliant": pd.NA,
                "filter_decision": "retained",
                "reason": "metadata provided without IBSI compliance column",
            }
        )
        summary = FeatureMetadataSummary(
            "provided_without_ibsi_column",
            str(path),
            int(len(audit)),
            0,
            0,
            "Feature metadata was recorded for provenance but no IBSI compliance column was found.",
        )
        return None, set(), summary, audit[audit_columns]

    compliant = table[ibsi_col].map(parse_bool)
    rejected_mask = compliant.ne(True) if config.require_ibsi_compliant else pd.Series(False, index=table.index)
    kept_mask = compliant.eq(True)
    if config.require_ibsi_compliant and config.ibsi_require_listed:
        allowed: set[str] | None = set(features[kept_mask])
    else:
        allowed = None
    rejected = set(features[rejected_mask])

    decisions = []
    reasons = []
    for value in compliant:
        if value is True:
            decisions.append("retained")
            reasons.append("IBSI compliant")
        elif config.require_ibsi_compliant:
            decisions.append("rejected")
            reasons.append("not marked IBSI compliant")
        else:
            decisions.append("retained")
            reasons.append("IBSI compliance recorded for audit only")
    audit = pd.DataFrame(
        {
            "feature": features,
            "ibsi_compliant": compliant,
            "filter_decision": decisions,
            "reason": reasons,
        }
    )
    status = "applied" if config.require_ibsi_compliant else "audit_only"
    details = (
        "Rejected features not marked IBSI compliant; "
        f"{'unlisted features rejected' if config.ibsi_require_listed else 'unlisted features retained'}."
        if config.require_ibsi_compliant
        else "Feature metadata was recorded for provenance; IBSI compliance was not enforced."
    )
    summary = FeatureMetadataSummary(
        status,
        str(path),
        int(len(audit)),
        int(kept_mask.sum()),
        int(len(rejected)),
        details,
    )
    return allowed, rejected, summary, audit[audit_columns]


def build_modalities(
    frame: pd.DataFrame,
    config: RunConfig,
    allowed_features: set[str] | None,
    rejected_features: set[str] | None,
) -> dict[str, list[str]]:
    base = filtered_candidate_features(frame, config, allowed_features, rejected_features)

    modalities: dict[str, list[str]] = {}
    radiomics = [column for column in config.radiomics_columns if column in base]
    clinical = [column for column in config.clinical_columns if column in base]
    if radiomics:
        modalities["radiomics"] = radiomics
    if clinical:
        modalities["clinical"] = clinical
    if radiomics or clinical:
        combined = list(dict.fromkeys([*radiomics, *clinical]))
        if combined:
            modalities["combined"] = combined
    for name, columns in config.domains.items():
        domain_columns = [column for column in columns if column in base]
        if domain_columns:
            modalities[name] = domain_columns
    if not modalities:
        modalities["all"] = base
    return modalities


def initial_candidate_features(frame: pd.DataFrame, config: RunConfig) -> list[str]:
    metadata = metadata_columns(config)
    if config.feature_columns:
        columns = [column for column in config.feature_columns if column in frame.columns and column not in metadata]
    else:
        columns = [
            column
            for column in frame.columns
            if column not in metadata and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
    return list(dict.fromkeys(columns))


def filtered_candidate_features(
    frame: pd.DataFrame,
    config: RunConfig,
    allowed_features: set[str] | None,
    rejected_features: set[str] | None,
) -> list[str]:
    base = initial_candidate_features(frame, config)
    if allowed_features is not None:
        base = [column for column in base if column in allowed_features]
    if rejected_features:
        base = [column for column in base if column not in rejected_features]
    return base


def build_modality_audit(
    frame: pd.DataFrame,
    config: RunConfig,
    modalities: dict[str, list[str]],
    allowed_features: set[str] | None,
    rejected_features: set[str] | None,
) -> pd.DataFrame:
    columns = [
        "modality",
        "feature",
        "source",
        "requested_in_config",
        "included_in_modality",
        "present_in_development",
        "numeric_usable",
        "passed_metadata_and_robustness_filters",
        "rejected_by_metadata_or_robustness",
        "reason",
    ]
    initial = set(initial_candidate_features(frame, config))
    filtered = set(filtered_candidate_features(frame, config, allowed_features, rejected_features))
    rejected = rejected_features or set()
    metadata = metadata_columns(config)
    definitions: dict[str, tuple[str, list[str]]] = {}
    if config.radiomics_columns:
        definitions["radiomics"] = ("radiomics_columns", list(dict.fromkeys(config.radiomics_columns)))
    if config.clinical_columns:
        definitions["clinical"] = ("clinical_columns", list(dict.fromkeys(config.clinical_columns)))
    if config.radiomics_columns or config.clinical_columns:
        definitions["combined"] = (
            "generated_radiomics_plus_clinical",
            list(dict.fromkeys([*config.radiomics_columns, *config.clinical_columns])),
        )
    for name, domain_columns in config.domains.items():
        definitions[name] = (f"domain:{name}", list(dict.fromkeys(domain_columns)))
    if not definitions and "all" in modalities:
        definitions["all"] = (
            "fallback_all_numeric_candidates",
            list(dict.fromkeys(initial_candidate_features(frame, config))),
        )

    rows = []
    for modality in list(dict.fromkeys([*definitions.keys(), *modalities.keys()])):
        source, requested = definitions.get(modality, ("derived_modality", modalities.get(modality, [])))
        requested_features = set(requested)
        included_features = set(modalities.get(modality, []))
        features = list(dict.fromkeys([*requested, *modalities.get(modality, [])]))
        for feature in features:
            present = feature in frame.columns
            numeric = bool(present and pd.to_numeric(frame[feature], errors="coerce").notna().any())
            included = feature in included_features
            rejected_by_filter = feature in rejected
            passes_filters = bool(feature in filtered)
            if included:
                reason = "included"
            elif rejected_by_filter:
                reason = "rejected_by_metadata_or_robustness"
            elif allowed_features is not None and feature not in allowed_features:
                reason = "not_allowed_by_required_metadata_or_robustness_listing"
            elif not present:
                reason = "missing_from_development_table"
            elif feature in metadata:
                reason = "metadata_column_protected"
            elif not numeric:
                reason = "non_numeric_or_all_missing"
            elif feature not in initial:
                reason = "not_in_initial_candidate_pool"
            elif feature not in filtered:
                reason = "not_in_filtered_candidate_pool"
            else:
                reason = "not_in_modality_after_domain_resolution"
            rows.append(
                {
                    "modality": modality,
                    "feature": feature,
                    "source": source,
                    "requested_in_config": feature in requested_features,
                    "included_in_modality": included,
                    "present_in_development": present,
                    "numeric_usable": numeric,
                    "passed_metadata_and_robustness_filters": passes_filters,
                    "rejected_by_metadata_or_robustness": rejected_by_filter,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def make_target(frame: pd.DataFrame, config: RunConfig) -> pd.Series | pd.DataFrame:
    if config.task in {"binary", "multiclass"}:
        assert config.target_column is not None
        y = frame[config.target_column]
        if config.task == "binary":
            numeric = pd.to_numeric(y, errors="coerce")
            if numeric.notna().all():
                return numeric.astype(int)
        return y.astype(str)
    if config.task == "regression":
        assert config.target_column is not None
        return pd.to_numeric(frame[config.target_column], errors="coerce")
    assert config.time_column is not None and config.event_column is not None
    return pd.DataFrame(
        {
            "time": pd.to_numeric(frame[config.time_column], errors="coerce"),
            "event": frame[config.event_column],
        },
        index=frame.index,
    )


def target_subset(y: pd.Series | pd.DataFrame, idx: np.ndarray) -> pd.Series | pd.DataFrame:
    return y.iloc[idx].reset_index(drop=True)


def iter_validation_splits(
    frame: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: RunConfig,
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    n = len(frame)
    if n < 2:
        raise ValueError("At least 2 development rows are required for outer validation.")
    indices = np.arange(n)
    if config.task in {"binary", "multiclass"}:
        labels = pd.Series(y).astype(str)
        class_counts = labels.value_counts()
        if len(class_counts) < 2:
            raise ValueError("Classification tasks require at least 2 outcome classes for outer validation.")
        min_class = int(class_counts.min())
        if min_class < 2:
            raise ValueError("Each outcome class needs at least 2 rows for stratified outer validation.")
    else:
        labels = pd.Series(dtype=str)
        min_class = 0

    if config.group_column and config.group_column in frame.columns:
        groups = frame[config.group_column].astype(str)
        n_groups = groups.nunique()
        if n_groups < 2:
            raise ValueError(f"group_column={config.group_column} has fewer than 2 groups.")
        splitter = GroupKFold(n_splits=min(config.outer_splits, n_groups))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, groups=groups), start=1):
            yield f"group_fold_{fold}", train_idx, test_idx
        return

    if config.task in {"binary", "multiclass"}:
        n_splits = min(config.outer_splits, min_class)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, labels), start=1):
            yield f"fold_{fold}", train_idx, test_idx
        return

    splitter = KFold(n_splits=min(config.outer_splits, n), shuffle=True, random_state=config.random_state)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(indices), start=1):
        yield f"fold_{fold}", train_idx, test_idx


def load_robustness_filter(config: RunConfig) -> tuple[set[str] | None, set[str], RobustnessSummary, pd.DataFrame]:
    path = config.robustness_csv
    audit_columns = [
        "feature",
        "filter_decision",
        "reason",
        "min_robustness_score",
        "threshold",
        "robustness_columns",
        "bool_column",
    ]
    if path is None:
        return (
            None,
            set(),
            RobustnessSummary("not_provided", None, 0, 0, "No robustness CSV was provided."),
            pd.DataFrame(columns=audit_columns),
        )
    table = pd.read_csv(path)
    feature_col = first_existing(table.columns, ["feature", "feature_name", "variable", "column", "name"])
    if feature_col is None:
        raise ValueError("robustness CSV needs a feature-name column.")

    bool_col = first_existing(table.columns, ["robust", "keep", "pass", "passes", "include"])
    score_cols = [
        column
        for column in [
            "icc",
            "test_retest_icc",
            "segmentation_icc",
            "acquisition_icc",
            "robustness",
            "ccc",
        ]
        if column in table.columns
    ]
    min_scores = pd.Series(np.nan, index=table.index)
    threshold = pd.Series(pd.NA, index=table.index)
    if bool_col is not None:
        parsed = table[bool_col].map(parse_bool)
        keep = parsed.fillna(False).astype(bool)
        reasons = [
            "pass column marked feature robust" if value is True else "pass column did not mark feature robust"
            for value in parsed
        ]
    elif score_cols:
        scores = table[score_cols].apply(pd.to_numeric, errors="coerce")
        min_scores = scores.min(axis=1)
        threshold = pd.Series(float(config.robustness_min_icc), index=table.index)
        keep = min_scores >= config.robustness_min_icc
        reasons = []
        for score, retained in zip(min_scores, keep, strict=False):
            if pd.isna(score):
                reasons.append("missing numeric robustness score")
            elif retained:
                reasons.append("all recorded robustness metrics met threshold")
            else:
                reasons.append("one or more recorded robustness metrics fell below threshold")
    else:
        raise ValueError("robustness CSV needs a keep/pass column or numeric ICC/robustness column.")

    features = table[feature_col].astype(str)
    kept = set(features[keep])
    rejected = set(features[~keep])
    if config.robustness_require_listed:
        allowed = kept
    else:
        allowed = None
    audit = pd.DataFrame(
        {
            "feature": features,
            "filter_decision": np.where(keep, "retained", "rejected"),
            "reason": reasons,
            "min_robustness_score": min_scores,
            "threshold": threshold,
            "robustness_columns": ";".join(score_cols),
            "bool_column": bool_col or "",
        }
    )
    return allowed, rejected, RobustnessSummary(
        status="applied",
        path=str(path),
        kept_features=len(kept),
        rejected_features=len(rejected),
        details=(
            f"Applied robustness filter using {feature_col}; "
            f"{'listed features only' if config.robustness_require_listed else 'unlisted features retained'}."
        ),
    ), audit[audit_columns]


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def parse_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "keep", "include"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "drop", "exclude"}:
        return False
    return None


def fit_selector(
    train: pd.DataFrame,
    *,
    y_train: pd.Series | pd.DataFrame,
    columns: list[str],
    modality: str,
    fold: str,
    config: RunConfig,
) -> FittedSelector:
    tuning_rows: list[TuningRow] = []
    if config.tune_elastic_net:
        tuned_config, tuning_rows = tune_elastic_net_parameters(
            train=train,
            y_train=y_train,
            columns=columns,
            modality=modality,
            outer_fold=fold,
            config=config,
        )
        config = tuned_config

    numeric = numeric_feature_frame(train, columns)
    rows: list[SelectionRow] = []
    drops: list[DropRow] = []
    correlation_rows: list[CorrelationRow] = []
    if numeric.empty:
        return FittedSelector(modality, [], rows, drops, tuning_rows, correlation_rows, "no_numeric_features")

    missing = numeric.isna().mean()
    keep_missing = missing[missing <= config.max_missing].index.tolist()
    for feature, value in missing.items():
        if feature not in keep_missing:
            drops.append(DropRow(modality, fold, feature, "missingness", float(value)))
    numeric = numeric[keep_missing]
    if numeric.empty:
        return FittedSelector(
            modality,
            [],
            rows,
            drops,
            tuning_rows,
            correlation_rows,
            "all_features_failed_missingness",
        )

    variances = numeric.var(skipna=True)
    uniques = numeric.nunique(dropna=True)
    keep_variance = [
        column
        for column in numeric.columns
        if variances.get(column, 0.0) > config.min_variance and uniques.get(column, 0) >= config.min_unique
    ]
    for feature in numeric.columns:
        if feature not in keep_variance:
            drops.append(DropRow(modality, fold, feature, "near_zero_variance", float(variances.get(feature, 0.0))))
    numeric = numeric[keep_variance]
    if numeric.empty:
        return FittedSelector(
            modality,
            [],
            rows,
            drops,
            tuning_rows,
            correlation_rows,
            "all_features_failed_variance",
        )

    relevance = rank_features(numeric, y_train, config)
    ranked = relevance.sort_values(["relevance", "feature"], ascending=[False, True]).head(config.top_k)
    ranked_features = ranked["feature"].tolist()
    kept_after_corr, corr_drops, correlation_rows = correlation_filter(
        numeric[ranked_features],
        ranked.set_index("feature")["relevance"].to_dict(),
        modality,
        fold,
        config,
    )
    drops.extend(corr_drops)
    selected, coefficients, model_status = elastic_net_features(
        numeric[kept_after_corr],
        y_train,
        config,
    )
    if not selected and kept_after_corr:
        selected = kept_after_corr[: min(3, len(kept_after_corr))]
        coefficients = {feature: math.nan for feature in selected}
        model_status = f"{model_status};fallback_top_relevance"

    rank_lookup = ranked.set_index("feature").to_dict(orient="index")
    for rank, feature in enumerate(selected, start=1):
        info = rank_lookup.get(feature, {})
        rows.append(
            SelectionRow(
                modality=modality,
                fold=fold,
                feature=feature,
                rank=rank,
                relevance=float(info.get("relevance", math.nan)),
                sign=int(info.get("sign", 0)),
                missing_fraction=float(missing.get(feature, math.nan)),
                coefficient=float(coefficients.get(feature, math.nan)),
                stage=f"elastic_net_after_{config.screening_method}",
            )
        )
    return FittedSelector(modality, selected, rows, drops, tuning_rows, correlation_rows, model_status)


def tune_elastic_net_parameters(
    *,
    train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    columns: list[str],
    modality: str,
    outer_fold: str,
    config: RunConfig,
) -> tuple[RunConfig, list[TuningRow]]:
    candidates = elastic_net_candidate_grid(config)
    if len(candidates) <= 1:
        return replace(config, tune_elastic_net=False), []

    inner_frame = train.reset_index(drop=True)
    inner_y = y_train.reset_index(drop=True)
    base_config = replace(config, tune_elastic_net=False, outer_splits=config.inner_splits)
    try:
        splits = list(iter_validation_splits(inner_frame, inner_y, base_config))
    except ValueError:
        return replace(config, tune_elastic_net=False), []
    if not splits:
        return replace(config, tune_elastic_net=False), []

    scored: list[dict] = []
    metric_name = validation_score_metric(config.task)
    for candidate_id, params in enumerate(candidates, start=1):
        candidate_config = replace(
            base_config,
            elastic_net_c=params["elastic_net_c"],
            elastic_net_alpha=params["elastic_net_alpha"],
            elastic_net_l1_ratio=params["elastic_net_l1_ratio"],
        )
        fold_scores = []
        for inner_fold, inner_train_idx, inner_test_idx in splits:
            fitted = fit_selector(
                inner_frame.iloc[inner_train_idx],
                y_train=target_subset(inner_y, inner_train_idx),
                columns=columns,
                modality=modality,
                fold=f"{outer_fold}:{inner_fold}",
                config=candidate_config,
            )
            validation = evaluate_selected_model(
                train=inner_frame.iloc[inner_train_idx],
                test=inner_frame.iloc[inner_test_idx],
                y_train=target_subset(inner_y, inner_train_idx),
                y_test=target_subset(inner_y, inner_test_idx),
                features=fitted.selected_features,
                modality=modality,
                fold=f"{outer_fold}:{inner_fold}",
                config=candidate_config,
                id_column=config.id_column,
            )
            score = validation_score(validation["performance"], config.task)
            if math.isfinite(score):
                fold_scores.append(score)
        mean_score = float(np.mean(fold_scores)) if fold_scores else math.nan
        scored.append({"candidate": f"candidate_{candidate_id}", **params, "mean_inner_score": mean_score})

    finite_scores = [row for row in scored if math.isfinite(row["mean_inner_score"])]
    if finite_scores:
        best = max(
            finite_scores,
            key=lambda row: (
                row["mean_inner_score"],
                -row["elastic_net_alpha"],
                row["elastic_net_c"],
                row["elastic_net_l1_ratio"],
            ),
        )
    else:
        best = {
            "candidate": "default",
            "elastic_net_c": config.elastic_net_c,
            "elastic_net_alpha": config.elastic_net_alpha,
            "elastic_net_l1_ratio": config.elastic_net_l1_ratio,
            "mean_inner_score": math.nan,
        }

    rows = [
        TuningRow(
            modality=modality,
            outer_fold=outer_fold,
            candidate=row["candidate"],
            elastic_net_c=float(row["elastic_net_c"]),
            elastic_net_alpha=float(row["elastic_net_alpha"]),
            elastic_net_l1_ratio=float(row["elastic_net_l1_ratio"]),
            mean_inner_score=float(row["mean_inner_score"]),
            selected=row["candidate"] == best["candidate"],
            metric=metric_name,
        )
        for row in scored
    ]
    tuned_config = replace(
        config,
        tune_elastic_net=False,
        elastic_net_c=float(best["elastic_net_c"]),
        elastic_net_alpha=float(best["elastic_net_alpha"]),
        elastic_net_l1_ratio=float(best["elastic_net_l1_ratio"]),
    )
    return tuned_config, rows


def elastic_net_candidate_grid(config: RunConfig) -> list[dict[str, float]]:
    l1_values = unique_floats(config.elastic_net_l1_ratio_grid or [config.elastic_net_l1_ratio])
    if config.task in {"binary", "multiclass"}:
        return [
            {
                "elastic_net_c": c_value,
                "elastic_net_alpha": config.elastic_net_alpha,
                "elastic_net_l1_ratio": l1_value,
            }
            for c_value in unique_floats(config.elastic_net_c_grid or [config.elastic_net_c])
            for l1_value in l1_values
        ]
    return [
        {
            "elastic_net_c": config.elastic_net_c,
            "elastic_net_alpha": alpha_value,
            "elastic_net_l1_ratio": l1_value,
        }
        for alpha_value in unique_floats(config.elastic_net_alpha_grid or [config.elastic_net_alpha])
        for l1_value in l1_values
    ]


def unique_floats(values: Iterable[float]) -> list[float]:
    unique: list[float] = []
    for value in values:
        number = float(value)
        if number not in unique:
            unique.append(number)
    return unique


def validation_score(performance_rows: list[dict], task: str) -> float:
    scores = []
    for row in performance_rows:
        if task == "binary":
            score = first_finite(row, ["roc_auc", "average_precision", "balanced_accuracy", "accuracy"])
        elif task == "multiclass":
            score = first_finite(row, ["roc_auc_ovr", "balanced_accuracy", "accuracy"])
        elif task == "regression":
            score = first_finite(row, ["r2"])
            if not math.isfinite(score):
                rmse = finite_float(row.get("rmse"), math.nan)
                mae = finite_float(row.get("mae"), math.nan)
                score = -rmse if math.isfinite(rmse) else -mae
        else:
            score = first_finite(row, ["c_index"])
        if math.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else math.nan


def validation_score_metric(task: str) -> str:
    if task == "binary":
        return "roc_auc|average_precision|balanced_accuracy"
    if task == "multiclass":
        return "roc_auc_ovr|balanced_accuracy"
    if task == "regression":
        return "r2|-rmse|-mae"
    return "c_index"


def first_finite(row: dict, keys: list[str]) -> float:
    for key in keys:
        value = finite_float(row.get(key), math.nan)
        if math.isfinite(value):
            return value
    return math.nan


def numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame[available].apply(pd.to_numeric, errors="coerce")


def imputed_array(frame: pd.DataFrame) -> np.ndarray:
    return SimpleImputer(strategy="median").fit_transform(frame)


def rank_features(frame: pd.DataFrame, y: pd.Series | pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    if config.screening_method == "mutual_info":
        return rank_features_mutual_info(frame, y, config)
    x = imputed_array(frame)
    rows = []
    for index, feature in enumerate(frame.columns):
        values = x[:, index]
        relevance, sign = feature_relevance(values, y, config)
        rows.append({"feature": feature, "relevance": relevance, "sign": sign})
    return pd.DataFrame(rows)


def rank_features_mutual_info(frame: pd.DataFrame, y: pd.Series | pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    x = imputed_array(frame)
    try:
        if config.task in {"binary", "multiclass"}:
            target = LabelEncoder().fit_transform(pd.Series(y).astype(str))
            scores = mutual_info_classif(
                x,
                target,
                discrete_features=False,
                n_neighbors=config.mutual_info_neighbors,
                random_state=config.random_state,
            )
        elif config.task == "regression":
            target = pd.to_numeric(pd.Series(y), errors="coerce").fillna(pd.Series(y).median()).to_numpy()
            scores = mutual_info_regression(
                x,
                target,
                discrete_features=False,
                n_neighbors=config.mutual_info_neighbors,
                random_state=config.random_state,
            )
        else:
            target_frame = pd.DataFrame(y)
            target = event_indicator(target_frame["event"], config).astype(int)
            scores = mutual_info_classif(
                x,
                target,
                discrete_features=False,
                n_neighbors=config.mutual_info_neighbors,
                random_state=config.random_state,
            )
    except Exception:
        fallback = replace(config, screening_method="univariate")
        return rank_features(frame, y, fallback)

    rows = []
    for index, feature in enumerate(frame.columns):
        _, sign = feature_relevance(x[:, index], y, config)
        rows.append({"feature": feature, "relevance": finite_float(scores[index]), "sign": sign})
    return pd.DataFrame(rows)


def feature_relevance(values: np.ndarray, y: pd.Series | pd.DataFrame, config: RunConfig) -> tuple[float, int]:
    try:
        if config.task == "binary":
            labels = pd.to_numeric(pd.Series(y), errors="coerce")
            if labels.nunique(dropna=True) != 2:
                return 0.0, 0
            auc = roc_auc_score(labels, values)
            return float(abs(auc - 0.5)), 1 if auc >= 0.5 else -1
        if config.task == "multiclass":
            labels = pd.Series(y).astype(str)
            stat = anova_f_statistic(values, labels.to_numpy())
            return finite_float(stat), 1
        if config.task == "regression":
            target = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
            corr = pd.Series(values).corr(pd.Series(target), method="spearman")
            return finite_float(abs(corr)), 1 if finite_float(corr) >= 0 else -1
        target = pd.DataFrame(y)
        event = event_indicator(target["event"], config)
        cindex = concordance_index(target["time"].to_numpy(dtype=float), values, event)
        return finite_float(abs(cindex - 0.5)), 1 if cindex >= 0.5 else -1
    except Exception:
        return 0.0, 0


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def anova_f_statistic(values: np.ndarray, labels: np.ndarray) -> float:
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    levels = np.unique(labels)
    if len(levels) < 2 or len(values) <= len(levels):
        return 0.0
    grand_mean = float(np.mean(values))
    between = 0.0
    within = 0.0
    for level in levels:
        group = values[labels == level]
        if len(group) == 0:
            continue
        group_mean = float(np.mean(group))
        between += len(group) * (group_mean - grand_mean) ** 2
        within += float(np.sum((group - group_mean) ** 2))
    df_between = len(levels) - 1
    df_within = len(values) - len(levels)
    if df_between <= 0 or df_within <= 0 or within <= 0:
        return 0.0
    return (between / df_between) / (within / df_within)


def event_indicator(event: pd.Series, config: RunConfig) -> np.ndarray:
    if config.task == "competing_risk":
        return event.astype(str).eq(str(config.competing_event_code)).to_numpy()
    numeric = pd.to_numeric(event, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int).to_numpy() > 0
    return event.astype(str).str.lower().isin({"1", "true", "yes", "event"}).to_numpy()


def concordance_index(time: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float:
    concordant = 0.0
    permissible = 0.0
    n = len(time)
    for i in range(n):
        if not event[i] or not math.isfinite(time[i]):
            continue
        for j in range(n):
            if time[i] >= time[j] or i == j:
                continue
            if not math.isfinite(time[j]):
                continue
            permissible += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5
    if permissible == 0:
        return 0.5
    return concordant / permissible


def correlation_filter(
    frame: pd.DataFrame,
    relevance: dict[str, float],
    modality: str,
    fold: str,
    config: RunConfig,
) -> tuple[list[str], list[DropRow], list[CorrelationRow]]:
    if frame.shape[1] <= 1:
        return frame.columns.tolist(), [], []
    imputed = pd.DataFrame(imputed_array(frame), columns=frame.columns)
    corr = imputed.corr(method=config.correlation_method).abs()
    ordered = sorted(frame.columns, key=lambda column: (-relevance.get(column, 0.0), column))
    kept: list[str] = []
    drops: list[DropRow] = []
    audit_rows: list[CorrelationRow] = []
    dropped: set[str] = set()
    for feature in ordered:
        if feature in dropped:
            continue
        kept.append(feature)
        for other in ordered:
            if other == feature or other in dropped or other in kept:
                continue
            value = finite_float(corr.loc[feature, other])
            if value >= config.correlation_threshold:
                dropped.add(other)
                drops.append(DropRow(modality, fold, other, "correlation", value, compared_with=feature))
                audit_rows.append(
                    CorrelationRow(
                        modality=modality,
                        fold=fold,
                        kept_feature=feature,
                        dropped_feature=other,
                        abs_correlation=value,
                        threshold=float(config.correlation_threshold),
                        method=config.correlation_method,
                        kept_relevance=float(relevance.get(feature, math.nan)),
                        dropped_relevance=float(relevance.get(other, math.nan)),
                        decision="drop_redundant_feature_keep_higher_relevance",
                    )
                )
    return kept, drops, audit_rows


def elastic_net_features(
    frame: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: RunConfig,
) -> tuple[list[str], dict[str, float], str]:
    if frame.empty:
        return [], {}, "no_features"
    x = frame.to_numpy(dtype=float)
    if config.task in {"binary", "multiclass"}:
        labels = LabelEncoder().fit_transform(pd.Series(y).astype(str))
        if len(np.unique(labels)) < 2:
            return [], {}, "one_class"
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            elastic_net_logistic(config),
        )
        model.fit(x, labels)
        classifier = model.named_steps["logisticregression"]
        coef = np.asarray(classifier.coef_)
        weights = np.max(np.abs(coef), axis=0)
        selected = [feature for feature, weight in zip(frame.columns, weights, strict=True) if abs(weight) > 1e-10]
        return selected, dict(zip(frame.columns, weights, strict=True)), "ok"
    if config.task == "regression":
        target = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(
                alpha=config.elastic_net_alpha,
                l1_ratio=config.elastic_net_l1_ratio,
                max_iter=10000,
                random_state=config.random_state,
            ),
        )
        model.fit(x, target)
        regressor = model.named_steps["elasticnet"]
        weights = np.asarray(regressor.coef_)
        selected = [feature for feature, weight in zip(frame.columns, weights, strict=True) if abs(weight) > 1e-10]
        return selected, dict(zip(frame.columns, weights, strict=True)), "ok"
    return survival_elastic_net_features(frame, y, config)


def survival_elastic_net_features(
    frame: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: RunConfig,
) -> tuple[list[str], dict[str, float], str]:
    try:
        from lifelines import CoxPHFitter
    except Exception:
        return [], {}, "lifelines_not_available"

    target = pd.DataFrame(y).reset_index(drop=True)
    model_frame, _, _ = fit_scaled_numeric_features(frame, list(frame.columns))
    model_frame["time"] = pd.to_numeric(target["time"], errors="coerce")
    model_frame["event"] = event_indicator(target["event"], config).astype(int)
    model_frame = model_frame.dropna(subset=["time", "event"])
    if model_frame["event"].sum() < 2:
        return [], {}, "too_few_events"
    fitter = CoxPHFitter(penalizer=config.elastic_net_alpha, l1_ratio=config.elastic_net_l1_ratio)
    try:
        fitter.fit(model_frame, duration_col="time", event_col="event", show_progress=False)
    except Exception as exc:
        return [], {}, f"cox_fit_failed:{exc}"
    coefficients = fitter.params_.drop(labels=["time", "event"], errors="ignore").to_dict()
    selected = [feature for feature, coef in coefficients.items() if abs(float(coef)) > 1e-10]
    return selected, {key: float(value) for key, value in coefficients.items()}, "ok"


def fit_scaled_numeric_features(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(numeric)
    scaled = scaler.fit_transform(imputed)
    return pd.DataFrame(scaled, columns=features, index=frame.index), imputer, scaler


def transform_scaled_numeric_features(
    frame: pd.DataFrame,
    features: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    imputed = imputer.transform(numeric)
    scaled = scaler.transform(imputed)
    return pd.DataFrame(scaled, columns=features, index=frame.index)


def evaluate_selected_model(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    features: list[str],
    modality: str,
    fold: str,
    config: RunConfig,
    id_column: str | None,
) -> dict[str, list[dict]]:
    if config.task in {"survival", "competing_risk"}:
        return evaluate_survival_model(train, test, y_train, y_test, features, modality, fold, config, id_column)
    estimator = build_predictive_estimator(config, y_train)
    if features:
        x_train = train[features].apply(pd.to_numeric, errors="coerce")
        x_test = test[features].apply(pd.to_numeric, errors="coerce")
    else:
        x_train = np.zeros((len(train), 1))
        x_test = np.zeros((len(test), 1))
    estimator.fit(x_train, y_train)
    predictions: list[dict] = []
    performance: list[dict] = []
    ids = test[id_column].tolist() if id_column and id_column in test.columns else list(test.index)

    if config.task in {"binary", "multiclass"}:
        y_true = pd.Series(y_test).astype(str)
        y_pred = pd.Series(estimator.predict(x_test)).astype(str)
        labels = sorted(pd.Series(y_train).astype(str).unique())
        proba = estimator.predict_proba(x_test) if hasattr(estimator, "predict_proba") else None
        for row_id, true, pred, row_proba in zip(ids, y_true, y_pred, proba if proba is not None else [None] * len(y_true)):
            record = {"id": row_id, "fold": fold, "modality": modality, "y_true": true, "prediction": pred}
            if row_proba is not None:
                for label, value in zip(estimator.classes_, row_proba, strict=False):
                    record[f"probability_{label}"] = float(value)
            predictions.append(record)
        row = {
            "fold": fold,
            "modality": modality,
            "task": config.task,
            "n_features": len(features),
            "n_test": len(test),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
        if config.task == "binary" and proba is not None and len(labels) == 2:
            positive = labels[-1]
            class_labels = [str(label) for label in estimator.classes_]
            positive_index = class_labels.index(positive)
            positive_proba = proba[:, positive_index]
            row["roc_auc"] = safe_metric(roc_auc_score, y_true.eq(positive).astype(int), positive_proba)
            row["average_precision"] = safe_metric(
                average_precision_score, y_true.eq(positive).astype(int), positive_proba
            )
        elif config.task == "multiclass" and proba is not None:
            row["roc_auc_ovr"] = safe_metric(roc_auc_score, y_true, proba, multi_class="ovr")
        performance.append(row)
        return {"performance": performance, "predictions": predictions}

    y_true_num = pd.to_numeric(pd.Series(y_test), errors="coerce")
    pred = pd.Series(estimator.predict(x_test))
    for row_id, true, value in zip(ids, y_true_num, pred, strict=False):
        predictions.append({"id": row_id, "fold": fold, "modality": modality, "y_true": true, "prediction": value})
    performance.append(
        {
            "fold": fold,
            "modality": modality,
            "task": config.task,
            "n_features": len(features),
            "n_test": len(test),
            "r2": safe_metric(r2_score, y_true_num, pred),
            "mae": safe_metric(mean_absolute_error, y_true_num, pred),
            "rmse": math.sqrt(safe_metric(mean_squared_error, y_true_num, pred)),
        }
    )
    return {"performance": performance, "predictions": predictions}


def evaluate_projection_model(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    columns: list[str],
    modality: str,
    fold: str,
    config: RunConfig,
    id_column: str | None,
) -> dict[str, list[dict]]:
    if config.projection == "none":
        return {"performance": [], "predictions": []}

    x_train, x_test, metadata = fit_projection_transform(train, test, y_train, columns, config)
    if x_train.empty or x_test.empty:
        return {
            "performance": [
                {
                    "fold": fold,
                    "modality": modality,
                    "task": config.task,
                    "projection": config.projection,
                    "n_input_features": int(metadata.get("n_input_features", 0)),
                    "n_components": int(metadata.get("n_components", 0)),
                    "n_test": len(test),
                    "status": metadata.get("status", "projection_unavailable"),
                }
            ],
            "predictions": [],
        }

    train_components = x_train.reset_index(drop=True)
    test_components = x_test.reset_index(drop=True)
    if id_column and id_column in train.columns:
        train_components[id_column] = train[id_column].reset_index(drop=True).to_numpy()
    if id_column and id_column in test.columns:
        test_components[id_column] = test[id_column].reset_index(drop=True).to_numpy()
    component_columns = list(x_train.columns)
    validation = evaluate_selected_model(
        train=train_components,
        test=test_components,
        y_train=y_train,
        y_test=y_test,
        features=component_columns,
        modality=modality,
        fold=fold,
        config=config,
        id_column=id_column,
    )
    for row in validation["performance"]:
        row["projection"] = config.projection
        row["n_input_features"] = int(metadata.get("n_input_features", 0))
        row["n_components"] = int(metadata.get("n_components", len(component_columns)))
        row["status"] = metadata.get("status", "ok")
    for row in validation["predictions"]:
        row["projection"] = config.projection
    return validation


def build_predictive_estimator(config: RunConfig, y_train: pd.Series | pd.DataFrame) -> BaseEstimator:
    if config.task in {"binary", "multiclass"}:
        labels = pd.Series(y_train).astype(str)
        if labels.nunique() < 2:
            return DummyClassifier(strategy="most_frequent")
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            elastic_net_logistic(config),
        )
    target = pd.to_numeric(pd.Series(y_train), errors="coerce")
    if target.nunique(dropna=True) < 2:
        return DummyRegressor(strategy="mean")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNet(
            alpha=config.elastic_net_alpha,
            l1_ratio=config.elastic_net_l1_ratio,
            max_iter=10000,
            random_state=config.random_state,
        ),
    )


def elastic_net_logistic(config: RunConfig) -> LogisticRegression:
    kwargs = {
        "solver": "saga",
        "C": config.elastic_net_c,
        "l1_ratio": config.elastic_net_l1_ratio,
        "class_weight": "balanced",
        "max_iter": 10000,
        "random_state": config.random_state,
    }
    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty is not None and penalty.default != "deprecated":
        kwargs["penalty"] = "elasticnet"
    return LogisticRegression(**kwargs)


def safe_metric(function, *args, **kwargs) -> float:
    try:
        result = function(*args, **kwargs)
    except Exception:
        return math.nan
    return finite_float(result, math.nan)


def evaluate_survival_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    features: list[str],
    modality: str,
    fold: str,
    config: RunConfig,
    id_column: str | None,
) -> dict[str, list[dict]]:
    if not features:
        return {
            "performance": [
                {
                    "fold": fold,
                    "modality": modality,
                    "task": config.task,
                    "n_features": 0,
                    "n_test": len(test),
                    "c_index": math.nan,
                    "status": "no_features",
                }
            ],
            "predictions": [],
        }
    try:
        from lifelines import CoxPHFitter
    except Exception:
        return evaluate_survival_signed_score(
            train, test, y_train, y_test, features, modality, fold, config, id_column, "lifelines_not_available"
        )

    train_y = pd.DataFrame(y_train).reset_index(drop=True)
    test_y = pd.DataFrame(y_test).reset_index(drop=True)
    train_frame, imputer, scaler = fit_scaled_numeric_features(train, features)
    train_frame = train_frame.reset_index(drop=True)
    train_frame["time"] = pd.to_numeric(train_y["time"], errors="coerce")
    train_frame["event"] = event_indicator(train_y["event"], config).astype(int)
    fitter = CoxPHFitter(penalizer=config.elastic_net_alpha, l1_ratio=config.elastic_net_l1_ratio)
    try:
        fitter.fit(train_frame.dropna(subset=["time", "event"]), duration_col="time", event_col="event")
    except Exception as exc:
        return evaluate_survival_signed_score(
            train, test, y_train, y_test, features, modality, fold, config, id_column, f"cox_fit_failed:{exc}"
        )
    x_test = transform_scaled_numeric_features(test, features, imputer, scaler).reset_index(drop=True)
    risk = fitter.predict_partial_hazard(x_test).to_numpy(dtype=float)
    event = event_indicator(test_y["event"], config)
    c_index = concordance_index(pd.to_numeric(test_y["time"], errors="coerce").to_numpy(dtype=float), risk, event)
    ids = test[id_column].tolist() if id_column and id_column in test.columns else list(test.index)
    predictions = [
        {"id": row_id, "fold": fold, "modality": modality, "risk": float(value)}
        for row_id, value in zip(ids, risk, strict=False)
    ]
    return {
        "performance": [
            {
                "fold": fold,
                "modality": modality,
                "task": config.task,
                "n_features": len(features),
                "n_test": len(test),
                "c_index": c_index,
                "status": "ok",
            }
        ],
        "predictions": predictions,
    }


def evaluate_survival_signed_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
    features: list[str],
    modality: str,
    fold: str,
    config: RunConfig,
    id_column: str | None,
    reason: str,
) -> dict[str, list[dict]]:
    train_y = pd.DataFrame(y_train).reset_index(drop=True)
    test_y = pd.DataFrame(y_test).reset_index(drop=True)
    train_numeric = train[features].apply(pd.to_numeric, errors="coerce")
    test_numeric = test[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(imputer.fit_transform(train_numeric), columns=features)
    x_test = pd.DataFrame(imputer.transform(test_numeric), columns=features)
    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0).replace(0, 1).fillna(1)
    z_test = (x_test - means) / stds
    signs = []
    for feature in features:
        _, sign = feature_relevance(x_train[feature].to_numpy(dtype=float), train_y, config)
        signs.append(sign if sign != 0 else 1)
    risk = z_test.to_numpy(dtype=float).dot(np.asarray(signs, dtype=float)) / math.sqrt(max(len(features), 1))
    event = event_indicator(test_y["event"], config)
    c_index = concordance_index(pd.to_numeric(test_y["time"], errors="coerce").to_numpy(dtype=float), risk, event)
    ids = test[id_column].tolist() if id_column and id_column in test.columns else list(test.index)
    predictions = [
        {"id": row_id, "fold": fold, "modality": modality, "risk": float(value)}
        for row_id, value in zip(ids, risk, strict=False)
    ]
    return {
        "performance": [
            {
                "fold": fold,
                "modality": modality,
                "task": config.task,
                "n_features": len(features),
                "n_test": len(test),
                "c_index": c_index,
                "status": f"signed_score_fallback:{reason}",
            }
        ],
        "predictions": predictions,
    }


def run_stability_selection(
    frame: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    modalities: dict[str, list[str]],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_columns = [
        "modality",
        "feature",
        "selected_count",
        "eligible_resamples",
        "selection_probability",
        "stable",
    ]
    resample_columns = [
        "modality",
        "resample",
        "fold",
        "sampling_unit",
        "n_train_rows",
        "n_train_groups",
        "train_groups",
        "train_row_indices",
        "train_ids",
        "n_candidate_features",
        "n_selected_features",
        "selected_features",
        "model_status",
    ]
    if config.stability_resamples <= 0:
        return pd.DataFrame(columns=stability_columns), pd.DataFrame(columns=resample_columns)
    stability_config = replace(config, tune_elastic_net=False)
    rows = []
    resample_rows = []
    for modality, columns in modalities.items():
        counts: dict[str, int] = {}
        eligible: dict[str, int] = {}
        for resample, train_idx in enumerate(iter_stability_indices(frame, y, config), start=1):
            fold_name = f"stability_{resample}"
            fitted = fit_selector(
                frame.iloc[train_idx],
                y_train=target_subset(y, train_idx),
                columns=columns,
                modality=modality,
                fold=fold_name,
                config=stability_config,
            )
            audit_idx = np.sort(train_idx)
            train_groups = stability_train_groups(frame, config, audit_idx)
            train_ids = (
                frame.iloc[audit_idx][config.id_column].astype(str).tolist()
                if config.id_column and config.id_column in frame.columns
                else [str(index) for index in audit_idx]
            )
            resample_rows.append(
                {
                    "modality": modality,
                    "resample": resample,
                    "fold": fold_name,
                    "sampling_unit": "group" if train_groups else "row",
                    "n_train_rows": int(len(train_idx)),
                    "n_train_groups": len(train_groups),
                    "train_groups": ";".join(train_groups),
                    "train_row_indices": ";".join(str(int(index)) for index in audit_idx),
                    "train_ids": ";".join(train_ids),
                    "n_candidate_features": int(len(columns)),
                    "n_selected_features": int(len(fitted.selected_features)),
                    "selected_features": ";".join(fitted.selected_features),
                    "model_status": fitted.model_status,
                }
            )
            for feature in columns:
                eligible[feature] = eligible.get(feature, 0) + 1
            for feature in fitted.selected_features:
                counts[feature] = counts.get(feature, 0) + 1
        for feature in sorted(eligible):
            probability = counts.get(feature, 0) / max(eligible[feature], 1)
            rows.append(
                {
                    "modality": modality,
                    "feature": feature,
                    "selected_count": counts.get(feature, 0),
                    "eligible_resamples": eligible[feature],
                    "selection_probability": probability,
                    "stable": probability >= config.stability_threshold,
                }
            )
    stability = pd.DataFrame(rows, columns=stability_columns)
    if not stability.empty:
        stability = stability.sort_values(
            ["modality", "selection_probability", "feature"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    resamples = pd.DataFrame(resample_rows, columns=resample_columns)
    return stability, resamples


def iter_stability_indices(
    frame: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: RunConfig,
) -> Iterable[np.ndarray]:
    n_rows = len(frame)
    indices = np.arange(n_rows)
    train_size = config.stability_train_fraction
    if config.group_column and config.group_column in frame.columns:
        groups = frame[config.group_column].astype(str).to_numpy()
        unique_groups = np.array(sorted(pd.unique(groups)))
        n_train_groups = max(1, int(round(train_size * len(unique_groups))))
        n_train_groups = min(n_train_groups, len(unique_groups))
        rng = np.random.default_rng(config.random_state)
        for _ in range(config.stability_resamples):
            selected_groups = rng.choice(unique_groups, size=n_train_groups, replace=False)
            yield np.flatnonzero(np.isin(groups, selected_groups))
        return
    if config.task in {"binary", "multiclass"}:
        labels = pd.Series(y).astype(str)
        min_class = labels.value_counts().min()
        if min_class >= 2:
            splitter = StratifiedShuffleSplit(
                n_splits=config.stability_resamples,
                train_size=train_size,
                random_state=config.random_state,
            )
            for train_idx, _ in splitter.split(indices, labels):
                yield train_idx
            return
    rng = np.random.default_rng(config.random_state)
    n_train = max(2, int(round(train_size * n_rows)))
    for _ in range(config.stability_resamples):
        yield np.sort(rng.choice(indices, size=n_train, replace=False))


def stability_train_groups(frame: pd.DataFrame, config: RunConfig, train_idx: np.ndarray) -> list[str]:
    if not (config.group_column and config.group_column in frame.columns):
        return []
    return sorted(frame.iloc[train_idx][config.group_column].astype(str).unique().tolist())
