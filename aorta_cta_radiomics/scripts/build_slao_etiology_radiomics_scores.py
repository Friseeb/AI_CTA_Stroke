#!/usr/bin/env python
"""Build imaging-only stroke etiology composites from SLAO aorta radiomics features.

The first etiology dashboard uses a binary Cardioembolic-vs-ESUS contrast.
Sparse mechanisms are counted and excluded from this model rather than forced
into an unstable multiclass classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import build_slao_mace_radiomics_scores as score_builder


AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = AORTA_ROOT / "outputs" / "aorta_batch_run" / "mace_slao" / "slao_mace_aorta_modeling.csv"
DEFAULT_OUTDIR = AORTA_ROOT / "outputs" / "aorta_batch_run" / "etiology_slao"
DEFAULT_AF_SUBTYPE_SOURCE = Path("~/Desktop/SLAO analysis/slaao_r_dataset_derived.csv").expanduser()

RECORD_ID_COLUMN = "record_id"
STUDY_ID_COLUMN = "Study_ID"
MECHANISM_COLUMN = "stroke_mechanism"
OUTCOME_COLUMN = "etiology_cardioembolic_vs_esus"
POSITIVE_LABEL = "Cardioembolic"
NEGATIVE_LABEL = "ESUS"
KNOWN_ETIOLOGY_COLUMN = "known_source_etiology"
ATRIAL_FIBRILLATION_COLUMN = "atrial_fibrillation"
ETIOLOGY_REVIEW_GROUP_COLUMN = "etiology_review_group"
SOURCE_ETIOLOGY_LABEL_COLUMN = "source_etiology_label"
AF_SUBTYPE_CODE_COLUMN = "kaf_or_afdas_event"
AF_SUBTYPE_LABEL_COLUMN = "af_subtype_label"
AF_SUBTYPE_CODE_LABELS = {
    "1": "KAF",
    "2": "AFDAS",
    "3": "New_ECG_AF",
}
AF_SUBTYPE_LABEL_CODES = {label: code for code, label in AF_SUBTYPE_CODE_LABELS.items()}
DIRECT_ETIOLOGY_LABELS = {
    "KAF": "KAF",
    "AFDAS": "AFDAS",
    "ECG_AF": "New_ECG_AF",
    "ESUS": "ESUS",
}
DIRECT_ETIOLOGY_SOURCE_COLUMNS = [
    "KAF",
    "AFDAS",
    "ECG_AF",
    "ESUS",
    "KAF_Off_OACs",
    "af_phenotype",
    "esus_group",
    "af_burden",
    "AF_Burden2",
    "AFDAS_Burden",
    "Unified_AFDAS_burden",
]
AF_SUBTYPE_SOURCE_COLUMNS = [
    "af",
    "aftype",
    AF_SUBTYPE_CODE_COLUMN,
    "dx_af___1",
    "dx_af___2",
    "dx_af___3",
    "dx_af___4",
    "dx_af___5",
    "dx_af___6",
    "dx_af___7",
    "dx_af___8",
    "dx_af___9",
    "dx_af___10",
    "dx_af___0",
    "dx_af___unk",
    "dx_af___navu",
    "dx_af___ni",
    "af_burden",
    "afdas_burden",
]
AF_SUBTYPE_RENAMES = {
    "af": "redcap_af",
    "aftype": "redcap_aftype",
}
AFDAS_LABELS = {
    "AFDAS",
    "AF DETECTED AFTER STROKE",
    "AF DETECTED AFTER STROKE AFDAS",
    "AF-DETECTED-AFTER-STROKE",
    "NEW AF",
    "NEW ATRIAL FIBRILLATION",
    "POST STROKE AF",
    "POST-STROKE AF",
}
KAF_LABELS = {
    "KAF",
    "KNOWN AF",
    "KNOWN ATRIAL FIBRILLATION",
    "PRE EXISTING AF",
    "PRE-EXISTING AF",
    "PREEXISTING AF",
    "PRIOR AF",
}
POSITIVE_MECHANISM_LABELS = {POSITIVE_LABEL.upper(), *AFDAS_LABELS, *KAF_LABELS}
NEGATIVE_MECHANISM_LABELS = {NEGATIVE_LABEL.upper()}
SUMMARY_SCORE_COLUMNS = [
    "domain_sum__probability_mean_cv__platt_cv",
    "domain_sum__signed_z_cv",
    "all_imaging__probability_cv__platt_cv",
]
PREDICTION_SCORE_COLUMNS = {
    "domain_sum_probability_platt": "domain_sum__probability_mean_cv__platt_cv",
    "domain_sum_signed_z": "domain_sum__signed_z_cv",
    "all_imaging_probability_platt": "all_imaging__probability_cv__platt_cv",
}


@dataclass(frozen=True)
class EtiologyTarget:
    slug: str
    outcome_column: str
    positive_label: str
    source_columns: tuple[str, ...]
    report_title: str
    extra_note: str


ETIOLOGY_TARGETS = [
    EtiologyTarget(
        slug="kaf",
        outcome_column="etiology_kaf",
        positive_label="KAF",
        source_columns=("KAF",),
        report_title="KAF radiomics prediction",
        extra_note="One-vs-rest etiology task: KAF is coded 1; AFDAS, ECG-AF, and ESUS are coded 0.",
    ),
    EtiologyTarget(
        slug="afdas",
        outcome_column="etiology_afdas",
        positive_label="AFDAS",
        source_columns=("AFDAS",),
        report_title="AFDAS radiomics prediction",
        extra_note="One-vs-rest etiology task: AFDAS is coded 1; KAF, ECG-AF, and ESUS are coded 0.",
    ),
    EtiologyTarget(
        slug="ecg_af",
        outcome_column="etiology_ecg_af",
        positive_label="ECG-AF",
        source_columns=("ECG_AF",),
        report_title="ECG-AF radiomics prediction",
        extra_note="One-vs-rest etiology task: ECG-AF is coded 1; KAF, AFDAS, and ESUS are coded 0.",
    ),
    EtiologyTarget(
        slug="esus",
        outcome_column="etiology_esus",
        positive_label="ESUS",
        source_columns=("ESUS",),
        report_title="ESUS radiomics prediction",
        extra_note="One-vs-rest etiology task: ESUS is coded 1; KAF, AFDAS, and ECG-AF are coded 0.",
    ),
    EtiologyTarget(
        slug="afdas_or_ecg_af",
        outcome_column="etiology_afdas_or_ecg_af",
        positive_label="AFDAS or ECG-AF",
        source_columns=("AFDAS", "ECG_AF"),
        report_title="AFDAS or ECG-AF radiomics prediction",
        extra_note=(
            "Sensitivity one-vs-rest etiology task: either direct AFDAS or ECG-AF is coded 1; "
            "KAF and ESUS are coded 0."
        ),
    ),
]


def main() -> None:
    args, passthrough = build_parser().parse_known_args()
    input_path = args.input.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    af_subtype_source = args.af_subtype_source.expanduser().resolve()
    scores_dir = outdir / "radiomics_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = outdir / "slao_etiology_aorta_modeling.csv"
    counts_path = outdir / "slao_etiology_class_counts.csv"
    source_group_counts_path = outdir / "slao_etiology_source_group_counts.csv"
    target_counts_path = outdir / "slao_etiology_target_counts.csv"
    prepare_etiology_analysis(
        input_path,
        analysis_path,
        counts_path,
        source_group_counts_path,
        target_counts_path,
        af_subtype_source=af_subtype_source,
    )

    target_outputs = []
    for target in ETIOLOGY_TARGETS:
        target_outputs.append(run_target_prediction(target, analysis_path, scores_dir, passthrough))

    predictions_path = scores_dir / "slao_etiology_one_vs_rest_predictions.csv"
    performance_summary_path = scores_dir / "slao_etiology_one_vs_rest_performance.csv"
    prediction_group_summary_path = scores_dir / "slao_etiology_prediction_by_true_group.csv"
    write_prediction_matrix(analysis_path, target_outputs, predictions_path)
    write_target_performance_summary(target_outputs, performance_summary_path)
    write_prediction_group_summary(predictions_path, prediction_group_summary_path)

    print(f"Etiology class counts: {counts_path}")
    print(f"Etiology source group counts: {source_group_counts_path}")
    print(f"Etiology target counts: {target_counts_path}")
    print(f"Etiology one-vs-rest predictions: {predictions_path}")
    print(f"Etiology one-vs-rest performance: {performance_summary_path}")


def prepare_etiology_analysis(
    input_path: Path,
    output_path: Path,
    counts_path: Path,
    source_group_counts_path: Path,
    target_counts_path: Path,
    *,
    af_subtype_source: Path,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input SLAO modeling CSV not found: {input_path}")
    frame = pd.read_csv(input_path, dtype=str)
    if MECHANISM_COLUMN not in frame.columns:
        raise ValueError(f"Expected etiology column '{MECHANISM_COLUMN}' in {input_path}.")
    frame = enrich_af_subtypes(frame, af_subtype_source)

    mechanism = frame[MECHANISM_COLUMN].fillna("").replace("", "(blank)")
    counts = mechanism.value_counts(dropna=False).rename_axis("stroke_mechanism").reset_index(name="n")
    counts["modeled_in_ce_vs_esus"] = counts["stroke_mechanism"].map(classify_ce_vs_esus).notna()
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(counts_path, index=False)

    outcome = mechanism.map(classify_ce_vs_esus)
    frame[OUTCOME_COLUMN] = outcome
    frame[ETIOLOGY_REVIEW_GROUP_COLUMN] = frame.apply(review_group, axis=1)
    frame[KNOWN_ETIOLOGY_COLUMN] = known_direct_etiology_mask(frame)
    for target in ETIOLOGY_TARGETS:
        frame[target.outcome_column] = target_positive_mask(frame, target).astype(int)
    write_source_group_counts(frame, source_group_counts_path)

    modeled = frame[frame[KNOWN_ETIOLOGY_COLUMN]].copy()
    if modeled.empty:
        raise ValueError("No rows with direct KAF/AFDAS/ECG-AF/ESUS labels were available for etiology modeling.")
    modeled[OUTCOME_COLUMN] = outcome.loc[modeled.index]
    modeled["study_arm"] = modeled[SOURCE_ETIOLOGY_LABEL_COLUMN].fillna("").replace("", "Unknown etiology")
    modeled.to_csv(output_path, index=False)
    write_target_counts(modeled, target_counts_path)

    summary_path = output_path.with_name("slao_etiology_modeling_summary.json")
    summary = {
        "input_csv": str(input_path),
        "analysis_csv": str(output_path),
        "class_counts_csv": str(counts_path),
        "source_group_counts_csv": str(source_group_counts_path),
        "target_counts_csv": str(target_counts_path),
        "af_subtype_source_csv": str(af_subtype_source),
        "mechanism_column": MECHANISM_COLUMN,
        "af_subtype_code_column": AF_SUBTYPE_CODE_COLUMN,
        "af_subtype_label_column": AF_SUBTYPE_LABEL_COLUMN,
        "source_etiology_label_column": SOURCE_ETIOLOGY_LABEL_COLUMN,
        "outcome_column": OUTCOME_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "raw_rows": int(len(frame)),
        "modeled_rows": int(len(modeled)),
        "known_source_etiology_rows": int(frame[KNOWN_ETIOLOGY_COLUMN].sum()),
        "excluded_rows": int(len(frame) - len(modeled)),
        "excluded_mechanisms": counts.loc[~counts["modeled_in_ce_vs_esus"], "stroke_mechanism"].tolist(),
        "source_etiology_label_counts": value_counts_dict(modeled.get(SOURCE_ETIOLOGY_LABEL_COLUMN)),
        "af_subtype_counts": value_counts_dict(modeled.get(AF_SUBTYPE_LABEL_COLUMN)),
        "review_group_counts": value_counts_dict(modeled.get(ETIOLOGY_REVIEW_GROUP_COLUMN)),
        "targets": [
            {
                "slug": target.slug,
                "outcome_column": target.outcome_column,
                "positive_label": target.positive_label,
                "source_columns": list(target.source_columns),
                "positive_rows": int(modeled[target.outcome_column].sum()),
                "negative_rows": int(len(modeled) - modeled[target.outcome_column].sum()),
            }
            for target in ETIOLOGY_TARGETS
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_target_prediction(
    target: EtiologyTarget,
    analysis_path: Path,
    scores_dir: Path,
    passthrough: list[str],
) -> dict[str, object]:
    target_dir = scores_dir / target.slug
    assets_dir = target_dir / "report_assets"
    target_dir.mkdir(parents=True, exist_ok=True)

    score_builder.OUTCOME = target.outcome_column
    score_builder.REPORT_EYEBROW = "SLAO stroke etiology one-vs-rest radiomics"
    score_builder.REPORT_TITLE = target.report_title
    score_builder.REPORT_PAGE_TITLE = f"SLAO Radiomics {target.positive_label} Prediction"
    score_builder.REPORT_OUTCOME_NAME = f"{target.positive_label} etiology"
    score_builder.REPORT_POSITIVE_LABEL = target.positive_label
    score_builder.REPORT_NEGATIVE_LABEL = "Other direct etiology labels"
    score_builder.REPORT_EXTRA_NOTE = (
        f"{target.extra_note} Direct labels are read from {DEFAULT_AF_SUBTYPE_SOURCE}; "
        "rows without a direct KAF/AFDAS/ECG-AF/ESUS label are excluded from these one-vs-rest tasks."
    )
    score_builder.CASE_GROUP_LABEL = "Source etiology"

    output_paths = {
        "target": target,
        "scores": target_dir / f"slao_etiology_{target.slug}_radiomics_scores.csv",
        "summary": target_dir / f"radiomics_etiology_{target.slug}_score_summary.json",
        "performance": target_dir / f"radiomics_etiology_{target.slug}_score_performance.csv",
        "selected_features": target_dir / f"radiomics_etiology_{target.slug}_score_selected_features.csv",
        "collinearity": target_dir / f"radiomics_etiology_{target.slug}_score_collinearity_dropped.csv",
        "stability": target_dir / f"radiomics_etiology_{target.slug}_score_stability_selection.csv",
        "report": target_dir / f"radiomics_etiology_{target.slug}_score_report.html",
        "assets_dir": assets_dir,
    }
    builder_args = [
        "--analysis",
        str(analysis_path),
        "--scores",
        str(output_paths["scores"]),
        "--summary",
        str(output_paths["summary"]),
        "--performance",
        str(output_paths["performance"]),
        "--selected-features",
        str(output_paths["selected_features"]),
        "--collinearity-report",
        str(output_paths["collinearity"]),
        "--stability-report",
        str(output_paths["stability"]),
        "--report",
        str(output_paths["report"]),
        "--assets-dir",
        str(assets_dir),
        *passthrough,
    ]
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *builder_args]
        score_builder.main()
    finally:
        sys.argv = original_argv
    return output_paths


def write_prediction_matrix(analysis_path: Path, target_outputs: list[dict[str, object]], output_path: Path) -> None:
    analysis = pd.read_csv(analysis_path, dtype=str)
    clinical_columns = [
        column
        for column in [
            score_builder.CASE_ID,
            RECORD_ID_COLUMN,
            "source_cohort",
            "study_arm",
            MECHANISM_COLUMN,
            ATRIAL_FIBRILLATION_COLUMN,
            SOURCE_ETIOLOGY_LABEL_COLUMN,
            AF_SUBTYPE_LABEL_COLUMN,
            "KAF",
            "AFDAS",
            "ECG_AF",
            "ESUS",
            "KAF_Off_OACs",
            "af_phenotype",
            "esus_group",
        ]
        if column in analysis.columns
    ]
    matrix = analysis[clinical_columns].drop_duplicates(score_builder.CASE_ID).copy()
    for output in target_outputs:
        target = output["target"]
        assert isinstance(target, EtiologyTarget)
        scores = pd.read_csv(output["scores"], dtype=str)
        best_probability_column = best_calibrated_probability_score(Path(output["performance"]))
        keep_columns = [score_builder.CASE_ID, target.outcome_column]
        keep_columns.extend(column for column in PREDICTION_SCORE_COLUMNS.values() if column in scores.columns)
        if best_probability_column and best_probability_column in scores.columns and best_probability_column not in keep_columns:
            keep_columns.append(best_probability_column)
        target_scores = scores[keep_columns].copy()
        rename = {target.outcome_column: f"{target.slug}__observed"}
        for short_name, column in PREDICTION_SCORE_COLUMNS.items():
            if column in target_scores.columns:
                rename[column] = f"{target.slug}__{short_name}"
        if best_probability_column and best_probability_column in target_scores.columns:
            rename[best_probability_column] = f"{target.slug}__best_calibrated_probability"
        target_scores = target_scores.rename(columns=rename)
        if best_probability_column:
            target_scores[f"{target.slug}__best_calibrated_probability_score"] = best_probability_column
        matrix = matrix.merge(target_scores, on=score_builder.CASE_ID, how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False)


def best_calibrated_probability_score(performance_path: Path) -> str:
    performance = pd.read_csv(performance_path, dtype=str)
    if performance.empty or "score_name" not in performance.columns:
        return ""
    performance = performance.copy()
    performance["auc_numeric"] = pd.to_numeric(performance.get("auc"), errors="coerce")
    score_names = performance["score_name"].fillna("").astype(str)
    probability = score_names.str.contains("probability")
    calibrated = score_names.str.endswith("__platt_cv")
    candidates = performance[probability & calibrated & performance["auc_numeric"].notna()]
    if candidates.empty:
        candidates = performance[probability & performance["auc_numeric"].notna()]
    if candidates.empty:
        return ""
    return str(candidates.sort_values("auc_numeric", ascending=False).iloc[0]["score_name"])


def write_target_performance_summary(target_outputs: list[dict[str, object]], output_path: Path) -> None:
    frames = []
    for output in target_outputs:
        target = output["target"]
        assert isinstance(target, EtiologyTarget)
        performance = pd.read_csv(output["performance"], dtype=str)
        performance.insert(0, "target", target.slug)
        performance.insert(1, "positive_label", target.positive_label)
        performance.insert(2, "outcome_column", target.outcome_column)
        frames.append(performance)
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def write_prediction_group_summary(predictions_path: Path, output_path: Path) -> None:
    predictions = pd.read_csv(predictions_path, dtype=str)
    probability_columns = [
        f"{target.slug}__domain_sum_probability_platt"
        for target in ETIOLOGY_TARGETS
        if f"{target.slug}__domain_sum_probability_platt" in predictions.columns
    ]
    rows = []
    grouped = predictions.groupby(SOURCE_ETIOLOGY_LABEL_COLUMN, dropna=False)
    for group_name, group in grouped:
        row: dict[str, object] = {
            SOURCE_ETIOLOGY_LABEL_COLUMN: clean_text(group_name) or "(blank)",
            "n": int(len(group)),
        }
        for column in probability_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}__mean"] = float(values.mean()) if not values.empty else ""
            row[f"{column}__median"] = float(values.median()) if not values.empty else ""
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("n", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def enrich_af_subtypes(frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    if RECORD_ID_COLUMN not in frame.columns:
        raise ValueError(f"Expected join column '{RECORD_ID_COLUMN}' in SLAO modeling CSV.")
    if not source_path.exists():
        raise FileNotFoundError(
            f"AF subtype source CSV not found: {source_path}. "
            "This file is required for explicit KAF/AFDAS/ESUS grouping."
        )

    source = pd.read_csv(source_path, dtype=str)
    join_column = source_join_column(source, source_path)
    if join_column != RECORD_ID_COLUMN:
        source = source.rename(columns={join_column: RECORD_ID_COLUMN})

    if "redcap_repeat_instrument" in source.columns:
        repeat = source["redcap_repeat_instrument"].fillna("").astype(str).str.strip()
        source = source[repeat.eq("")].copy()

    if any(column in source.columns for column in DIRECT_ETIOLOGY_LABELS):
        source = prepare_direct_etiology_source(source)
    elif AF_SUBTYPE_CODE_COLUMN in source.columns:
        source = prepare_legacy_af_subtype_source(source)
    else:
        raise ValueError(
            f"Expected direct etiology columns {sorted(DIRECT_ETIOLOGY_LABELS)} or "
            f"'{AF_SUBTYPE_CODE_COLUMN}' in etiology source: {source_path}"
        )

    merged = frame.copy()
    merged[RECORD_ID_COLUMN] = merged[RECORD_ID_COLUMN].map(clean_text)
    return merged.merge(source, on=RECORD_ID_COLUMN, how="left")


def source_join_column(source: pd.DataFrame, source_path: Path) -> str:
    for column in [RECORD_ID_COLUMN, STUDY_ID_COLUMN]:
        if column in source.columns:
            return column
    raise ValueError(
        f"Expected join column '{RECORD_ID_COLUMN}' or '{STUDY_ID_COLUMN}' in etiology source: {source_path}"
    )


def prepare_direct_etiology_source(source: pd.DataFrame) -> pd.DataFrame:
    available_columns = [column for column in DIRECT_ETIOLOGY_SOURCE_COLUMNS if column in source.columns]
    source = source[[RECORD_ID_COLUMN, *available_columns]].copy()
    source[RECORD_ID_COLUMN] = source[RECORD_ID_COLUMN].map(clean_text)
    source = source[source[RECORD_ID_COLUMN] != ""].drop_duplicates(RECORD_ID_COLUMN)
    source[SOURCE_ETIOLOGY_LABEL_COLUMN] = source.apply(direct_etiology_label, axis=1)
    source[AF_SUBTYPE_LABEL_COLUMN] = source[SOURCE_ETIOLOGY_LABEL_COLUMN].map(
        lambda label: label if label in AF_SUBTYPE_LABEL_CODES else ""
    )
    source[AF_SUBTYPE_CODE_COLUMN] = source[AF_SUBTYPE_LABEL_COLUMN].map(AF_SUBTYPE_LABEL_CODES).fillna("")
    return source


def prepare_legacy_af_subtype_source(source: pd.DataFrame) -> pd.DataFrame:
    available_columns = [column for column in AF_SUBTYPE_SOURCE_COLUMNS if column in source.columns]
    source = source[[RECORD_ID_COLUMN, *available_columns]].copy()
    source[RECORD_ID_COLUMN] = source[RECORD_ID_COLUMN].map(clean_text)
    source = source[source[RECORD_ID_COLUMN] != ""].drop_duplicates(RECORD_ID_COLUMN)
    source[AF_SUBTYPE_LABEL_COLUMN] = source[AF_SUBTYPE_CODE_COLUMN].map(af_subtype_label)
    source[SOURCE_ETIOLOGY_LABEL_COLUMN] = source[AF_SUBTYPE_LABEL_COLUMN]
    source = source.rename(columns=AF_SUBTYPE_RENAMES)
    return source


def write_source_group_counts(frame: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for group_name, group in frame.groupby(ETIOLOGY_REVIEW_GROUP_COLUMN, dropna=False):
        outcome = group.get(OUTCOME_COLUMN)
        if outcome is None:
            outcome_values = pd.Series(dtype=float)
        else:
            outcome_values = pd.to_numeric(outcome, errors="coerce")
        rows.append(
            {
                ETIOLOGY_REVIEW_GROUP_COLUMN: group_name,
                "n": int(len(group)),
                "modeled_in_ce_vs_esus": int(outcome_values.notna().sum()) if not outcome_values.empty else 0,
                "stroke_mechanism_values": ";".join(sorted(clean_values(group.get(MECHANISM_COLUMN)))),
                "source_etiology_label_values": ";".join(
                    sorted(clean_values(group.get(SOURCE_ETIOLOGY_LABEL_COLUMN)))
                ),
                "af_subtype_label_values": ";".join(sorted(clean_values(group.get(AF_SUBTYPE_LABEL_COLUMN)))),
                "kaf_or_afdas_event_values": ";".join(sorted(clean_values(group.get(AF_SUBTYPE_CODE_COLUMN)))),
                "KAF_values": ";".join(sorted(clean_values(group.get("KAF")))),
                "AFDAS_values": ";".join(sorted(clean_values(group.get("AFDAS")))),
                "ECG_AF_values": ";".join(sorted(clean_values(group.get("ECG_AF")))),
                "ESUS_values": ";".join(sorted(clean_values(group.get("ESUS")))),
                "redcap_af_values": ";".join(sorted(clean_values(group.get("redcap_af")))),
                "interpretation_note": group_interpretation_note(str(group_name)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["n", ETIOLOGY_REVIEW_GROUP_COLUMN], ascending=[False, True])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def known_direct_etiology_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in DIRECT_ETIOLOGY_LABELS:
        if column in frame.columns:
            mask = mask | frame[column].map(positive_flag)
    return mask


def target_positive_mask(frame: pd.DataFrame, target: EtiologyTarget) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in target.source_columns:
        if column in frame.columns:
            mask = mask | frame[column].map(positive_flag)
    return mask


def write_target_counts(frame: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for target in ETIOLOGY_TARGETS:
        outcome = pd.to_numeric(frame[target.outcome_column], errors="coerce")
        positive = int(outcome.sum())
        total = int(outcome.notna().sum())
        rows.append(
            {
                "target": target.slug,
                "outcome_column": target.outcome_column,
                "positive_label": target.positive_label,
                "source_columns": ";".join(target.source_columns),
                "n": total,
                "positive_rows": positive,
                "negative_rows": total - positive,
                "positive_rate": float(positive / total) if total else "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def normalize_label(value: object) -> str:
    text = str(value or "").strip().upper()
    for old, new in [("_", " "), ("-", " "), ("/", " ")]:
        text = text.replace(old, new)
    return " ".join(text.split())


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def af_subtype_label(value: object) -> str:
    return AF_SUBTYPE_CODE_LABELS.get(clean_text(value), "")


def direct_etiology_label(row: pd.Series) -> str:
    for column, label in DIRECT_ETIOLOGY_LABELS.items():
        if positive_flag(row.get(column, "")):
            return label

    phenotype = normalize_label(row.get("af_phenotype", ""))
    if phenotype == "KAF":
        return "KAF"
    if phenotype == "AFDAS":
        return "AFDAS"
    if phenotype in {"ECG AF", "ECGAF", "NEW ECG AF", "NEW ECG DETECTED AF"}:
        return "New_ECG_AF"

    esus_group = normalize_label(row.get("esus_group", ""))
    if esus_group == "ESUS":
        return "ESUS"
    return ""


def positive_flag(value: object) -> bool:
    return normalize_label(value) in {"1", "YES", "TRUE"}


def value_counts_dict(values: pd.Series | None) -> dict[str, int]:
    if values is None:
        return {}
    cleaned = values.fillna("").astype(str).str.strip().replace("", "(blank)")
    return {str(key): int(value) for key, value in cleaned.value_counts(dropna=False).items()}


def classify_ce_vs_esus(value: object) -> int | None:
    label = normalize_label(value)
    if label in NEGATIVE_MECHANISM_LABELS:
        return 0
    if label in POSITIVE_MECHANISM_LABELS:
        return 1
    return None


def review_group(row: pd.Series) -> str:
    source_label = clean_text(row.get(SOURCE_ETIOLOGY_LABEL_COLUMN, ""))
    if source_label in {*AF_SUBTYPE_LABEL_CODES, "ESUS"}:
        return source_label

    explicit_subtype = clean_text(row.get(AF_SUBTYPE_LABEL_COLUMN, ""))
    if explicit_subtype in set(AF_SUBTYPE_CODE_LABELS.values()):
        return explicit_subtype

    mechanism = clean_text(row.get(MECHANISM_COLUMN, ""))
    mechanism_label = normalize_label(mechanism)
    af = clean_text(row.get(ATRIAL_FIBRILLATION_COLUMN, ""))
    if mechanism_label in NEGATIVE_MECHANISM_LABELS:
        return "ESUS"
    if mechanism_label in AFDAS_LABELS:
        return "AFDAS"
    if mechanism_label in KAF_LABELS:
        return "KAF"
    if mechanism_label == POSITIVE_LABEL.upper():
        if af == "1":
            return "Cardioembolic_untyped_AF_present"
        if af == "0":
            return "Cardioembolic_untyped_AF_not_recorded_or_non_AF_source"
        return "Cardioembolic_untyped_AF_unknown"
    return f"Other_or_unmodeled:{mechanism or '(blank)'}"


def write_etiology_group_summary(analysis_path: Path, scores_path: Path, output_path: Path) -> None:
    analysis = pd.read_csv(analysis_path, dtype=str)
    scores = pd.read_csv(scores_path, dtype=str)
    clinical_columns = [
        column
        for column in [
            score_builder.CASE_ID,
            RECORD_ID_COLUMN,
            MECHANISM_COLUMN,
            ATRIAL_FIBRILLATION_COLUMN,
            SOURCE_ETIOLOGY_LABEL_COLUMN,
            AF_SUBTYPE_CODE_COLUMN,
            AF_SUBTYPE_LABEL_COLUMN,
            "KAF",
            "AFDAS",
            "ECG_AF",
            "ESUS",
            "KAF_Off_OACs",
            "af_phenotype",
            "esus_group",
            "redcap_af",
            "redcap_aftype",
            "af_burden",
            "afdas_burden",
            "AF_Burden2",
            "AFDAS_Burden",
            "Unified_AFDAS_burden",
        ]
        if column in analysis.columns
    ]
    clinical = analysis[clinical_columns].drop_duplicates(score_builder.CASE_ID)
    merged = scores.merge(clinical, on=score_builder.CASE_ID, how="left", suffixes=("", "_clinical"))
    if f"{MECHANISM_COLUMN}_clinical" in merged.columns:
        merged[MECHANISM_COLUMN] = merged[f"{MECHANISM_COLUMN}_clinical"].fillna(merged.get(MECHANISM_COLUMN, ""))
    if f"{ATRIAL_FIBRILLATION_COLUMN}_clinical" in merged.columns:
        merged[ATRIAL_FIBRILLATION_COLUMN] = merged[f"{ATRIAL_FIBRILLATION_COLUMN}_clinical"].fillna(
            merged.get(ATRIAL_FIBRILLATION_COLUMN, "")
        )
    clinical_passthrough_columns = [
        SOURCE_ETIOLOGY_LABEL_COLUMN,
        AF_SUBTYPE_CODE_COLUMN,
        AF_SUBTYPE_LABEL_COLUMN,
        "KAF",
        "AFDAS",
        "ECG_AF",
        "ESUS",
        "KAF_Off_OACs",
        "af_phenotype",
        "esus_group",
        "redcap_af",
        "redcap_aftype",
        "af_burden",
        "afdas_burden",
        "AF_Burden2",
        "AFDAS_Burden",
        "Unified_AFDAS_burden",
    ]
    for column in clinical_passthrough_columns:
        clinical_column = f"{column}_clinical"
        if clinical_column in merged.columns:
            merged[column] = merged[clinical_column].fillna(merged.get(column, ""))
    merged[ETIOLOGY_REVIEW_GROUP_COLUMN] = merged.apply(review_group, axis=1)

    score_columns = [column for column in SUMMARY_SCORE_COLUMNS if column in merged.columns]
    rows = []
    for group_name, group in merged.groupby(ETIOLOGY_REVIEW_GROUP_COLUMN, dropna=False):
        outcome = pd.to_numeric(group.get(OUTCOME_COLUMN), errors="coerce")
        row: dict[str, object] = {
            ETIOLOGY_REVIEW_GROUP_COLUMN: group_name,
            "stroke_mechanism_values": ";".join(sorted(clean_values(group.get(MECHANISM_COLUMN)))),
            "atrial_fibrillation_values": ";".join(sorted(clean_values(group.get(ATRIAL_FIBRILLATION_COLUMN)))),
            "source_etiology_label_values": ";".join(sorted(clean_values(group.get(SOURCE_ETIOLOGY_LABEL_COLUMN)))),
            "af_subtype_label_values": ";".join(sorted(clean_values(group.get(AF_SUBTYPE_LABEL_COLUMN)))),
            "kaf_or_afdas_event_values": ";".join(sorted(clean_values(group.get(AF_SUBTYPE_CODE_COLUMN)))),
            "KAF_values": ";".join(sorted(clean_values(group.get("KAF")))),
            "AFDAS_values": ";".join(sorted(clean_values(group.get("AFDAS")))),
            "ECG_AF_values": ";".join(sorted(clean_values(group.get("ECG_AF")))),
            "ESUS_values": ";".join(sorted(clean_values(group.get("ESUS")))),
            "redcap_af_values": ";".join(sorted(clean_values(group.get("redcap_af")))),
            "n": int(len(group)),
            "cardioembolic_events": int(outcome.fillna(0).sum()) if not outcome.empty else "",
            "cardioembolic_event_rate": float(outcome.mean()) if outcome.notna().any() else "",
            "interpretation_note": group_interpretation_note(str(group_name)),
        }
        for column in score_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}__n"] = int(len(values))
            row[f"{column}__mean"] = float(values.mean()) if not values.empty else ""
            row[f"{column}__median"] = float(values.median()) if not values.empty else ""
            row[f"{column}__q25"] = float(values.quantile(0.25)) if not values.empty else ""
            row[f"{column}__q75"] = float(values.quantile(0.75)) if not values.empty else ""
            row[f"{column}__min"] = float(values.min()) if not values.empty else ""
            row[f"{column}__max"] = float(values.max()) if not values.empty else ""
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["n", ETIOLOGY_REVIEW_GROUP_COLUMN], ascending=[False, True])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def clean_values(values: pd.Series | None) -> set[str]:
    if values is None:
        return set()
    cleaned = values.fillna("").astype(str).str.strip().replace("", "(blank)")
    return set(cleaned)


def group_interpretation_note(group_name: str) -> str:
    if group_name == "Cardioembolic_untyped_AF_present":
        return "Cardioembolic mechanism with AF recorded, but no explicit KAF/AFDAS event code in the joined subtype source."
    if group_name == "Cardioembolic_untyped_AF_not_recorded_or_non_AF_source":
        return "Cardioembolic mechanism with atrial_fibrillation recorded as 0; do not treat as AFDAS without a specific source field."
    if group_name == "Cardioembolic_untyped_AF_unknown":
        return "Cardioembolic mechanism without a specific AF subtype code in the joined subtype source."
    if group_name in {"AFDAS", "KAF", "New_ECG_AF"}:
        return "Direct etiology label from the joined source CSV."
    if group_name == "ESUS":
        return "Direct ESUS source label when present; otherwise coarse stroke_mechanism fallback."
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--af-subtype-source",
        type=Path,
        default=DEFAULT_AF_SUBTYPE_SOURCE,
        help="REDCap export carrying kaf_or_afdas_event; joined by record_id for explicit KAF/AFDAS grouping.",
    )
    return parser


if __name__ == "__main__":
    main()
