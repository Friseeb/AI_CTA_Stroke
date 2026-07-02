#!/usr/bin/env python
"""Build a PHI-light SLAO MACE table joined to aorta CTA features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable


DEFAULT_REDCAP = Path("~/Downloads/BreakthroughStrokesS_DATA_2026-06-15_1451.csv").expanduser()
DEFAULT_FEATURES = Path("aorta_cta_radiomics/outputs/aorta_batch_run/features/modeling_wide_features.csv")
DEFAULT_MANIFEST = Path("aorta_cta_radiomics/outputs/manifests/slaobids_aorta_manifest.csv")
DEFAULT_OUTDIR = Path("aorta_cta_radiomics/outputs/aorta_batch_run/mace_slao")

REDCAP_ID = "record_id"
CASE_ID = "case_id"

DATA_SOURCE_LABELS = {
    "data_source___1": "LOSR",
    "data_source___2": "TAVI Database",
    "data_source___3": "Cardiothoracic Database",
    "data_source___4": "DS reports for LOSR",
    "data_source___5": "Other",
    "data_source___unk": "Unknown",
    "data_source___navu": "Not available",
    "data_source___ni": "No information",
}

OUTCOME_COLUMNS = [
    "source_cohort",
    "cta_date",
    "cta_protocol_type",
    "cta_extended",
    "age",
    "sex",
    "smoking_status",
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "coronary_artery_disease",
    "heart_failure",
    "atrial_fibrillation",
    "prior_stroke_tia",
    "main_diagnosis",
    "stroke_type",
    "stroke_confirmed",
    "stroke_mechanism",
    "mechanism_lvd",
    "mechanism_ce",
    "mechanism_svd",
    "mechanism_esus",
    "mechanism_other",
    "mechanism_undetermined",
    "lvo",
    "lvo_branch",
    "baseline_nihss",
    "iv_thrombolysis",
    "evt",
    "recurrent_ischemic_stroke",
    "recurrent_tia",
    "myocardial_infarction",
    "cardiovascular_hospitalization",
    "all_cause_death",
    "mace_composite",
    "last_follow_up_date",
    "follow_up_duration_days",
    "follow_up_timepoint",
    "study_arm",
    "outcomes_complete",
    "outcomes_checked_date",
    "doev",
    "index_date",
    "index_date_source",
    "date_last_seen",
    "fupd",
    "dod",
    "death_diff_days",
    "cause_of_death",
    "cardiovasc_death",
    "aci_yn",
    "aci_date",
    "acs_after_yn",
    "acs_after_date",
    "chf_after_yn",
    "chf_after_date",
    "sys_emb",
    "sys_emb_date",
    "recur",
    "dorecur",
    "major_bleed",
    "major_bleed_date",
    "gi_bleed",
    "gi_bleed_date",
    "ich",
    "gi_bleed_date_2",
    "mace_acute_coronary_infarction",
    "mace_acute_coronary_syndrome",
    "mace_recurrent_stroke_tia",
    "mace_systemic_embolism",
    "mace_cardiovascular_death",
    "mace_primary",
    "mace_primary_components",
    "mace_primary_date",
    "mace_primary_time_days",
    "mace_plus_heart_failure",
    "net_adverse_event",
]


def main() -> None:
    args = build_parser().parse_args()
    redcap_path = args.redcap_csv.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    outcome_records, outcome_summary = load_outcome_records(redcap_path, source_cohort=args.source_cohort)
    feature_rows, feature_header = read_csv(features_path)
    feature_columns = select_aorta_feature_columns(feature_header)
    feature_dictionary = [
        {"feature_name": name, "domain": feature_domain(name)}
        for name in feature_columns
    ]

    manifest_case_ids = set()
    if manifest_path and manifest_path.exists():
        manifest_rows, _ = read_csv(manifest_path)
        manifest_case_ids = {row.get(CASE_ID, "") for row in manifest_rows if row.get(CASE_ID)}

    merged_rows: list[dict[str, str]] = []
    unmatched_feature_cases: list[str] = []
    for feature_row in feature_rows:
        case_id = feature_row.get(CASE_ID, "")
        record_id = case_id.removeprefix("sub-")
        record = outcome_records.get(record_id)
        if record is None:
            unmatched_feature_cases.append(case_id)
            continue
        outcome = derive_outcomes(record)
        row = {
            CASE_ID: case_id,
            REDCAP_ID: record_id,
            "excluded": record.get("excluded", ""),
            "in_slaobids_manifest": flag(case_id in manifest_case_ids) if manifest_case_ids else "",
            "data_source_labels": ";".join(
                label
                for column, label in DATA_SOURCE_LABELS.items()
                if record.get(column) == "1"
            ),
        }
        for column in DATA_SOURCE_LABELS:
            row[column] = record.get(column, "")
        for column in OUTCOME_COLUMNS:
            row[column] = outcome.get(column, record.get(column, ""))
        for column in feature_columns:
            row[column] = feature_row.get(column, "")
        merged_rows.append(row)

    output_columns = [
        CASE_ID,
        REDCAP_ID,
        "excluded",
        "in_slaobids_manifest",
        "data_source_labels",
        *DATA_SOURCE_LABELS.keys(),
        *OUTCOME_COLUMNS,
        *feature_columns,
    ]
    analysis_path = outdir / "slao_mace_aorta_modeling.csv"
    write_csv(analysis_path, output_columns, merged_rows)

    outcomes_path = outdir / "slao_mace_outcomes.csv"
    write_csv(
        outcomes_path,
        [
            CASE_ID,
            REDCAP_ID,
            "excluded",
            "in_slaobids_manifest",
            "data_source_labels",
            *DATA_SOURCE_LABELS.keys(),
            *OUTCOME_COLUMNS,
        ],
        merged_rows,
    )

    feature_dictionary_path = outdir / "slao_aorta_feature_dictionary.csv"
    write_csv(feature_dictionary_path, ["feature_name", "domain"], feature_dictionary)

    univariate = univariate_feature_screen(
        merged_rows,
        feature_columns,
        outcome_column="mace_primary",
        min_n=args.univariate_min_n,
    )
    univariate_path = outdir / "univariate_mace_primary.csv"
    write_csv(univariate_path, list(univariate[0].keys()) if univariate else univariate_columns(), univariate)

    summary = build_summary(
        args=args,
        redcap_path=redcap_path,
        features_path=features_path,
        manifest_path=manifest_path,
        outcome_summary=outcome_summary,
        feature_rows=feature_rows,
        feature_columns=feature_columns,
        merged_rows=merged_rows,
        unmatched_feature_cases=unmatched_feature_cases,
        univariate=univariate,
    )
    summary_path = outdir / "slao_mace_aorta_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Outcome records: {summary['outcome_records']}")
    print(f"Aorta feature rows: {summary['feature_rows']}")
    print(f"Merged SLAO rows with outcomes: {summary['merged_rows']}")
    print(f"MACE primary events: {summary['mace_primary_events']}")
    print(f"Selected aorta feature columns: {summary['selected_feature_columns']}")
    print(f"Analysis table: {analysis_path}")
    print(f"Univariate screen: {univariate_path}")
    print(f"Summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redcap-csv", type=Path, default=DEFAULT_REDCAP)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--source-cohort",
        default=None,
        help="For final_cta_eda_dataset-style inputs, filter to this cohort. Defaults to SLAO_DAYLIGHT_BROAD when present.",
    )
    parser.add_argument(
        "--univariate-min-n",
        type=int,
        default=50,
        help="Minimum non-missing feature/outcome pairs required for the exploratory univariate screen.",
    )
    return parser


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_outcome_records(path: Path, *, source_cohort: str | None) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    rows, header = read_csv(path)
    if {"source_cohort", "mace_composite"}.issubset(header):
        return load_eda_outcome_records(rows, source_cohort=source_cohort)
    return load_redcap_records(rows)


def load_eda_outcome_records(
    rows: list[dict[str, str]],
    *,
    source_cohort: str | None,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    selected_cohort = source_cohort
    available_cohorts = sorted({row.get("source_cohort", "") for row in rows})
    if selected_cohort is None and "SLAO_DAYLIGHT_BROAD" in available_cohorts:
        selected_cohort = "SLAO_DAYLIGHT_BROAD"
    selected = [row for row in rows if selected_cohort is None or row.get("source_cohort", "") == selected_cohort]
    records: dict[str, dict[str, str]] = {}
    duplicate_record_ids: list[str] = []
    for row in selected:
        record_id = row.get(REDCAP_ID, "").strip()
        if not record_id:
            continue
        if record_id in records:
            duplicate_record_ids.append(record_id)
        records[record_id] = row
    return records, {
        "input_kind": "final_cta_eda_dataset",
        "redcap_rows": len(rows),
        "outcome_records": len(records),
        "source_cohort": selected_cohort or "",
        "available_source_cohorts": available_cohorts,
        "duplicate_record_ids_after_filter": sorted(set(duplicate_record_ids)),
    }


def load_redcap_records(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    primary: dict[str, dict[str, str]] = {}
    ct_dates_by_record: dict[str, list[str]] = defaultdict(list)
    repeat_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        record_id = row.get(REDCAP_ID, "").strip()
        if not record_id:
            continue
        instrument = row.get("redcap_repeat_instrument", "")
        repeat_counts[instrument or "primary"] += 1
        if row.get("date_ctheart"):
            ct_dates_by_record[record_id].append(row["date_ctheart"])
        if not instrument and not row.get("redcap_repeat_instance", ""):
            primary[record_id] = row

    for record_id, row in primary.items():
        ctheart_dates = sorted(
            date for date in ct_dates_by_record.get(record_id, []) if parse_date(date) is not None
        )
        row["ctheart_first_date"] = ctheart_dates[0] if ctheart_dates else ""
        index_date = first_nonempty(row.get("doev", ""), row.get("ctheart_first_date", ""))
        row["index_date"] = date_only(index_date)
        row["index_date_source"] = "doev" if row.get("doev") else ("ctheart_first_date" if row.get("ctheart_first_date") else "")

    return primary, {
        "input_kind": "redcap_repeating_export",
        "redcap_rows": len(rows),
        "outcome_records": len(primary),
        "repeat_counts": dict(sorted(repeat_counts.items())),
    }


def select_aorta_feature_columns(header: Iterable[str]) -> list[str]:
    columns: list[str] = []
    for name in header:
        if name == CASE_ID:
            continue
        domain = feature_domain(name)
        if domain:
            columns.append(name)
    return columns


def feature_domain(name: str) -> str:
    lower = name.lower()
    if "wall_thickness" in lower:
        return "wall_thickness"
    if "wall_from_fat" in lower or "experimental_wall_from_fat" in lower:
        return "wall_from_fat"
    if "calcium" in lower or "calcification" in lower:
        return "calcium"
    if "fat" in lower or "periaortic" in lower:
        return "fat"
    return ""


def derive_outcomes(record: dict[str, str]) -> dict[str, str]:
    if "mace_composite" in record:
        recurrent_ischemic_stroke = yes_no_flag(record.get("recurrent_ischemic_stroke", ""))
        recurrent_tia = yes_no_flag(record.get("recurrent_tia", ""))
        myocardial_infarction = yes_no_flag(record.get("myocardial_infarction", ""))
        cardiovascular_hospitalization = yes_no_flag(record.get("cardiovascular_hospitalization", ""))
        all_cause_death = yes_no_flag(record.get("all_cause_death", ""))
        mace_primary = yes_no_flag(record.get("mace_composite", ""))
        components = [
            ("recurrent_ischemic_stroke", recurrent_ischemic_stroke),
            ("recurrent_tia", recurrent_tia),
            ("myocardial_infarction", myocardial_infarction),
            ("cardiovascular_hospitalization", cardiovascular_hospitalization),
            ("all_cause_death", all_cause_death),
        ]
        return {
            "index_date": record.get("cta_date", ""),
            "index_date_source": "cta_date" if record.get("cta_date", "") else "",
            "mace_acute_coronary_infarction": myocardial_infarction,
            "mace_acute_coronary_syndrome": "",
            "mace_recurrent_stroke_tia": composite_flag([recurrent_ischemic_stroke, recurrent_tia]),
            "mace_systemic_embolism": "",
            "mace_cardiovascular_death": "",
            "mace_primary": mace_primary,
            "mace_primary_components": ";".join(name for name, value in components if value == "1"),
            "mace_primary_date": "",
            "mace_primary_time_days": record.get("follow_up_duration_days", "") if mace_primary == "1" else "",
            "mace_plus_heart_failure": composite_flag([mace_primary, yes_no_flag(record.get("heart_failure", ""))]),
            "net_adverse_event": mace_primary,
        }

    aci = yes_no_flag(record.get("aci_yn", ""))
    acs = yes_no_flag(record.get("acs_after_yn", ""))
    recurrent = yes_no_flag(record.get("recur", ""))
    systemic_embolism = yes_no_flag(record.get("sys_emb", ""))
    cardiovascular_death = cardiovascular_death_flag(record)
    heart_failure = yes_no_flag(record.get("chf_after_yn", ""))
    major_bleed = yes_no_flag(record.get("major_bleed", ""))
    gi_bleed = yes_no_flag(record.get("gi_bleed", ""))
    ich = yes_no_flag(record.get("ich", ""))

    primary_components = [
        ("acute_coronary_infarction", aci, record.get("aci_date", "")),
        ("acute_coronary_syndrome", acs, record.get("acs_after_date", "")),
        ("recurrent_stroke_tia", recurrent, record.get("dorecur", "")),
        ("systemic_embolism", systemic_embolism, record.get("sys_emb_date", "")),
        ("cardiovascular_death", cardiovascular_death, record.get("dod", "")),
    ]
    mace_primary = composite_flag(value for _, value, _ in primary_components)
    mace_primary_date = earliest_date(date for _, value, date in primary_components if value == "1")
    index_date = record.get("index_date", "")

    return {
        "index_date": record.get("index_date", ""),
        "index_date_source": record.get("index_date_source", ""),
        "mace_acute_coronary_infarction": aci,
        "mace_acute_coronary_syndrome": acs,
        "mace_recurrent_stroke_tia": recurrent,
        "mace_systemic_embolism": systemic_embolism,
        "mace_cardiovascular_death": cardiovascular_death,
        "mace_primary": mace_primary,
        "mace_primary_components": ";".join(name for name, value, _ in primary_components if value == "1"),
        "mace_primary_date": mace_primary_date,
        "mace_primary_time_days": days_between(index_date, mace_primary_date),
        "mace_plus_heart_failure": composite_flag([mace_primary, heart_failure]),
        "net_adverse_event": composite_flag([mace_primary, major_bleed, gi_bleed, ich]),
    }


def yes_no_flag(value: str) -> str:
    if str(value).strip() == "1":
        return "1"
    if str(value).strip() == "0":
        return "0"
    return ""


def cardiovascular_death_flag(record: dict[str, str]) -> str:
    if record.get("cause_of_death", "").strip() == "2" or record.get("cardiovasc_death", "").strip():
        return "1"
    if record.get("fupd", "").strip() == "0":
        return "0"
    if record.get("dod", "").strip() or record.get("fupd", "").strip() == "1":
        return "0"
    return ""


def composite_flag(values: Iterable[str]) -> str:
    values = list(values)
    if any(value == "1" for value in values):
        return "1"
    if values and all(value == "0" for value in values):
        return "0"
    if any(value == "" for value in values):
        return ""
    return "0"


def parse_date(value: str) -> datetime | None:
    value = str(value).strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def date_only(value: str) -> str:
    parsed = parse_date(value)
    return parsed.date().isoformat() if parsed else ""


def earliest_date(values: Iterable[str]) -> str:
    parsed = [parse_date(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return min(parsed).date().isoformat() if parsed else ""


def days_between(start: str, end: str) -> str:
    parsed_start = parse_date(start)
    parsed_end = parse_date(end)
    if parsed_start is None or parsed_end is None:
        return ""
    return f"{(parsed_end - parsed_start).days:.0f}"


def first_nonempty(*values: str) -> str:
    for value in values:
        if str(value).strip():
            return str(value).strip()
    return ""


def flag(value: bool) -> str:
    return "1" if value else "0"


def univariate_feature_screen(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    *,
    outcome_column: str,
    min_n: int,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for feature in feature_columns:
        pairs: list[tuple[float, int]] = []
        missing = 0
        for row in rows:
            outcome = row.get(outcome_column, "")
            value = parse_float(row.get(feature, ""))
            if outcome not in {"0", "1"}:
                continue
            if value is None:
                missing += 1
                continue
            pairs.append((value, int(outcome)))
        if len(pairs) < min_n:
            continue
        event_values = [value for value, outcome in pairs if outcome == 1]
        nonevent_values = [value for value, outcome in pairs if outcome == 0]
        if not event_values or not nonevent_values:
            continue
        event_mean = mean(event_values)
        nonevent_mean = mean(nonevent_values)
        pooled = pooled_std(event_values, nonevent_values)
        smd = (event_mean - nonevent_mean) / pooled if pooled > 0 else math.nan
        auc = rank_auc(pairs)
        results.append(
            {
                "feature_name": feature,
                "domain": feature_domain(feature),
                "n": str(len(pairs)),
                "n_events": str(len(event_values)),
                "n_nonevents": str(len(nonevent_values)),
                "n_missing": str(missing),
                "event_mean": format_float(event_mean),
                "nonevent_mean": format_float(nonevent_mean),
                "event_median": format_float(median(event_values)),
                "nonevent_median": format_float(median(nonevent_values)),
                "mean_difference": format_float(event_mean - nonevent_mean),
                "standardized_mean_difference": format_float(smd),
                "abs_standardized_mean_difference": format_float(abs(smd) if not math.isnan(smd) else math.nan),
                "auc": format_float(auc),
                "auc_directional": format_float(max(auc, 1.0 - auc)),
            }
        )
    return sorted(
        results,
        key=lambda row: (
            parse_float(row["auc_directional"]) or 0.0,
            parse_float(row["abs_standardized_mean_difference"]) or 0.0,
        ),
        reverse=True,
    )


def parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def pooled_std(a: list[float], b: list[float]) -> float:
    if len(a) + len(b) < 3:
        return 0.0
    var = ((len(a) - 1) * sample_std(a) ** 2 + (len(b) - 1) * sample_std(b) ** 2) / (len(a) + len(b) - 2)
    return math.sqrt(max(var, 0.0))


def rank_auc(pairs: list[tuple[float, int]]) -> float:
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1][0] == ordered[i][1][0]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = average_rank
        i = j
    event_ranks = [rank for rank, (_, outcome) in zip(ranks, pairs) if outcome == 1]
    n_event = len(event_ranks)
    n_nonevent = len(pairs) - n_event
    if n_event == 0 or n_nonevent == 0:
        return math.nan
    return (sum(event_ranks) - n_event * (n_event + 1) / 2.0) / (n_event * n_nonevent)


def format_float(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.8g}"


def univariate_columns() -> list[str]:
    return [
        "feature_name",
        "domain",
        "n",
        "n_events",
        "n_nonevents",
        "n_missing",
        "event_mean",
        "nonevent_mean",
        "event_median",
        "nonevent_median",
        "mean_difference",
        "standardized_mean_difference",
        "abs_standardized_mean_difference",
        "auc",
        "auc_directional",
    ]


def build_summary(
    *,
    args: argparse.Namespace,
    redcap_path: Path,
    features_path: Path,
    manifest_path: Path | None,
    outcome_summary: dict[str, object],
    feature_rows: list[dict[str, str]],
    feature_columns: list[str],
    merged_rows: list[dict[str, str]],
    unmatched_feature_cases: list[str],
    univariate: list[dict[str, str]],
) -> dict[str, object]:
    domain_counts: dict[str, int] = defaultdict(int)
    for feature in feature_columns:
        domain_counts[feature_domain(feature)] += 1
    event_count = sum(1 for row in merged_rows if row.get("mace_primary") == "1")
    non_event_count = sum(1 for row in merged_rows if row.get("mace_primary") == "0")
    if outcome_summary.get("input_kind") == "final_cta_eda_dataset":
        incomplete_count = sum(1 for row in merged_rows if row.get("mace_primary") not in {"0", "1"})
    else:
        incomplete_count = sum(1 for row in merged_rows if row.get("outcomes_complete") != "2")
    data_source_counts: dict[str, int] = defaultdict(int)
    excluded_counts: dict[str, int] = defaultdict(int)
    for row in merged_rows:
        data_source_counts[row.get("data_source_labels", "") or "(blank)"] += 1
        excluded_counts[row.get("excluded", "") or "(blank)"] += 1
    return {
        "redcap_csv": str(redcap_path),
        "features_csv": str(features_path),
        "manifest_csv": str(manifest_path) if manifest_path else "",
        "outdir": str(args.outdir.expanduser().resolve()),
        "input_kind": outcome_summary.get("input_kind", ""),
        "redcap_rows": outcome_summary.get("redcap_rows", 0),
        "outcome_records": outcome_summary.get("outcome_records", 0),
        "source_cohort": outcome_summary.get("source_cohort", ""),
        "available_source_cohorts": outcome_summary.get("available_source_cohorts", []),
        "duplicate_record_ids_after_filter": outcome_summary.get("duplicate_record_ids_after_filter", []),
        "redcap_repeat_counts": outcome_summary.get("repeat_counts", {}),
        "feature_rows": len(feature_rows),
        "merged_rows": len(merged_rows),
        "unmatched_feature_cases": unmatched_feature_cases,
        "unmatched_feature_case_count": len(unmatched_feature_cases),
        "selected_feature_columns": len(feature_columns),
        "selected_feature_domains": dict(sorted(domain_counts.items())),
        "mace_definition": mace_definition_for_summary(outcome_summary),
        "mace_primary_events": event_count,
        "mace_primary_nonevents": non_event_count,
        "outcomes_not_complete_count": incomplete_count,
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "data_source_label_counts": dict(sorted(data_source_counts.items())),
        "component_counts": {
            "acute_coronary_infarction": sum(1 for row in merged_rows if row.get("mace_acute_coronary_infarction") == "1"),
            "acute_coronary_syndrome": sum(1 for row in merged_rows if row.get("mace_acute_coronary_syndrome") == "1"),
            "recurrent_stroke_tia": sum(1 for row in merged_rows if row.get("mace_recurrent_stroke_tia") == "1"),
            "systemic_embolism": sum(1 for row in merged_rows if row.get("mace_systemic_embolism") == "1"),
            "cardiovascular_death": sum(1 for row in merged_rows if row.get("mace_cardiovascular_death") == "1"),
            "recurrent_ischemic_stroke": sum(1 for row in merged_rows if row.get("recurrent_ischemic_stroke") == "1"),
            "recurrent_tia": sum(1 for row in merged_rows if row.get("recurrent_tia") == "1"),
            "myocardial_infarction": sum(1 for row in merged_rows if row.get("myocardial_infarction") == "1"),
            "cardiovascular_hospitalization": sum(1 for row in merged_rows if row.get("cardiovascular_hospitalization") == "1"),
            "all_cause_death": sum(1 for row in merged_rows if row.get("all_cause_death") == "1"),
            "heart_failure": sum(1 for row in merged_rows if row.get("chf_after_yn") == "1"),
            "major_bleed": sum(1 for row in merged_rows if row.get("major_bleed") == "1"),
            "gi_bleed": sum(1 for row in merged_rows if row.get("gi_bleed") == "1"),
            "intracranial_hemorrhage": sum(1 for row in merged_rows if row.get("ich") == "1"),
        },
        "univariate_rows": len(univariate),
        "univariate_min_n": args.univariate_min_n,
        "top_univariate_features": univariate[:10],
    }


def mace_definition_for_summary(outcome_summary: dict[str, object]) -> dict[str, object]:
    if outcome_summary.get("input_kind") == "final_cta_eda_dataset":
        return {
            "mace_primary": "mace_composite from final_cta_eda_dataset.csv",
            "components_exposed": [
                "recurrent_ischemic_stroke",
                "recurrent_tia",
                "myocardial_infarction",
                "cardiovascular_hospitalization",
                "all_cause_death",
            ],
            "source_cohort_filter": outcome_summary.get("source_cohort", ""),
        }
    return {
        "mace_primary": [
            "aci_yn: acute coronary infarction",
            "acs_after_yn: acute coronary syndrome",
            "recur: recurrent ischemic stroke or TIA",
            "sys_emb: systemic embolism",
            "cause_of_death == 2 or cardiovasc_death non-empty: cardiovascular death",
        ],
        "mace_plus_heart_failure": "mace_primary OR chf_after_yn",
        "net_adverse_event": "mace_primary OR major_bleed OR gi_bleed OR ich",
    }


if __name__ == "__main__":
    main()
