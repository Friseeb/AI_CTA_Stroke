#!/usr/bin/env python
"""Run aorta CTA stages from a manifest with per-stage process workers.

The staged runner keeps high-memory stages in separate processes and writes
each case to an isolated output directory. That avoids CSV/mask write races
when multiple cases are processed in parallel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
AORTA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NV_PYTHON = Path("/opt/anaconda3/envs/nv-segment-ct/bin/python")
sys.path.insert(0, str(AORTA_ROOT / "src"))

from aorta_cta_radiomics.metadata_filter import evaluate_neuro_cta_metadata

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


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_path: str
    aorta_mask_path: str = ""


@dataclass(frozen=True)
class StageResult:
    case_id: str
    stage: str
    detail: str
    status: str
    command: str
    log_path: str
    output_path: str = ""
    start_time_utc: str = ""
    end_time_utc: str = ""
    returncode: int = 0
    error: str = ""


@dataclass(frozen=True)
class RadiomicsRegionTask:
    case: CaseRecord
    region: str
    output_name: str


ProgressCallback = Callable[[StageResult, list[StageResult], int], None]


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate_case_csvs(outdir / "cases", outdir)
        print(f"Aggregated CSVs: {outdir / 'features'} and {outdir / 'qc'}")
        return
    cases, metadata_eligibility = _load_manifest(
        args.manifest,
        args.case_id,
        metadata_filter=args.metadata_filter,
        metadata_include_keywords=args.metadata_include_keyword,
        metadata_exclude_keywords=args.metadata_exclude_keyword,
        allow_missing_metadata=bool(args.allow_missing_metadata),
    )
    if metadata_eligibility:
        metadata_path = outdir / "metadata_eligibility.csv"
        pd.DataFrame(metadata_eligibility).to_csv(metadata_path, index=False)
        eligible_count = sum(1 for row in metadata_eligibility if row.get("eligible") is True)
        print(f"Metadata filter kept {eligible_count}/{len(metadata_eligibility)} case(s): {metadata_path}")
    if not cases:
        raise SystemExit("No cases to process after manifest and metadata filtering.")

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    all_results: list[StageResult] = []
    status_path = outdir / "stage_status.csv"
    for stage in stages:
        workers = _stage_workers(args, stage)
        print(f"Running stage {stage!r} for {len(cases)} cases with workers={workers}")
        progress_callback = _status_progress_callback(all_results, status_path)
        if stage == "radiomics" and args.radiomics_split_by_region:
            stage_results = _run_radiomics_split_stage(cases, args, outdir, progress_callback)
        else:
            stage_results = _run_stage(cases, stage, args, outdir, workers, progress_callback)
        all_results.extend(stage_results)
        _write_status_csv(all_results, status_path)
        failed = [result for result in stage_results if result.status != "ok"]
        if failed and not args.keep_going:
            raise SystemExit(f"Stage {stage} failed for {len(failed)} case(s). See {status_path}")

    if not args.no_aggregate:
        _aggregate_case_csvs(outdir / "cases", outdir)
    _write_status_csv(all_results, status_path)
    print(f"Done. Stage status: {status_path}")
    if args.no_aggregate:
        print("Skipped final aggregation because --no-aggregate was set.")
    else:
        print(f"Aggregated CSVs: {outdir / 'features'} and {outdir / 'qc'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--stages",
        default="base,calcium,fat-wall,protrusions,wall-thickness",
        help="Comma-separated stages: vista,base,calcium,fat-wall,protrusions,analysis,radiomics,wall-thickness,qc.",
    )
    parser.add_argument("--config", type=Path, default=AORTA_ROOT / "configs" / "calcium_dynamic_500hu.yaml")
    parser.add_argument("--case-id", action="append", default=[], help="Restrict to one or more case IDs.")
    parser.add_argument("--workers", type=int, default=1, help="Default workers for stages.")
    parser.add_argument("--vista-workers", type=int, default=1)
    parser.add_argument("--base-workers", type=int, default=2)
    parser.add_argument("--calcium-workers", type=int, default=2)
    parser.add_argument("--fat-wall-workers", type=int, default=1)
    parser.add_argument("--protrusion-workers", type=int, default=1)
    parser.add_argument("--analysis-workers", type=int, default=1)
    parser.add_argument("--radiomics-workers", type=int, default=1)
    parser.add_argument(
        "--radiomics-split-by-region",
        action="store_true",
        help="Run each radiomics ROI as a separate subprocess and combine per-case outputs.",
    )
    parser.add_argument(
        "--radiomics-region-workers",
        type=int,
        default=None,
        help="Workers for --radiomics-split-by-region. Defaults to --radiomics-workers.",
    )
    parser.add_argument("--wall-thickness-workers", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip final cross-case CSV aggregation. Per-case outputs are still written.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate existing per-case CSVs and exit without running stages.",
    )

    parser.add_argument("--python", default=sys.executable, help="Python executable for aorta stages.")
    parser.add_argument("--nv-python", default=str(DEFAULT_NV_PYTHON), help="Python executable for NV-Segment-CT.")
    parser.add_argument("--nv-script", default=str(REPO_ROOT / "scripts" / "run_nv_segment_ct_laa.py"))
    parser.add_argument("--nv-model-dir", default=str(REPO_ROOT / "external" / "nv_segment_ct"))
    parser.add_argument("--nv-device", default="auto")
    parser.add_argument("--vista-label-id", type=int, default=6)

    parser.add_argument("--crop-margin-mm", type=float, default=8.0)
    parser.add_argument("--risk-thickness-threshold-mm", type=float, default=4.0)
    parser.add_argument("--reviewer", default=os.environ.get("USER", "reviewer"))
    parser.add_argument("--open-slicer", action="store_true")
    parser.add_argument(
        "--metadata-filter",
        choices=["none", "neuro-cta"],
        default="none",
        help="Optional manifest/JSON metadata eligibility filter before processing.",
    )
    parser.add_argument(
        "--metadata-include-keyword",
        action="append",
        default=[],
        help="Extra neuro/stroke inclusion keyword for --metadata-filter neuro-cta. Repeatable.",
    )
    parser.add_argument(
        "--metadata-exclude-keyword",
        action="append",
        default=[],
        help="Extra non-target exclusion keyword for --metadata-filter neuro-cta. Repeatable.",
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="With --metadata-filter neuro-cta, process rows lacking metadata instead of skipping them.",
    )
    return parser.parse_args()


def _load_manifest(
    path: Path,
    case_ids: list[str],
    metadata_filter: str = "none",
    metadata_include_keywords: list[str] | None = None,
    metadata_exclude_keywords: list[str] | None = None,
    allow_missing_metadata: bool = False,
) -> tuple[list[CaseRecord], list[dict[str, object]]]:
    frame = pd.read_csv(path)
    required = {"case_id", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    if case_ids:
        frame = frame[frame["case_id"].astype(str).isin({str(case_id) for case_id in case_ids})]
    records: list[CaseRecord] = []
    metadata_rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        if metadata_filter == "neuro-cta":
            eligibility = evaluate_neuro_cta_metadata(
                row,
                manifest_base=path.parent,
                include_keywords=metadata_include_keywords or [],
                exclude_keywords=metadata_exclude_keywords or [],
                allow_missing_metadata=allow_missing_metadata,
            )
            metadata_rows.append(eligibility.as_dict())
            if not eligibility.eligible:
                continue
        elif metadata_filter != "none":
            raise ValueError(f"Unsupported metadata filter: {metadata_filter}")

        records.append(
            CaseRecord(
                case_id=_cell_as_str(row.get("case_id")),
                image_path=_cell_as_str(row.get("image_path")),
                aorta_mask_path=_cell_as_str(row.get("aorta_mask_path", "")),
            )
        )
    return records, metadata_rows


def _cell_as_str(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _stage_workers(args: argparse.Namespace, stage: str) -> int:
    if stage == "vista":
        return max(int(args.vista_workers), 1)
    if stage == "base":
        return max(int(args.base_workers), 1)
    if stage == "calcium":
        return max(int(args.calcium_workers), 1)
    if stage == "fat-wall":
        return max(int(args.fat_wall_workers), 1)
    if stage == "protrusions":
        return max(int(args.protrusion_workers), 1)
    if stage == "analysis":
        return max(int(args.analysis_workers), 1)
    if stage == "radiomics":
        return max(int(args.radiomics_workers), 1)
    if stage == "wall-thickness":
        return max(int(args.wall_thickness_workers), 1)
    return max(int(args.workers), 1)


def _run_stage(
    cases: list[CaseRecord],
    stage: str,
    args: argparse.Namespace,
    outdir: Path,
    workers: int,
    progress_callback: ProgressCallback | None = None,
) -> list[StageResult]:
    total = len(cases)
    if workers == 1:
        results: list[StageResult] = []
        for index, case in enumerate(cases, start=1):
            print(f"[{stage}] start {index}/{total} {case.case_id}", flush=True)
            result = _run_one_stage(case, stage, args, outdir)
            results.append(result)
            _print_stage_progress(result, len(results), total)
            if progress_callback:
                progress_callback(result, results, total)
        return results
    results: list[StageResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one_stage, case, stage, args, outdir) for case in cases]
        for future in as_completed(futures):
            results.append(future.result())
            _print_stage_progress(results[-1], len(results), total)
            if progress_callback:
                progress_callback(results[-1], results, total)
    return sorted(results, key=lambda result: result.case_id)


def _run_radiomics_split_stage(
    cases: list[CaseRecord],
    args: argparse.Namespace,
    outdir: Path,
    progress_callback: ProgressCallback | None = None,
) -> list[StageResult]:
    config_path = _stage_config_path(args=args, outdir=outdir, stage="radiomics")
    regions = _radiomics_regions(config_path)
    if not regions:
        raise ValueError("Radiomics split requested, but no radiomics.regions are configured.")
    workers = max(int(args.radiomics_region_workers or args.radiomics_workers), 1)
    tasks = [
        RadiomicsRegionTask(
            case=case,
            region=region,
            output_name=f"radiomics_features__{_safe_token(region)}.csv",
        )
        for case in cases
        for region in regions
    ]
    print(f"Running split radiomics for {len(tasks)} case-region task(s) with workers={workers}")
    total = len(tasks)
    if workers == 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            print(f"[radiomics-region] start {index}/{total} {task.case.case_id} {task.region}", flush=True)
            result = _run_one_radiomics_region(task, args, outdir, config_path)
            results.append(result)
            _print_stage_progress(result, len(results), total)
            if progress_callback:
                progress_callback(result, results, total)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one_radiomics_region, task, args, outdir, config_path) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                _print_stage_progress(results[-1], len(results), total)
                if progress_callback:
                    progress_callback(results[-1], results, total)
    _combine_split_radiomics(cases, outdir, regions)
    return sorted(results, key=lambda result: (result.case_id, result.detail))


def _run_one_radiomics_region(
    task: RadiomicsRegionTask,
    args: argparse.Namespace,
    outdir: Path,
    config_path: Path,
) -> StageResult:
    case = task.case
    stage = "radiomics-region"
    stage_dir = outdir / "logs" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{case.case_id}__{_safe_token(task.region)}.log"
    case_outdir = outdir / "cases" / case.case_id
    output_path = str(case_outdir / "features" / task.output_name)
    command = [
        str(args.python),
        str(AORTA_ROOT / "scripts" / "run_radiomics_case.py"),
        "--image",
        case.image_path,
        "--case-id",
        case.case_id,
        "--outdir",
        str(case_outdir),
        "--config",
        str(config_path),
        "--region",
        task.region,
        "--output-name",
        task.output_name,
        "--no-rebuild-wide",
    ]
    start = _now()
    try:
        if args.skip_existing and Path(output_path).exists():
            return StageResult(
                case_id=case.case_id,
                stage=stage,
                detail=task.region,
                status="ok",
                command=" ".join(command),
                log_path=str(log_path),
                output_path=output_path,
                start_time_utc=start,
                end_time_utc=_now(),
            )
        if args.dry_run:
            log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
            return StageResult(
                case_id=case.case_id,
                stage=stage,
                detail=task.region,
                status="ok",
                command=" ".join(command),
                log_path=str(log_path),
                output_path=output_path,
                start_time_utc=start,
                end_time_utc=_now(),
            )
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        return StageResult(
            case_id=case.case_id,
            stage=stage,
            detail=task.region,
            status="ok" if result.returncode == 0 else "failed",
            command=" ".join(command),
            log_path=str(log_path),
            output_path=output_path,
            start_time_utc=start,
            end_time_utc=_now(),
            returncode=int(result.returncode),
            error="" if result.returncode == 0 else f"exit={result.returncode}",
        )
    except Exception as exc:  # noqa: BLE001
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return StageResult(
            case_id=case.case_id,
            stage=stage,
            detail=task.region,
            status="failed",
            command=" ".join(command),
            log_path=str(log_path),
            output_path=output_path,
            start_time_utc=start,
            end_time_utc=_now(),
            returncode=1,
            error=str(exc),
        )


def _run_one_stage(case: CaseRecord, stage: str, args: argparse.Namespace, outdir: Path) -> StageResult:
    stage_dir = outdir / "logs" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{case.case_id}.log"
    start = _now()
    try:
        command, output_path = _stage_command(case, stage, args, outdir)
        if args.skip_existing and output_path and Path(output_path).exists():
            return StageResult(
                case_id=case.case_id,
                stage=stage,
                detail="",
                status="ok",
                command=" ".join(command),
                log_path=str(log_path),
                output_path=output_path,
                start_time_utc=start,
                end_time_utc=_now(),
            )
        if args.dry_run:
            log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
            return StageResult(
                case_id=case.case_id,
                stage=stage,
                detail="",
                status="ok",
                command=" ".join(command),
                log_path=str(log_path),
                output_path=output_path,
                start_time_utc=start,
                end_time_utc=_now(),
            )
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        status = "ok" if result.returncode == 0 else "failed"
        return StageResult(
            case_id=case.case_id,
            stage=stage,
            detail="",
            status=status,
            command=" ".join(command),
            log_path=str(log_path),
            output_path=output_path,
            start_time_utc=start,
            end_time_utc=_now(),
            returncode=int(result.returncode),
            error="" if result.returncode == 0 else f"exit={result.returncode}",
        )
    except Exception as exc:  # noqa: BLE001
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return StageResult(
            case_id=case.case_id,
            stage=stage,
            detail="",
            status="failed",
            command="",
            log_path=str(log_path),
            start_time_utc=start,
            end_time_utc=_now(),
            returncode=1,
            error=str(exc),
        )


def _stage_command(
    case: CaseRecord,
    stage: str,
    args: argparse.Namespace,
    outdir: Path,
) -> tuple[list[str], str]:
    case_outdir = outdir / "cases" / case.case_id
    vista_mask = outdir / "vista_aorta" / case.case_id / f"{case.case_id}_aorta6.nii.gz"
    aorta_mask = Path(case.aorta_mask_path) if case.aorta_mask_path else vista_mask
    if stage == "vista":
        output_path = str(vista_mask)
        return (
            [
                str(args.nv_python),
                str(args.nv_script),
                "--input",
                case.image_path,
                "--output",
                output_path,
                "--label-id",
                str(args.vista_label_id),
                "--model-dir",
                str(args.nv_model_dir),
                "--device",
                str(args.nv_device),
            ],
            output_path,
        )
    if stage == "base":
        output_path = str(case_outdir / "masks" / case.case_id / f"{case.case_id}_aorta_mask_cleaned.nii.gz")
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "run_base_case.py"),
                "--image",
                case.image_path,
                "--aorta-mask",
                str(aorta_mask),
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--config",
                str(args.config),
            ],
            output_path,
        )
    if stage == "calcium":
        output_path = str(
            case_outdir
            / "masks"
            / case.case_id
            / f"{case.case_id}_calcification_aorta_wall_dynamic_seed500HU.nii.gz"
        )
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "run_calcium_case.py"),
                "--image",
                case.image_path,
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--config",
                str(args.config),
                "--crop-margin-mm",
                str(args.crop_margin_mm),
            ],
            output_path,
        )
    if stage == "fat-wall":
        output_path = str(
            case_outdir
            / "masks"
            / case.case_id
            / f"{case.case_id}_aortic_wall_candidate_from_fat_lumen.nii.gz"
        )
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "run_fat_wall_case.py"),
                "--image",
                case.image_path,
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--config",
                str(args.config),
                "--input-aorta-mask",
                str(aorta_mask),
                "--crop-margin-mm",
                str(args.crop_margin_mm),
            ],
            output_path,
        )
    if stage == "protrusions":
        output_path = str(case_outdir / "features" / "lumen_protrusion_candidates.csv")
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "run_protrusions_case.py"),
                "--image",
                case.image_path,
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--config",
                str(args.config),
            ],
            output_path,
        )
    if stage == "analysis":
        output_path = str(case_outdir / "features" / "modeling_wide_features.csv")
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "run_single_case.py"),
                "--image",
                case.image_path,
                "--aorta-mask",
                str(aorta_mask),
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--config",
                str(_stage_config_path(args=args, outdir=outdir, stage="analysis")),
            ],
            output_path,
        )
    if stage == "radiomics":
        output_path = str(case_outdir / "features" / "radiomics_features.csv")
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "run_radiomics_case.py"),
                "--image",
                case.image_path,
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--config",
                str(_stage_config_path(args=args, outdir=outdir, stage="radiomics")),
            ],
            output_path,
        )
    if stage == "wall-thickness":
        mask_dir = case_outdir / "masks" / case.case_id
        threshold_suffix = f"{float(args.risk_thickness_threshold_mm):g}".replace(".", "p")
        output_path = str(
            mask_dir / f"{case.case_id}_wall_thickness_gt_{threshold_suffix}mm_TEE_analogue_labels.nii.gz"
        )
        return (
            [
                str(args.python),
                str(AORTA_ROOT / "scripts" / "measure_wall_thickness.py"),
                "--image",
                case.image_path,
                "--lumen-mask",
                str(mask_dir / f"{case.case_id}_aortic_wall_contrast_lumen_from_centerline_hu.nii.gz"),
                "--wall-mask",
                str(mask_dir / f"{case.case_id}_aortic_wall_candidate_from_fat_lumen.nii.gz"),
                "--calcium-mask",
                str(mask_dir / f"{case.case_id}_calcification_aorta_wall_dynamic_seed500HU.nii.gz"),
                "--subtract-calcium-from-lumen",
                "--add-calcium-to-wall",
                "--crop-margin-mm",
                str(args.crop_margin_mm),
                "--case-id",
                case.case_id,
                "--outdir",
                str(case_outdir),
                "--risk-thickness-threshold-mm",
                str(args.risk_thickness_threshold_mm),
            ],
            output_path,
        )
    if stage == "qc":
        manifest_path = outdir / "qc_manifests" / f"{case.case_id}.csv"
        qc_outdir = outdir / "qc_slicer" / case.case_id
        _write_stage_manifest([case], manifest_path, outdir)
        output_path = str(qc_outdir / "qc_slicer_selection.csv")
        command = [
            str(args.python),
            str(AORTA_ROOT / "scripts" / "qc_slicer.py"),
            "--manifest",
            str(manifest_path),
            "--outputs-root",
            str(outdir / "cases"),
            "--project-root",
            str(REPO_ROOT),
            "--outdir",
            str(qc_outdir),
            "--case-id",
            case.case_id,
            "--anatomy",
            "aorta",
            "--task",
            "all",
            "--reviewer",
            str(args.reviewer),
        ]
        if args.open_slicer:
            command.append("--open-slicer")
        return command, output_path
    raise ValueError(f"Unknown stage: {stage}")


def _aggregate_case_csvs(cases_root: Path, outdir: Path) -> None:
    for table_type in ["features", "qc"]:
        target_dir = outdir / table_type
        target_dir.mkdir(parents=True, exist_ok=True)
        names = sorted({path.name for path in cases_root.glob(f"*/{table_type}/*.csv")})
        for name in names:
            frames = []
            for path in sorted(cases_root.glob(f"*/{table_type}/{name}")):
                if path.stat().st_size == 0:
                    continue
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                frames.append(frame)
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(target_dir / name, index=False)
    _rebuild_aggregated_modeling_wide(outdir / "features")


def _combine_split_radiomics(cases: list[CaseRecord], outdir: Path, regions: list[str]) -> None:
    from aorta_cta_radiomics.stage_outputs import rebuild_modeling_wide

    for case in cases:
        features_dir = outdir / "cases" / case.case_id / "features"
        frames = []
        for region in regions:
            path = features_dir / f"radiomics_features__{_safe_token(region)}.csv"
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                continue
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(features_dir / "radiomics_features.csv", index=False)
        rebuild_modeling_wide(features_dir)


def _rebuild_aggregated_modeling_wide(features_dir: Path) -> None:
    from aorta_cta_radiomics.features import ensure_feature_columns, long_to_wide_features

    frames = []
    for name in LONG_FEATURE_FILES:
        path = features_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if {"case_id", "region", "feature_group", "feature_name", "feature_value"}.issubset(frame.columns):
            frames.append(ensure_feature_columns(frame))
    if frames:
        long_to_wide_features(pd.concat(frames, ignore_index=True)).to_csv(
            features_dir / "modeling_wide_features.csv",
            index=False,
        )


def _radiomics_regions(config_path: Path) -> list[str]:
    from aorta_cta_radiomics.config import load_config

    config = load_config(config_path)
    return [str(region) for region in config.get("radiomics", {}).get("regions", [])]


def _safe_token(text: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(text).strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "region"


def _write_stage_manifest(cases: list[CaseRecord], path: Path, outdir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "image_path", "aorta_mask_path"])
        writer.writeheader()
        for case in cases:
            vista_mask = outdir / "vista_aorta" / case.case_id / f"{case.case_id}_aorta6.nii.gz"
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "image_path": case.image_path,
                    "aorta_mask_path": case.aorta_mask_path or str(vista_mask),
                }
            )


def _stage_config_path(args: argparse.Namespace, outdir: Path, stage: str) -> Path:
    """Write an effective config for stages that need feature toggles."""
    requested_stages = {item.strip() for item in str(args.stages).split(",") if item.strip()}
    override: dict[str, object] = {}
    if stage == "analysis" and "radiomics" in requested_stages:
        override = {"radiomics": {"enabled": False}}
    elif stage == "radiomics":
        override = {"radiomics": {"enabled": True}}
    if not override:
        return Path(args.config)

    from aorta_cta_radiomics.config import deep_update, load_config

    config = deep_update(load_config(args.config), override)
    config_dir = outdir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{stage}_effective.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_status_csv(results: list[StageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(result) for result in results])
    frame.to_csv(path, index=False)
    (path.with_suffix(".json")).write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


def _status_progress_callback(prior_results: list[StageResult], status_path: Path) -> ProgressCallback:
    def callback(_result: StageResult, partial_results: list[StageResult], _total: int) -> None:
        _write_status_csv(prior_results + partial_results, status_path)

    return callback


def _print_stage_progress(result: StageResult, done: int, total: int) -> None:
    detail = f" {result.detail}" if result.detail else ""
    elapsed = _format_seconds(_elapsed_seconds(result.start_time_utc, result.end_time_utc))
    percent = 100.0 * done / total if total else 100.0
    message = (
        f"[{result.stage}{detail}] done {done}/{total} ({percent:5.1f}%) "
        f"{result.case_id}: {result.status} in {elapsed}"
    )
    if result.status != "ok":
        message += f" log={result.log_path}"
    print(message, flush=True)


def _elapsed_seconds(start_time_utc: str, end_time_utc: str) -> float:
    try:
        start = datetime.fromisoformat(start_time_utc)
        end = datetime.fromisoformat(end_time_utc)
        return max((end - start).total_seconds(), 0.0)
    except ValueError:
        return 0.0


def _format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
