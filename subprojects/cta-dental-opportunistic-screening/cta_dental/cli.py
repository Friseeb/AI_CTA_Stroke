"""CLI entry-point for cta-dental pipeline.

Commands:
  convert   DICOM folder → NIfTI + metadata sidecar
  roi       Detect dentition ROI from NIfTI CTA
  segment   Run dental segmentation backend
  features  Extract candidate markers from segmentation
  qc        Generate or regenerate QC images
  run       Full pipeline end-to-end
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import DISCLAIMER
from .config import PipelineConfig, load_config
from .dicom_io import (
    check_adult, load_dicom_series, load_nifti, save_nifti,
    write_metadata_sidecar, image_to_numpy_hu,
)
from .deid import run_deface, scrub_metadata
from .features import extract_features, write_features_json
from .logging_utils import configure_logging, get_logger
from .preprocess import preprocess
from .qc import (
    generate_failure_qc, generate_qc_summary_json,
    generate_roi_qc, generate_segmentation_qc,
    open_slicer_scene,
)
from .report import DentalReport, FOVCompleteness, PreprocessingRecord, write_report
from .roi import detect_roi
from .segmenters import get_segmenter

app = typer.Typer(
    name="cta-dental",
    help=f"Research prototype for opportunistic dental analysis from head/neck CTA.\n\n{DISCLAIMER}",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console(stderr=True)
log = get_logger("cli")

# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]
_ConfigOpt = Annotated[Optional[Path], typer.Option("--config", help="Path to config YAML.")]


def _setup(verbose: bool, config: Optional[Path]) -> PipelineConfig:
    configure_logging(verbose=verbose)
    cfg = load_config(config)
    console.print(f"[bold yellow]{DISCLAIMER}[/bold yellow]", highlight=False)
    return cfg


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

@app.command()
def convert(
    input: Annotated[Path, typer.Argument(help="DICOM folder.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output NIfTI path.")],
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Convert a DICOM series to NIfTI, preserving HU and writing a metadata sidecar."""
    cfg = _setup(verbose, config)
    if not input.is_dir():
        typer.echo(f"ERROR: {input} is not a directory.", err=True)
        raise typer.Exit(1)

    console.print(f"Loading DICOM from [cyan]{input}[/cyan] …")
    image, meta = load_dicom_series(input)

    age_status = check_adult(meta, cfg.preprocessing.min_patient_age_years)
    if age_status == "pediatric":
        console.print("[bold red]Excluded: pediatric patient (age < 18). Stopping.[/bold red]")
        result = {"status": "excluded_pediatric", "case_id": out.stem}
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(image, out)

    clean_meta = scrub_metadata({**meta, "age_status": age_status})
    sidecar = out.with_suffix("").with_suffix(".json")
    write_metadata_sidecar(clean_meta, sidecar)
    console.print(f"[green]Written:[/green] {out}  (sidecar: {sidecar})")


# ---------------------------------------------------------------------------
# roi
# ---------------------------------------------------------------------------

@app.command()
def roi(
    input: Annotated[Path, typer.Argument(help="NIfTI CTA file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")],
    roi_method: Annotated[str, typer.Option("--roi-method")] = "totalseg_teeth",
    target_spacing: Annotated[float, typer.Option("--target-spacing")] = 0.5,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Detect dentition ROI from NIfTI CTA and write cropped image + bbox + QC."""
    cfg = _setup(verbose, config)
    cfg.preprocessing.target_spacing_mm = target_spacing
    cfg.roi.method = roi_method  # type: ignore[assignment]

    console.print(f"Loading [cyan]{input}[/cyan] …")
    image = load_nifti(input)

    console.print("Preprocessing …")
    pre = preprocess(image, cfg.preprocessing)

    out.mkdir(parents=True, exist_ok=True)
    preprocessed_path = out / "preprocessed.nii.gz"
    save_nifti(pre.resampled, preprocessed_path)

    console.print(f"Running ROI detection: [bold]{roi_method}[/bold] …")
    roi_result = detect_roi(
        image=pre.resampled,
        cfg=cfg.roi,
        out_dir=out,
        seg_cfg=cfg.segmentation.model_dump(),
    )

    if not roi_result.success:
        console.print(f"[bold red]ROI detection failed:[/bold red] {roi_result.errors}")
        generate_failure_qc(
            image=pre.resampled,
            reason="; ".join(roi_result.errors),
            out_dir=out / "qc",
            cfg=cfg.qc,
        )
        raise typer.Exit(1)

    spacing_str = "×".join(f"{s:.2f}" for s in pre.resampled.GetSpacing())
    qc_paths = generate_roi_qc(
        image=pre.resampled,
        mask=roi_result.roi_mask,
        out_dir=out / "qc",
        cfg=cfg.qc,
        spacing_info=spacing_str,
        bbox_info=roi_result.bbox_voxel,
        image_path=preprocessed_path,
    )

    console.print(f"[green]ROI complete[/green]  quality={roi_result.roi_quality}  "
                  f"bbox={roi_result.bbox_voxel}")
    for w in roi_result.warnings:
        console.print(f"[yellow]WARN:[/yellow] {w}")


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

@app.command()
def segment(
    input: Annotated[Path, typer.Argument(help="NIfTI file (full CTA or dentition ROI).")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")],
    segmenter: Annotated[str, typer.Option("--segmenter")] = "totalseg_teeth",
    dentalseg_weights: Annotated[Optional[str], typer.Option("--dentalseg-weights")] = None,
    oralseg_model: Annotated[Optional[str], typer.Option("--oralseg-model-path")] = None,
    rail_model: Annotated[Optional[str], typer.Option("--rail-model-path")] = None,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Run a dental segmentation backend on a NIfTI file."""
    cfg = _setup(verbose, config)

    if dentalseg_weights:
        cfg.segmentation.dentalsegmentator.weights_path = dentalseg_weights
    if oralseg_model:
        cfg.segmentation.oralseg.model_path = oralseg_model
    if rail_model:
        cfg.segmentation.rail.model_path = rail_model

    seg = _build_segmenter(segmenter, cfg)
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"Running segmenter [bold]{seg.name}[/bold] on [cyan]{input}[/cyan] …")
    result = seg.run(input_nifti=input, output_dir=out, config=cfg.segmentation.model_dump())

    if not result.success:
        console.print("[bold red]Segmentation failed:[/bold red]")
        for err in result.errors:
            console.print(f"  {err}")
        raise typer.Exit(1)

    for w in result.domain_warnings:
        console.print(f"[yellow]DOMAIN WARNING:[/yellow] {w}")

    image = load_nifti(input)
    qc_paths = generate_segmentation_qc(
        image=image,
        label_files=result.label_files,
        out_dir=out / "qc",
        cfg=cfg.qc,
        segmenter_name=seg.name,
        image_path=input,
    )
    console.print(f"[green]Segmentation complete.[/green]  Labels: {list(result.label_files.keys())}")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@app.command()
def features(
    image: Annotated[Path, typer.Argument(help="NIfTI CTA (for HU extraction).")],
    labels_dir: Annotated[Path, typer.Option("--labels-dir", help="Directory with label NIfTIs.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output candidate_features.json.")],
    case_id: Annotated[str, typer.Option("--case-id")] = "unknown",
    roi_quality: Annotated[str, typer.Option("--roi-quality")] = "unknown",
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Extract candidate markers from segmentation outputs."""
    cfg = _setup(verbose, config)

    hu_image = load_nifti(image)
    label_files = {
        f.stem.replace(".nii", ""): f
        for f in sorted(labels_dir.glob("*.nii.gz"))
    }

    console.print(f"Extracting features for case [bold]{case_id}[/bold]  ({len(label_files)} labels) …")
    result = extract_features(
        case_id=case_id,
        hu_image=hu_image,
        label_files=label_files,
        cfg=cfg.features,
        roi_quality=roi_quality,
    )

    for w in result.warnings:
        console.print(f"[yellow]WARN:[/yellow] {w}")

    out_path = out if out.suffix == ".json" else out / "candidate_features.json"
    write_features_json(result, out_path)
    console.print(f"[green]Features written:[/green] {out_path}")


# ---------------------------------------------------------------------------
# qc (standalone regeneration)
# ---------------------------------------------------------------------------

@app.command()
def qc(
    image: Annotated[Path, typer.Argument(help="NIfTI CTA file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="QC output directory.")],
    roi_mask: Annotated[Optional[Path], typer.Option("--roi-mask")] = None,
    labels_dir: Annotated[Optional[Path], typer.Option("--labels-dir")] = None,
    segmenter: Annotated[str, typer.Option("--segmenter")] = "unknown",
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Generate or regenerate QC images from existing outputs."""
    cfg = _setup(verbose, config)
    sitk_image = load_nifti(image)
    mask_img = load_nifti(roi_mask) if roi_mask else None
    out.mkdir(parents=True, exist_ok=True)

    qc_paths = generate_roi_qc(
        image=sitk_image, mask=mask_img, out_dir=out, cfg=cfg.qc, image_path=image,
    )

    if labels_dir and labels_dir.is_dir():
        label_files = {f.stem.replace(".nii", ""): f for f in sorted(labels_dir.glob("*.nii.gz"))}
        # Look for candidate_features.json next to / above the QC output dir
        features_path: Optional[Path] = None
        for candidate in (
            out.parent / "candidate_features.json",
            out / "candidate_features.json",
        ):
            if candidate.is_file():
                features_path = candidate
                break
        seg_paths = generate_segmentation_qc(
            image=sitk_image, label_files=label_files,
            out_dir=out, cfg=cfg.qc, segmenter_name=segmenter,
            image_path=image, features_path=features_path,
        )
        qc_paths.update(seg_paths)

    generate_qc_summary_json(out, qc_paths, [], "unknown", segmenter)
    scene = qc_paths.get("scene", qc_paths.get("roi_scene"))
    if scene:
        console.print(f"[green]Slicer scene:[/green] {scene}")
        open_slicer_scene(scene)
    else:
        console.print(f"[green]QC written to[/green] {out}")


# ---------------------------------------------------------------------------
# run — full pipeline
# ---------------------------------------------------------------------------

@app.command()
def run(
    input: Annotated[Path, typer.Argument(help="DICOM folder or NIfTI file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output root directory.")],
    case_id: Annotated[str, typer.Option("--case-id")] = "unknown",
    roi_method: Annotated[str, typer.Option("--roi-method")] = "totalseg_teeth",
    segmenter_backend: Annotated[str, typer.Option("--segmenter")] = "totalseg_teeth",
    skip_existing: Annotated[bool, typer.Option(
        "--skip-existing/--no-skip-existing",
        help="Reuse a completed segmentation in the output dir instead of re-running the model.",
    )] = False,
    reuse_roi_seg: Annotated[bool, typer.Option(
        "--reuse-roi-seg/--no-reuse-roi-seg",
        help="When the ROI method and segmenter are the same TotalSegmentator task, reuse the "
             "ROI-detection labels as the final segmentation (cropped to the ROI) instead of "
             "running TotalSegmentator a second time. ~2x faster; results may differ slightly.",
    )] = False,
    target_spacing: Annotated[float, typer.Option("--target-spacing")] = 0.5,
    deface_mode: Annotated[str, typer.Option("--deface-mode")] = "mask_only",
    dentalseg_weights: Annotated[Optional[str], typer.Option("--dentalseg-weights")] = None,
    oralseg_model: Annotated[Optional[str], typer.Option("--oralseg-model-path")] = None,
    rail_model: Annotated[Optional[str], typer.Option("--rail-model-path")] = None,
    verbose: _VerboseOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """
    Run the full opportunistic dental analysis pipeline.

    [bold yellow]RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS[/bold yellow]
    """
    cfg = _setup(verbose, config)
    cfg.preprocessing.target_spacing_mm = target_spacing
    cfg.roi.method = roi_method  # type: ignore[assignment]
    cfg.deface.mode = deface_mode  # type: ignore[assignment]
    cfg.segmentation.backend = segmenter_backend  # type: ignore[assignment]
    if dentalseg_weights:
        cfg.segmentation.dentalsegmentator.weights_path = dentalseg_weights
    if oralseg_model:
        cfg.segmentation.oralseg.model_path = oralseg_model
    if rail_model:
        cfg.segmentation.rail.model_path = rail_model

    out.mkdir(parents=True, exist_ok=True)
    report = DentalReport(case_id=case_id)
    all_warnings: list[str] = []
    all_errors: list[str] = []
    qc_paths: dict[str, str] = {}

    # ── 1. Load ──────────────────────────────────────────────────────────────
    console.rule("[bold]Step 1/6 — Load Input[/bold]")
    dicom_meta: dict = {}
    if input.is_dir():
        console.print(f"Loading DICOM from [cyan]{input}[/cyan] …")
        try:
            image, dicom_meta = load_dicom_series(input)
            report.input_type = "dicom"
        except Exception as exc:
            console.print(f"[bold red]DICOM load failed:[/bold red] {exc}")
            report.status = "failed_load"
            report.errors = [str(exc)]
            write_report(report, out / "report.json")
            raise typer.Exit(1)

        age_status = check_adult(dicom_meta, cfg.preprocessing.min_patient_age_years)
        report.age_status = age_status  # type: ignore[assignment]
        if age_status == "pediatric":
            console.print("[bold red]Excluded: pediatric patient. Stopping.[/bold red]")
            report.status = "excluded_pediatric"
            write_report(report, out / "report.json")
            raise typer.Exit(0)
    else:
        console.print(f"Loading NIfTI [cyan]{input}[/cyan] …")
        try:
            image = load_nifti(input)
            report.input_type = "nifti"
        except Exception as exc:
            console.print(f"[bold red]NIfTI load failed:[/bold red] {exc}")
            report.status = "failed_load"
            report.errors = [str(exc)]
            write_report(report, out / "report.json")
            raise typer.Exit(1)

    report.input_path = str(input)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    console.rule("[bold]Step 2/6 — Preprocess[/bold]")
    console.print(f"Resampling to {target_spacing} mm isotropic, orientation {cfg.preprocessing.orientation} …")
    try:
        pre = preprocess(image, cfg.preprocessing, dicom_meta)
    except Exception as exc:
        console.print(f"[bold red]Preprocessing failed:[/bold red] {exc}")
        report.status = "failed_preprocess"
        report.errors = [str(exc)]
        write_report(report, out / "report.json")
        raise typer.Exit(1)

    report.original_spacing_xyz_mm = pre.meta.get("original_spacing_xyz_mm")
    report.target_spacing_xyz_mm = pre.meta.get("target_spacing_xyz_mm")
    report.preprocessing = PreprocessingRecord(**{
        k: v for k, v in pre.meta.items()
        if k in PreprocessingRecord.model_fields
    })

    preprocessed_path = out / "preprocessed.nii.gz"
    save_nifti(pre.resampled, preprocessed_path)
    clean_meta = scrub_metadata({**dicom_meta, **pre.meta})
    write_metadata_sidecar(clean_meta, out / "preprocessing_meta.json")

    # ── 3. Deface / de-ID ────────────────────────────────────────────────────
    console.rule("[bold]Step 3/6 — De-identification[/bold]")
    console.print(f"Deface mode: [bold]{deface_mode}[/bold]")
    protected_dir = out / "_protected"
    deface_result = run_deface(
        image=pre.resampled,
        cfg=cfg.deface,
        out_dir=out / "deid",
        protected_dir=protected_dir,
        image_path=preprocessed_path,
    )
    report.deface_mode = deface_mode
    report.deface_result = deface_result

    analysis_image = pre.resampled  # default: always use non-defaced for analysis
    if deface_mode == "pre" and "defaced_analysis_path" in deface_result:
        analysis_image = load_nifti(Path(deface_result["defaced_analysis_path"]))
        all_warnings.append("Analysis ran on pre-defaced image. Segmentation performance may differ.")

    # ── 4. ROI ───────────────────────────────────────────────────────────────
    console.rule("[bold]Step 4/6 — ROI Detection[/bold]")
    console.print(f"ROI method: [bold]{roi_method}[/bold] …")
    roi_out = out / "roi"
    roi_result = detect_roi(
        image=analysis_image,
        cfg=cfg.roi,
        out_dir=roi_out,
        seg_cfg=cfg.segmentation.model_dump(),
    )

    report.roi_method = roi_method
    report.roi_quality = roi_result.roi_quality  # type: ignore[assignment]
    report.roi_bbox_voxel = roi_result.bbox_voxel
    report.roi_bbox_physical = roi_result.bbox_physical
    if roi_result.fov_completeness:
        report.fov_completeness = FOVCompleteness(**roi_result.fov_completeness)
    all_warnings.extend(roi_result.warnings)
    all_errors.extend(roi_result.errors)

    if not roi_result.success:
        console.print(f"[bold red]ROI detection failed.[/bold red]")
        fail_img = generate_failure_qc(
            image=analysis_image,
            reason="; ".join(roi_result.errors),
            out_dir=out / "qc",
            cfg=cfg.qc,
        )
        qc_paths["roi_failure"] = str(fail_img)
        report.segmentation_status = "skipped"
        report.status = "failed_roi"
        report.warnings = all_warnings
        report.errors = all_errors
        report.qc_paths = qc_paths
        write_report(report, out / "report.json")
        console.print(f"[yellow]Report written to[/yellow] {out / 'report.json'}")
        raise typer.Exit(1)

    for w in roi_result.warnings:
        console.print(f"[yellow]WARN:[/yellow] {w}")

    roi_image = roi_result.roi_image or analysis_image
    spacing_str = "×".join(f"{s:.2f}" for s in analysis_image.GetSpacing())
    # ROI mask and full preprocessed image are in the same space
    roi_qc = generate_roi_qc(
        image=analysis_image,
        mask=roi_result.roi_mask,
        out_dir=out / "qc",
        cfg=cfg.qc,
        spacing_info=spacing_str,
        image_path=preprocessed_path,
    )
    qc_paths.update({k: str(v) for k, v in roi_qc.items()})

    # ── 5. Segmentation ──────────────────────────────────────────────────────
    console.rule("[bold]Step 5/6 — Segmentation[/bold]")
    seg_image_path = roi_out / "dentition_roi.nii.gz"
    if not seg_image_path.exists():
        seg_image_path = preprocessed_path

    if segmenter_backend == "none":
        console.print("Segmentation skipped (--segmenter none).")
        seg_result_label_files: dict[str, Path] = {}
        report.segmentation_status = "skipped"
        domain_warnings: list[str] = []
    else:
        seg = _build_segmenter(segmenter_backend, cfg)
        seg_out = out / "segmentations" / seg.name
        roi_seg_dir = roi_out / f"_tseg_{_ROI_TASK.get(roi_method, '')}"
        can_reuse_roi = (
            reuse_roi_seg
            and segmenter_backend == roi_method
            and segmenter_backend in _ROI_TASK
            and roi_seg_dir.is_dir()
            and any(roi_seg_dir.glob("*.nii.gz"))
        )
        seg_result = seg.load_existing(seg_out) if skip_existing else None
        if seg_result is not None:
            console.print(
                f"[green]Reusing existing {seg.name} segmentation[/green] "
                f"({len(seg_result.label_files)} labels) — --skip-existing."
            )
        elif can_reuse_roi:
            console.print(
                "[green]Reusing ROI-detection labels as final segmentation[/green] "
                "(--reuse-roi-seg; skips a 2nd TotalSegmentator run)."
            )
            seg_result = _reuse_roi_segmentation(roi_seg_dir, roi_image, seg, seg_out)
        else:
            console.print(f"Running segmenter [bold]{seg.name}[/bold] …")
            seg_result = seg.run(
                input_nifti=seg_image_path,
                output_dir=seg_out,
                config=cfg.segmentation.model_dump(),
            )
        domain_warnings = seg_result.domain_warnings
        for w in domain_warnings:
            console.print(f"[yellow]DOMAIN WARNING:[/yellow] {w}")

        report.segmentation_backend = seg.name
        report.domain_warnings = domain_warnings
        all_warnings.extend(domain_warnings)

        if seg_result.success:
            report.segmentation_status = "success"
            seg_result_label_files = seg_result.label_files
            all_errors.extend(seg_result.errors)
        else:
            report.segmentation_status = "failed"
            seg_result_label_files = {}
            all_errors.extend(seg_result.errors)
            console.print(f"[bold red]Segmentation failed:[/bold red]")
            for err in seg_result.errors:
                console.print(f"  {err}")

    # ── 6. Candidate Features ────────────────────────────────────────────────
    console.rule("[bold]Step 6/6 — Candidate Features[/bold]")
    console.print("Extracting candidate markers …")
    feat_result = extract_features(
        case_id=case_id,
        hu_image=roi_image,
        label_files=seg_result_label_files,
        cfg=cfg.features,
        roi_quality=roi_result.roi_quality,
        roi_method=roi_method,
        domain_warnings=domain_warnings,
    )
    all_warnings.extend(feat_result.warnings)

    features_path = out / "candidate_features.json"
    write_features_json(feat_result, features_path)
    report.candidate_features_path = str(features_path)

    # Segmentation QC runs *after* features so pathology fiducials can be
    # embedded into the Slicer scene.
    if seg_result_label_files:
        seg_qc = generate_segmentation_qc(
            image=roi_image,
            label_files=seg_result_label_files,
            out_dir=out / "qc",
            cfg=cfg.qc,
            segmenter_name=report.segmentation_backend or segmenter_backend,
            image_path=seg_image_path,
            features_path=features_path,
        )
        qc_paths.update({k: str(v) for k, v in seg_qc.items()})

    # ── Finalize Report ───────────────────────────────────────────────────────
    qc_summary = generate_qc_summary_json(
        out / "qc", {k: Path(v) for k, v in qc_paths.items()},
        all_warnings, roi_result.roi_quality, segmenter_backend,
    )
    qc_paths["qc_summary"] = str(qc_summary)

    report.qc_paths = qc_paths
    report.warnings = all_warnings
    report.errors = all_errors
    report.status = "complete" if not all_errors else "complete_with_errors"

    report_path = out / "report.json"
    write_report(report, report_path)

    slicer_scene = qc_paths.get("scene", qc_paths.get("roi_scene", ""))
    console.rule("[bold green]Pipeline complete[/bold green]")
    console.print(f"Output directory: [cyan]{out}[/cyan]")
    console.print(f"Report:           [cyan]{report_path}[/cyan]")
    console.print(f"Features:         [cyan]{features_path}[/cyan]")
    if slicer_scene:
        console.print(f"Open in Slicer:   [cyan]{slicer_scene}[/cyan]")
        open_slicer_scene(Path(slicer_scene))
    if all_warnings:
        console.print(f"[yellow]Warnings ({len(all_warnings)}):[/yellow]")
        for w in all_warnings[:5]:
            console.print(f"  {w}")
    console.print(f"\n[bold yellow]{DISCLAIMER}[/bold yellow]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ROI method -> TotalSegmentator task name (for --reuse-roi-seg matching).
_ROI_TASK = {"totalseg_teeth": "teeth", "totalseg_craniofacial": "craniofacial_structures"}


def _reuse_roi_segmentation(roi_seg_dir: Path, roi_image, seg, seg_out: Path):
    """Build a final segmentation by cropping the ROI-detection labels to the ROI.

    Avoids a second TotalSegmentator inference when the ROI method and segmenter
    use the same task. Each ROI label (full analysis space) is resampled to the
    ROI-crop geometry (nearest-neighbor) and written to ``seg_out``, mirroring the
    layout produced by ``seg.run`` so features/QC consume it unchanged.
    """
    import SimpleITK as sitk

    from .segmenters.base import SegmentationResult

    seg_out.mkdir(parents=True, exist_ok=True)
    label_files: dict[str, Path] = {}
    for src in sorted(roi_seg_dir.glob("*.nii.gz")):
        if src.name.startswith("_"):
            continue
        lbl = sitk.ReadImage(str(src))
        cropped = sitk.Resample(
            lbl, roi_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, lbl.GetPixelID()
        )
        dst = seg_out / src.name
        sitk.WriteImage(cropped, str(dst), useCompression=True)
        label_files[src.name.replace(".nii.gz", "")] = dst

    labels_json = seg._write_labels_json(seg_out, label_files)
    return SegmentationResult(
        success=True,
        label_map=None,
        label_files=label_files,
        labels_json=labels_json,
        meta={"reused_roi_segmentation": True, "n_labels": len(label_files)},
    )


def _build_segmenter(name: str, cfg: PipelineConfig):
    from .segmenters.dentalsegmentator import DentalSegmentatorSegmenter
    from .segmenters.oralseg import OralSegSegmenter
    from .segmenters.rail import RAILSegmenter
    from .segmenters.totalsegmentator import TotalSegmentatorTeethSegmenter

    if name in ("totalseg_teeth", "totalseg_craniofacial"):
        task = "teeth" if name == "totalseg_teeth" else "craniofacial_structures"
        return TotalSegmentatorTeethSegmenter(task=task)
    elif name == "dentalsegmentator":
        return DentalSegmentatorSegmenter(
            weights_path=cfg.segmentation.dentalsegmentator.weights_path,
            nnunet_results_dir=cfg.segmentation.dentalsegmentator.nnunet_results_dir,
        )
    elif name == "oralseg":
        return OralSegSegmenter(model_path=cfg.segmentation.oralseg.model_path)
    elif name == "rail":
        return RAILSegmenter(model_path=cfg.segmentation.rail.model_path)
    else:
        return get_segmenter(name)


if __name__ == "__main__":
    app()
