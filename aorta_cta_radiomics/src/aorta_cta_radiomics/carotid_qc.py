"""Headless, anatomy-honest QC for FLOWCAT-backed carotid processing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

from .flowcat_adapter import (  # noqa: E402
    DEFAULT_REQUIRED_OUTPUTS,
    FlowcatGeometryError,
    FlowcatSchemaError,
    ImageGeometry,
    assert_geometry_compatible,
    geometry_from_nifti,
    load_binary_nifti,
)


QC_JSON_NAME = "carotid_qc.json"
MONTAGE_NAME = "carotid_qc_montage.png"


class CarotidQCError(RuntimeError):
    """A QC input is missing, malformed, or geometrically incompatible."""


@dataclass(frozen=True, slots=True)
class _Overlay:
    name: str
    data: np.ndarray
    metrics: dict[str, Any]


def run_carotid_qc(
    *,
    case_id: str,
    cta_path: str | os.PathLike[str],
    segmentation_path: str | os.PathLike[str],
    flowcat_manifest_path: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    labelled_artery_path: str | os.PathLike[str] | None = None,
    shell_path: str | os.PathLike[str] | None = None,
    tissue_path: str | os.PathLike[str] | None = None,
    calcium_path: str | os.PathLike[str] | None = None,
    focus_overlay: str | None = None,
) -> dict[str, Any]:
    """Validate inputs, render a three-plane montage, and write QC JSON.

    Only array data and geometric fields needed for alignment are read from the
    NIfTI files. Descriptive header fields and DICOM metadata are not inspected.
    """

    if not isinstance(case_id, str) or not case_id.strip():
        raise CarotidQCError("An explicit non-empty case ID is required")
    output_directory = Path(outdir).expanduser()
    cta_file = _require_file(cta_path, "CTA")
    segmentation_file = _require_file(segmentation_path, "arterial segmentation")
    manifest_file = _require_file(flowcat_manifest_path, "FLOWCAT manifest")

    cta_image = _load_nifti(cta_file, "CTA")
    cta_geometry = _geometry(cta_image, "CTA")
    cta = _load_cta_array(cta_image)
    try:
        segmentation_volume = load_binary_nifti(
            segmentation_file,
            reference_geometry=cta_geometry,
        )
    except (FlowcatGeometryError, FlowcatSchemaError, OSError, ValueError) as exc:
        raise CarotidQCError(f"Invalid arterial segmentation: {exc}") from exc
    segmentation = segmentation_volume.data

    display_cta_image = nib.as_closest_canonical(cta_image)
    display_geometry = _geometry(display_cta_image, "canonical display CTA")
    if display_geometry.orientation != ("R", "A", "S"):
        raise CarotidQCError(
            "Could not reorient CTA to canonical RAS display orientation; "
            f"observed {display_geometry.orientation!r}"
        )
    display_cta = _load_cta_array(display_cta_image)
    display_segmentation, display_segmentation_geometry = _canonicalize_label_array(
        segmentation,
        cta_geometry,
        description="arterial segmentation",
    )
    try:
        assert_geometry_compatible(display_segmentation_geometry, display_geometry)
    except FlowcatGeometryError as exc:
        raise CarotidQCError(f"Canonical segmentation display geometry is incompatible: {exc}") from exc
    display_segmentation = np.asarray(display_segmentation != 0, dtype=bool)

    manifest = _load_manifest(manifest_file)
    completeness = _required_output_completeness(manifest, segmentation_validated=True)
    voxel_volume_mm3 = float(np.prod(cta_geometry.spacing_mm))
    segmentation_metrics = _mask_metrics(segmentation, voxel_volume_mm3, categorical=False)
    segmentation_metrics.update(
        {
            "binary": True,
            "geometry_matches_cta": True,
            "affine_matches_cta": True,
        }
    )

    overlay_paths = {
        "labelled_artery": labelled_artery_path,
        "shell": shell_path,
        "tissue": tissue_path,
        "calcium": calcium_path,
    }
    overlays: dict[str, _Overlay] = {}
    overlay_availability: dict[str, bool] = {}
    overlay_metrics: dict[str, dict[str, Any]] = {}
    for name, path in overlay_paths.items():
        if path is None:
            overlay_availability[name] = False
            overlay_metrics[name] = _missing_overlay_metrics()
            continue
        overlay = _load_overlay(name, path, cta_geometry, voxel_volume_mm3)
        overlays[name] = overlay
        overlay_availability[name] = True
        overlay_metrics[name] = overlay.metrics

    display_overlays: dict[str, _Overlay] = {}
    for name, overlay in overlays.items():
        display_data, overlay_display_geometry = _canonicalize_label_array(
            overlay.data,
            cta_geometry,
            description=f"{name} overlay",
        )
        try:
            assert_geometry_compatible(overlay_display_geometry, display_geometry)
        except FlowcatGeometryError as exc:
            raise CarotidQCError(f"Canonical {name} display geometry is incompatible: {exc}") from exc
        display_overlays[name] = _Overlay(name=name, data=display_data, metrics=overlay.metrics)

    supported_focus_overlays = {
        "segmentation",
        "labelled_artery",
        "shell",
        "tissue",
        "calcium",
    }
    if focus_overlay is not None and focus_overlay not in supported_focus_overlays:
        raise CarotidQCError(
            f"Unsupported focus overlay {focus_overlay!r}; expected one of "
            f"{sorted(supported_focus_overlays)}"
        )
    if focus_overlay == "segmentation":
        focus_data = display_segmentation
    elif focus_overlay is None:
        focus_data = None
    else:
        focused = display_overlays.get(focus_overlay)
        if focused is None:
            raise CarotidQCError(
                f"Requested focus overlay {focus_overlay!r} was not provided"
            )
        focus_data = focused.data
    selection, selection_record = _display_slice_selection(
        display_segmentation,
        display_geometry=display_geometry,
        original_geometry=cta_geometry,
        focus_ras=focus_data,
        focus_name=focus_overlay,
    )
    window = _display_window(display_cta)
    output_directory.mkdir(parents=True, exist_ok=True)
    montage_path = output_directory / MONTAGE_NAME
    qc_json_path = output_directory / QC_JSON_NAME
    _write_montage(
        cta=display_cta,
        segmentation=display_segmentation,
        overlays=display_overlays,
        selection=selection,
        window=window,
        spacing_mm=display_geometry.spacing_mm,
        case_id=case_id,
        output_path=montage_path,
        selection_record=selection_record,
    )

    named_overlay_available = overlay_availability["labelled_artery"]
    named_results = {
        "available": False,
        "status": (
            "label_map_available_but_named_measurements_not_generated"
            if named_overlay_available
            else "not_generated_labelled_artery_overlay_unavailable"
        ),
        "vessel_results": [],
        "label_ids_available_for_visual_qc": overlay_metrics["labelled_artery"]["nonzero_label_ids"],
        "note": (
            "The labelled overlay is shown for QC only; this module does not infer artery names or burdens."
            if named_overlay_available
            else "Binary whole-tree segmentation does not provide named artery results."
        ),
    }

    manual_review_reasons = ["visual_review_pending"]
    if not completeness["complete"]:
        manual_review_reasons.append("required_flowcat_outputs_incomplete")
    if not named_overlay_available:
        manual_review_reasons.append("named_artery_overlay_unavailable")
    if segmentation_metrics["nonzero_voxels"] == 0:
        manual_review_reasons.append("arterial_segmentation_empty")
    empty_overlays = sorted(
        name
        for name, metrics in overlay_metrics.items()
        if metrics["provided"] and not metrics["nonempty"]
    )
    if empty_overlays:
        manual_review_reasons.append("one_or_more_provided_overlays_empty")

    qc_flags: list[dict[str, str]] = []
    if not completeness["complete"]:
        qc_flags.append(
            {
                "code": "FLOWCAT_REQUIRED_OUTPUTS_INCOMPLETE",
                "severity": "warning",
                "message": "One or more required native FLOWCAT outputs are absent or invalid.",
            }
        )
    if not named_overlay_available:
        qc_flags.append(
            {
                "code": "NAMED_ARTERY_RESULTS_UNAVAILABLE",
                "severity": "info",
                "message": (
                    "Only whole-tree binary segmentation is available; "
                    "named artery results were not inferred."
                ),
            }
        )
    if segmentation_metrics["nonzero_voxels"] == 0:
        qc_flags.append(
            {
                "code": "ARTERIAL_SEGMENTATION_EMPTY",
                "severity": "error",
                "message": "The supplied binary arterial segmentation contains no foreground voxels.",
            }
        )

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "case_id": case_id,
        "processing_status": "qc_generated_pending_visual_review",
        "source_data_policy": {
            "descriptive_nifti_metadata_inspected": False,
            "dicom_metadata_inspected": False,
            "geometry_fields_used": ["shape", "affine", "voxel_spacing", "orientation"],
        },
        "required_output_completeness": completeness,
        "geometry": {
            "reference": "original_nifti_geometry",
            "shape_ijk": list(cta_geometry.shape_ijk),
            "spacing_mm": list(cta_geometry.spacing_mm),
            "orientation": list(cta_geometry.orientation),
            "voxel_volume_mm3": voxel_volume_mm3,
            "cta_segmentation_shape_match": True,
            "cta_segmentation_affine_match": True,
            "all_provided_overlays_geometry_match": True,
        },
        "display": {
            "coordinate_convention": "canonical_RAS_plus",
            "reorientation": "axis_permutation_and_flip_only_no_resampling",
            "orientation": list(display_geometry.orientation),
            "shape_ijk": list(display_geometry.shape_ijk),
            "spacing_mm": list(display_geometry.spacing_mm),
            "plane_direction_labels": {
                "axial": {"left": "L", "right": "R", "top": "A", "bottom": "P"},
                "coronal": {"left": "L", "right": "R", "top": "S", "bottom": "I"},
                "sagittal": {"left": "P", "right": "A", "top": "S", "bottom": "I"},
            },
            "selection": selection_record,
        },
        "cta_display": {
            "finite_voxel_fraction": float(np.isfinite(display_cta).mean()),
            "window_lower": window[0],
            "window_upper": window[1],
            "selected_canonical_voxel_ijk": list(selection),
            "selection_details": "see display.selection",
        },
        "segmentation_metrics": segmentation_metrics,
        "overlay_availability": overlay_availability,
        "overlay_metrics": overlay_metrics,
        "named_artery_results": named_results,
        "visual_review": {
            "status": "pending",
            "pending": True,
            "reviewed_by": None,
            "reviewed_at": None,
        },
        "manual_review": {
            "required": True,
            "reasons": sorted(set(manual_review_reasons)),
        },
        "qc_flags": qc_flags,
        "outputs": {
            "montage_png": MONTAGE_NAME,
            "qc_json": QC_JSON_NAME,
        },
    }
    _write_json_atomic(report, qc_json_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cta", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--flowcat-manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--labelled-artery",
        "--labelled-artery-nifti",
        "--labelled-artery-mask",
        dest="labelled_artery",
    )
    parser.add_argument("--shell", "--shell-nifti", "--shell-mask", dest="shell")
    parser.add_argument("--tissue", "--tissue-nifti", "--tissue-mask", dest="tissue")
    parser.add_argument("--calcium", "--calcium-nifti", "--calcium-mask", dest="calcium")
    parser.add_argument(
        "--focus-overlay",
        choices=("segmentation", "labelled_artery", "shell", "tissue", "calcium"),
        help=(
            "Center all three planes on a supplied overlay foreground voxel. "
            "Omit for the anatomy-neutral superior whole-tree overview."
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = run_carotid_qc(
            case_id=args.case_id,
            cta_path=args.cta,
            segmentation_path=args.segmentation,
            flowcat_manifest_path=args.flowcat_manifest,
            outdir=args.outdir,
            labelled_artery_path=args.labelled_artery,
            shell_path=args.shell,
            tissue_path=args.tissue,
            calcium_path=args.calcium,
            focus_overlay=args.focus_overlay,
        )
    except (CarotidQCError, OSError, ValueError) as exc:
        json.dump(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            fp=sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 2
    json.dump(
        {
            "status": report["processing_status"],
            "manual_review_required": report["manual_review"]["required"],
            "qc_json": str(Path(args.outdir).expanduser() / QC_JSON_NAME),
            "montage_png": str(Path(args.outdir).expanduser() / MONTAGE_NAME),
        },
        fp=sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


def _require_file(path: str | os.PathLike[str], description: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise CarotidQCError(f"{description} file not found: {file_path}")
    return file_path


def _load_nifti(path: Path, description: str) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise CarotidQCError(f"Could not read {description} NIfTI: {exc}") from exc
    if len(image.shape) != 3:
        raise CarotidQCError(f"{description} NIfTI must be 3D, got shape {image.shape!r}")
    return image


def _geometry(image: nib.spatialimages.SpatialImage, description: str) -> ImageGeometry:
    try:
        return geometry_from_nifti(image)
    except (FlowcatGeometryError, TypeError, ValueError) as exc:
        raise CarotidQCError(f"Invalid {description} geometry: {exc}") from exc


def _load_cta_array(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    try:
        cta = np.asarray(image.dataobj, dtype=np.float32)
    except Exception as exc:
        raise CarotidQCError(f"Could not read CTA voxel data: {exc}") from exc
    if not np.isfinite(cta).any():
        raise CarotidQCError("CTA contains no finite voxel intensities")
    return cta


def _canonicalize_label_array(
    data: np.ndarray,
    original_geometry: ImageGeometry,
    *,
    description: str,
) -> tuple[np.ndarray, ImageGeometry]:
    """Reorder/flip a label array to RAS without interpolating or resampling."""

    storage = np.asarray(data, dtype=np.uint8) if data.dtype == np.bool_ else np.asarray(data)
    try:
        image = nib.Nifti1Image(storage, np.asarray(original_geometry.affine_ras_mm, dtype=float))
        canonical = nib.as_closest_canonical(image)
        canonical_data = np.asarray(canonical.dataobj)
        canonical_geometry = geometry_from_nifti(canonical)
    except Exception as exc:
        raise CarotidQCError(f"Could not create canonical RAS display for {description}: {exc}") from exc
    if canonical_geometry.orientation != ("R", "A", "S"):
        raise CarotidQCError(
            f"Canonical display for {description} has orientation {canonical_geometry.orientation!r}, not RAS"
        )
    return canonical_data, canonical_geometry


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CarotidQCError(f"Could not read FLOWCAT manifest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CarotidQCError("FLOWCAT manifest must contain a JSON object")
    return payload


def _required_output_completeness(
    manifest: Mapping[str, Any],
    *,
    segmentation_validated: bool,
) -> dict[str, Any]:
    required = set(DEFAULT_REQUIRED_OUTPUTS)
    present: set[str] = {"segmentation"} if segmentation_validated else set()
    invalid: set[str] = set()
    explicit_missing: set[str] = set()

    artifacts = manifest.get("artifacts", [])
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("name"), str):
                continue
            name = str(artifact["name"])
            if artifact.get("required") is True:
                required.add(name)
            if artifact.get("present") is True:
                present.add(name)
            if artifact.get("required") is True and artifact.get("validation_status") == "invalid":
                invalid.add(name)
    raw_missing = manifest.get("missing_required_outputs", [])
    if isinstance(raw_missing, list):
        explicit_missing.update(str(name) for name in raw_missing if isinstance(name, str))

    # Also accept the read-only audit report shape without importing any paths.
    flowcat = manifest.get("flowcat")
    if isinstance(flowcat, Mapping):
        raw_required = flowcat.get("required_native_outputs", [])
        if isinstance(raw_required, list):
            required.update(str(name) for name in raw_required if isinstance(name, str))
        raw_missing = flowcat.get("missing_required_native_outputs", [])
        if isinstance(raw_missing, list):
            explicit_missing.update(str(name) for name in raw_missing if isinstance(name, str))
        raw_invalid = flowcat.get("invalid_required_native_outputs", [])
        if isinstance(raw_invalid, list):
            invalid.update(str(name) for name in raw_invalid if isinstance(name, str))
        raw_outputs = flowcat.get("native_outputs", [])
        if isinstance(raw_outputs, list):
            for artifact in raw_outputs:
                if isinstance(artifact, Mapping) and artifact.get("present") is True:
                    name = artifact.get("name")
                    if isinstance(name, str):
                        present.add(name)

    missing = (required - present) | explicit_missing
    missing -= {"segmentation"} if segmentation_validated else set()
    complete = not missing and not invalid
    return {
        "required_outputs": sorted(required),
        "present_required_outputs": sorted(required & present),
        "missing_required_outputs": sorted(missing),
        "invalid_required_outputs": sorted(invalid),
        "complete": complete,
        "status": "complete" if complete else "incomplete",
    }


def _load_overlay(
    name: str,
    path: str | os.PathLike[str],
    reference_geometry: ImageGeometry,
    voxel_volume_mm3: float,
) -> _Overlay:
    file_path = _require_file(path, f"{name} overlay")
    image = _load_nifti(file_path, f"{name} overlay")
    geometry = _geometry(image, f"{name} overlay")
    try:
        assert_geometry_compatible(geometry, reference_geometry)
    except FlowcatGeometryError as exc:
        raise CarotidQCError(f"{name} overlay geometry does not match CTA: {exc}") from exc
    try:
        raw = np.asarray(image.dataobj)
    except Exception as exc:
        raise CarotidQCError(f"Could not read {name} overlay voxel data: {exc}") from exc
    if not np.issubdtype(raw.dtype, np.number) and raw.dtype != np.bool_:
        raise CarotidQCError(f"{name} overlay must be numeric")
    if not np.isfinite(raw).all():
        raise CarotidQCError(f"{name} overlay contains NaN or infinite values")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1e-6) or bool((rounded < 0).any()):
        raise CarotidQCError(f"{name} overlay must contain non-negative integer labels")
    data = rounded.astype(np.int32, copy=False)
    labels = np.unique(data)
    if len(labels) > 256:
        raise CarotidQCError(f"{name} overlay contains too many distinct labels ({len(labels)})")
    metrics = _mask_metrics(data, voxel_volume_mm3, categorical=True)
    metrics.update(
        {
            "provided": True,
            "available": True,
            "geometry_matches_cta": True,
            "affine_matches_cta": True,
        }
    )
    return _Overlay(name=name, data=data, metrics=metrics)


def _missing_overlay_metrics() -> dict[str, Any]:
    return {
        "provided": False,
        "available": False,
        "nonempty": False,
        "nonzero_voxels": 0,
        "volume_ml": 0.0,
        "nonzero_label_ids": [],
        "label_voxel_counts": {},
        "geometry_matches_cta": None,
        "affine_matches_cta": None,
    }


def _mask_metrics(data: np.ndarray, voxel_volume_mm3: float, *, categorical: bool) -> dict[str, Any]:
    foreground = np.asarray(data != 0, dtype=bool)
    nonzero_voxels = int(foreground.sum())
    _, component_count = ndimage.label(foreground, structure=np.ones((3, 3, 3), dtype=np.uint8))
    bbox = None
    if nonzero_voxels:
        indices = np.argwhere(foreground)
        bbox = {
            "min_ijk": [int(value) for value in indices.min(axis=0)],
            "max_ijk": [int(value) for value in indices.max(axis=0)],
        }
    nonzero_labels = [int(value) for value in np.unique(data) if int(value) != 0]
    label_counts = (
        {str(label): int(np.count_nonzero(data == label)) for label in nonzero_labels}
        if categorical
        else {}
    )
    return {
        "provided": True,
        "available": True,
        "nonempty": bool(nonzero_voxels),
        "nonzero_voxels": nonzero_voxels,
        "foreground_fraction": float(nonzero_voxels / data.size),
        "volume_ml": float(nonzero_voxels * voxel_volume_mm3 / 1000.0),
        "connected_component_count_26": int(component_count),
        "bounding_box": bbox,
        "nonzero_label_ids": nonzero_labels,
        "label_voxel_counts": label_counts,
    }


def _display_slice_selection(
    segmentation_ras: np.ndarray,
    *,
    display_geometry: ImageGeometry,
    original_geometry: ImageGeometry,
    superior_percentile: float = 0.85,
    focus_ras: np.ndarray | None = None,
    focus_name: str | None = None,
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    """Choose an anatomy-neutral overview or an explicitly supplied overlay focus."""

    focus = None if focus_ras is None else np.asarray(focus_ras != 0, dtype=bool)
    if focus is not None and focus.shape != segmentation_ras.shape:
        raise CarotidQCError("Focus overlay does not share the canonical display shape")
    focused_coordinates = np.argwhere(focus) if focus is not None else np.empty((0, 3), int)
    if focus is not None and len(focused_coordinates):
        spacing = np.asarray(display_geometry.spacing_mm, dtype=float)
        median = np.median(focused_coordinates, axis=0)
        physical_squared_distance = np.square(
            (focused_coordinates - median) * spacing[None, :]
        ).sum(axis=1)
        selected = focused_coordinates[int(np.argmin(physical_squared_distance))]
        selection = tuple(int(value) for value in selected)
        occupied_k = np.flatnonzero(focus.any(axis=(0, 1)))
        method = "provided_overlay_foreground_voxel_nearest_physical_median"
        percentile_value = None
        percentile_basis = None
        purpose = f"provided {focus_name} overlay focus for visual QC"
        anatomical_claim = "focus_from_provided_overlay_not_inferred_anatomy"
        in_plane_selection = "foreground_voxel_nearest_3d_physical_median"
        selection_mask = focus
        display_context = (
            f"provided {focus_name} overlay focus; location supplied by caller, not anatomically inferred"
        )
        axial_title = f"Axial — {focus_name} overlay focus"
    else:
        occupied_k = np.flatnonzero(segmentation_ras.any(axis=(0, 1)))
        selection_mask = segmentation_ras
        purpose = "superior whole-tree arterial context for visual QC"
        anatomical_claim = "not_a_named_carotid_localization"
        in_plane_selection = (
            "foreground_voxel_nearest_slice_median" if len(occupied_k) else "image_midpoint"
        )
        display_context = "superior whole-tree context (not a named-carotid localization)"
        axial_title = "Axial — superior arterial context"
    if focus is None or not len(focused_coordinates):
        if focus is not None and not len(focused_coordinates):
            purpose = f"empty provided {focus_name} overlay; fallback to whole-tree context"
            anatomical_claim = "empty_focus_overlay_fallback_not_anatomical_localization"
            display_context = (
                f"empty provided {focus_name} overlay; superior whole-tree fallback"
            )
        if len(occupied_k) == 0:
            selection = tuple(int(size // 2) for size in segmentation_ras.shape)
            method = "image_midpoint_empty_segmentation"
            percentile_value = None
            percentile_basis = None
        else:
            # An unweighted percentile of occupied slices prevents a large thoracic
            # aorta cross-section from dominating a foreground-voxel quantile.
            rank = int(round((len(occupied_k) - 1) * superior_percentile))
            k = int(occupied_k[rank])
            in_plane = np.argwhere(segmentation_ras[:, :, k])
            median_ij = np.median(in_plane, axis=0)
            squared_distance = np.square(in_plane - median_ij).sum(axis=1)
            selected_ij = in_plane[int(np.argmin(squared_distance))]
            selection = (int(selected_ij[0]), int(selected_ij[1]), k)
            method = "superior_arterial_context_occupied_slice_percentile"
            percentile_value = superior_percentile
            percentile_basis = "unweighted_axial_slices_with_whole_tree_foreground"

    if len(occupied_k) == 0:
        selection = tuple(int(size // 2) for size in segmentation_ras.shape)

    display_affine = np.asarray(display_geometry.affine_ras_mm, dtype=float)
    original_affine = np.asarray(original_geometry.affine_ras_mm, dtype=float)
    physical_ras = (display_affine @ np.asarray((*selection, 1.0), dtype=float))[:3]
    original_voxel = (np.linalg.inv(original_affine) @ np.asarray((*physical_ras, 1.0)))[:3]
    original_nearest = np.rint(original_voxel).astype(int)
    original_nearest = np.clip(original_nearest, 0, np.asarray(original_geometry.shape_ijk) - 1)
    selected_k = selection[2]
    record = {
        "purpose": purpose,
        "anatomical_claim": anatomical_claim,
        "method": method,
        "focus_overlay": focus_name,
        "superior_percentile": percentile_value,
        "percentile_basis": percentile_basis,
        "superior_direction": (
            "increasing canonical RAS k (S)" if focus is None else None
        ),
        "occupied_axial_slice_count": int(len(occupied_k)),
        "selected_axial_foreground_voxels": int(selection_mask[:, :, selected_k].sum()),
        "canonical_voxel_ijk": list(selection),
        "original_voxel_ijk_nearest": [int(value) for value in original_nearest],
        "physical_ras_mm": [float(value) for value in physical_ras],
        "in_plane_selection": in_plane_selection,
        "display_context": display_context,
        "axial_title": axial_title,
    }
    return selection, record


def _display_window(cta: np.ndarray) -> tuple[float, float]:
    finite = cta[np.isfinite(cta)]
    lower, upper = (float(value) for value in np.percentile(finite, [1.0, 99.0]))
    if upper <= lower:
        lower = float(finite.min())
        upper = float(finite.max())
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _write_montage(
    *,
    cta: np.ndarray,
    segmentation: np.ndarray,
    overlays: Mapping[str, _Overlay],
    selection: tuple[int, int, int],
    window: tuple[float, float],
    spacing_mm: tuple[float, float, float],
    case_id: str,
    output_path: Path,
    selection_record: Mapping[str, Any],
) -> None:
    planes = (
        (
            str(selection_record["axial_title"]),
            2,
            selection[2],
            spacing_mm[1] / spacing_mm[0],
            "L",
            "R",
            "A",
            "P",
        ),
        ("Coronal", 1, selection[1], spacing_mm[2] / spacing_mm[0], "L", "R", "S", "I"),
        ("Sagittal", 0, selection[0], spacing_mm[2] / spacing_mm[1], "P", "A", "S", "I"),
    )
    overlay_styles = {
        "labelled_artery": ("tab20", 0.40, "Labelled artery"),
        "shell": ("Blues", 0.35, "Radial shell"),
        "tissue": ("Greens", 0.32, "Tissue class"),
        "calcium": ("autumn", 0.65, "Calcium"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 6))
    for axis_plot, plane in zip(axes, planes):
        title, axis_index, slice_index, aspect, left, right, top, bottom = plane
        base = _plane(cta, axis_index, slice_index)
        arterial = _plane(segmentation, axis_index, slice_index)
        axis_plot.imshow(base, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
        if arterial.any():
            axis_plot.contour(arterial.astype(float), levels=[0.5], colors=["#ff3030"], linewidths=0.8)
        for name, overlay in overlays.items():
            layer = _plane(overlay.data, axis_index, slice_index)
            masked = np.ma.masked_where(layer == 0, layer)
            cmap, alpha, _ = overlay_styles[name]
            axis_plot.imshow(masked, cmap=cmap, alpha=alpha, interpolation="nearest")
        axis_plot.set_title(f"{title} (index {slice_index})")
        axis_plot.set_aspect(aspect)
        _add_plane_direction_labels(axis_plot, left=left, right=right, top=top, bottom=bottom)
        axis_plot.axis("off")
    handles = [Patch(facecolor="none", edgecolor="#ff3030", label="Whole-tree artery")]
    for name in overlays:
        cmap_name, _, label = overlay_styles[name]
        handles.append(Patch(facecolor=plt.get_cmap(cmap_name)(0.75), label=label))
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.14, top=0.80, wspace=0.12)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(handles)),
        frameon=False,
    )
    figure.suptitle(
        f"Carotid pipeline QC — {case_id} — visual review pending\n"
        f"Canonical RAS display; {selection_record['display_context']}"
    )
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)


def _plane(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def _add_plane_direction_labels(
    axis_plot: Any,
    *,
    left: str,
    right: str,
    top: str,
    bottom: str,
) -> None:
    style = {
        "color": "white",
        "fontsize": 9,
        "fontweight": "bold",
        "bbox": {"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5},
        "transform": axis_plot.transAxes,
        "clip_on": False,
    }
    axis_plot.text(0.01, 0.50, left, ha="left", va="center", **style)
    axis_plot.text(0.99, 0.50, right, ha="right", va="center", **style)
    axis_plot.text(0.50, 0.99, top, ha="center", va="top", **style)
    axis_plot.text(0.50, 0.01, bottom, ha="center", va="bottom", **style)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
