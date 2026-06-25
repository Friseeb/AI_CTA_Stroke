"""ROI detection strategies for dentition/dentoalveolar region.

Strategies:
  totalseg_teeth             — TotalSegmentator task=teeth (preferred)
  totalseg_craniofacial      — TotalSegmentator task=craniofacial_structures
  dentalsegmentator_coarse   — DentalSegmentator coarse pass for ROI only
  threshold_fallback         — HU threshold fallback (degraded mode, poor quality)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .config import ROIConfig
from .external_tools import find_totalsegmentator, run_totalsegmentator
from .geometry import BoundingBox, crop_array
from .logging_utils import get_logger

log = get_logger("roi")

# Label stems to include when building dental ROI from TotalSegmentator teeth task.
_TEETH_ROI_STEMS = {
    "teeth_upper", "teeth_lower",
    "jawbone_upper", "jawbone_lower",
    "tooth_canal", "implant", "crown", "bridge",
    "mandibular_canal", "mandible",
    # fallback partial matches
    "teeth", "tooth", "jaw", "dental",
}

_CRANIOFACIAL_ROI_STEMS = {
    "mandible", "teeth_upper", "teeth_lower",
    "teeth", "tooth", "jaw",
    "maxilla", "skull_base",
}

# Substrings identifying individual tooth labels (for the smear sanity check).
_TOOTH_LABEL_KEYS = ("molar", "incisor", "canine", "premolar", "fdi")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component, dropping scattered noise specks.

    TotalSegmentator-teeth on out-of-domain contrast CTA emits tooth labels as a
    real tooth blob plus tiny false-positive specks far away, which inflate the
    label's bounding box (a premolar "spanning" 80+ mm is 3 specks, the biggest
    ~2 mm). Reducing each label to its largest component recovers the true extent.
    """
    from scipy import ndimage as ndi

    if not mask.any():
        return mask
    lbl, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == int(np.argmax(sizes))


def _max_tooth_extent_mm(label_files, spacing_xyz) -> tuple[float, Optional[str]]:
    """Largest single-tooth bounding-box extent (mm) across individual tooth labels.

    A real tooth is ~10-30 mm. Each label is first reduced to its largest connected
    component so scattered false-positive specks don't inflate the extent; a still-
    large extent then means a genuinely smeared label (no real dentition).
    """
    sx, sy, sz = (float(s) for s in spacing_xyz)  # ITK order x, y, z
    worst, worst_name = 0.0, None
    for f in label_files:
        if not any(k in f.stem.lower() for k in _TOOTH_LABEL_KEYS):
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0  # array order z, y, x
        arr = _largest_component(arr)
        nz = np.argwhere(arr)
        if nz.size == 0:
            continue
        ext = max(
            (nz[:, 0].max() - nz[:, 0].min() + 1) * sz,
            (nz[:, 1].max() - nz[:, 1].min() + 1) * sy,
            (nz[:, 2].max() - nz[:, 2].min() + 1) * sx,
        )
        if ext > worst:
            worst, worst_name = ext, f.stem
    return worst, worst_name


@dataclass
class ROIResult:
    success: bool
    roi_image: Optional[sitk.Image]        # cropped CTA in ROI
    roi_mask: Optional[sitk.Image]         # binary ROI mask (original space)
    bbox_voxel: Optional[dict]
    bbox_physical: Optional[dict]
    fov_completeness: dict = field(default_factory=dict)
    roi_quality: str = "unknown"           # good | fair | poor | failed
    method_used: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "roi_quality": self.roi_quality,
            "method_used": self.method_used,
            "bbox_voxel": self.bbox_voxel,
            "bbox_physical": self.bbox_physical,
            "fov_completeness": self.fov_completeness,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def detect_roi(
    image: sitk.Image,
    cfg: ROIConfig,
    out_dir: Path,
    seg_cfg: Optional[dict] = None,
) -> ROIResult:
    method = cfg.method
    log.info("ROI detection method: %s", method)

    if method == "totalseg_teeth":
        return _roi_totalseg(image, cfg, out_dir, task="teeth", seg_cfg=seg_cfg)
    elif method == "totalseg_craniofacial":
        return _roi_totalseg(image, cfg, out_dir, task="craniofacial_structures", seg_cfg=seg_cfg)
    elif method == "dentalsegmentator_coarse":
        return _roi_dentalseg_coarse(image, cfg, out_dir, seg_cfg=seg_cfg)
    elif method == "threshold_fallback":
        return _roi_threshold_fallback(image, cfg, out_dir)
    else:
        raise ValueError(f"Unknown ROI method: {method}")


def _roi_totalseg(
    image: sitk.Image,
    cfg: ROIConfig,
    out_dir: Path,
    task: str,
    seg_cfg: Optional[dict],
) -> ROIResult:
    if find_totalsegmentator() is None:
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used=f"totalseg_{task}",
            errors=[
                "TotalSegmentator not found. Install it with: pip install TotalSegmentator\n"
                "or choose --roi-method threshold_fallback for degraded ROI only."
            ],
        )

    seg_out = out_dir / f"_tseg_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "_roi_input.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    ts_cfg = (seg_cfg or {}).get("totalsegmentator", {})
    try:
        run_totalsegmentator(
            input_nifti=input_path,
            output_dir=seg_out,
            task=task,
            fast=ts_cfg.get("fast", False),
            device=ts_cfg.get("device", "cpu"),
            weights_dir=ts_cfg.get("weights_dir"),
        )
    except RuntimeError as exc:
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used=f"totalseg_{task}",
            errors=[str(exc)],
        )

    stems = _TEETH_ROI_STEMS if task == "teeth" else _CRANIOFACIAL_ROI_STEMS
    roi_stems = stems
    label_files = list(seg_out.glob("*.nii.gz"))
    selected = [
        f for f in label_files
        if any(s in f.stem.lower() for s in roi_stems)
    ]
    if not selected:
        log.warning("No dental/jaw labels found by TotalSegmentator %s; using all labels.", task)
        selected = label_files

    # Sanity gate: reject smeared teeth segmentations (a single tooth spanning the
    # whole scan) so we fail honestly instead of yielding a whole-volume ROI.
    if task == "teeth" and cfg.max_tooth_extent_mm and cfg.max_tooth_extent_mm > 0:
        max_ext, name = _max_tooth_extent_mm(label_files, image.GetSpacing())
        if max_ext > cfg.max_tooth_extent_mm:
            msg = (f"Teeth segmentation implausible: label '{name}' spans {max_ext:.0f} mm "
                   f"(> {cfg.max_tooth_extent_mm:.0f} mm max). TotalSegmentator-teeth likely "
                   f"failed on this CTA (out-of-domain); dentition ROI not trustworthy.")
            log.warning(msg)
            return ROIResult(
                success=False, roi_image=None, roi_mask=None,
                bbox_voxel=None, bbox_physical=None,
                roi_quality="failed", method_used=f"totalseg_{task}", errors=[msg],
            )

    return _build_roi_from_label_files(
        image=image,
        label_files=selected,
        cfg=cfg,
        out_dir=out_dir,
        method=f"totalseg_{task}",
        all_label_files=label_files,
    )


def _roi_dentalseg_coarse(
    image: sitk.Image,
    cfg: ROIConfig,
    out_dir: Path,
    seg_cfg: Optional[dict],
) -> ROIResult:
    from .segmenters.dentalsegmentator import DentalSegmentatorSegmenter, LABELS

    ds_cfg = (seg_cfg or {}).get("dentalsegmentator", {})
    seg = DentalSegmentatorSegmenter(
        weights_path=ds_cfg.get("weights_path"),
        nnunet_results_dir=ds_cfg.get("nnunet_results_dir"),
    )
    if not seg.check_available():
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used="dentalsegmentator_coarse",
            errors=["DentalSegmentator not available. Configure weights or choose another ROI method."],
        )

    seg_out = out_dir / "_dentalseg_coarse"
    input_path = out_dir / "_roi_input.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = seg.run(input_path, seg_out, ds_cfg)
    if not result.success:
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used="dentalsegmentator_coarse",
            errors=result.errors,
        )

    selected = list(result.label_files.values())
    return _build_roi_from_label_files(
        image=image,
        label_files=selected,
        cfg=cfg,
        out_dir=out_dir,
        method="dentalsegmentator_coarse",
        all_label_files=selected,
    )


def _roi_threshold_fallback(
    image: sitk.Image,
    cfg: ROIConfig,
    out_dir: Path,
) -> ROIResult:
    log.warning(
        "Using threshold_fallback ROI (roi_quality=poor). "
        "This mode is for last-resort use only. "
        "Disease feature extraction is disabled unless explicitly allowed."
    )
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_ijk = tuple(reversed(spacing_xyz))

    binary = arr > cfg.threshold_fallback_hu

    from scipy import ndimage
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used="threshold_fallback",
            errors=["No high-density structures found above HU threshold."],
        )

    # Take the inferior 40% of the volume (face/jaw region)
    slice_cutoff = int(arr.shape[0] * 0.40)
    inferior_mask = np.zeros_like(binary)
    inferior_mask[:slice_cutoff] = binary[:slice_cutoff]

    if not inferior_mask.any():
        inferior_mask = binary  # fallback to full volume

    try:
        bbox = BoundingBox.from_mask(inferior_mask)
        bbox_expanded = bbox.expand_mm(cfg.margin_mm, spacing_ijk).clip_to_shape(arr.shape)
    except ValueError as exc:
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used="threshold_fallback",
            errors=[str(exc)],
        )

    roi_arr = crop_array(arr, bbox_expanded)
    roi_img = sitk.GetImageFromArray(roi_arr)
    roi_img.SetSpacing(image.GetSpacing())
    roi_img.SetDirection(image.GetDirection())

    mask_img = sitk.GetImageFromArray(inferior_mask.astype(np.uint8))
    mask_img.CopyInformation(image)

    _save_roi_outputs(roi_img, mask_img, bbox_expanded, image, out_dir)

    return ROIResult(
        success=True,
        roi_image=roi_img,
        roi_mask=mask_img,
        bbox_voxel=bbox_expanded.to_dict(),
        bbox_physical=bbox_expanded.to_physical(image.GetOrigin(), spacing_ijk),
        fov_completeness=_estimate_fov(None),
        roi_quality="poor",
        method_used="threshold_fallback",
        warnings=[
            "Threshold fallback ROI has poor quality. "
            "Disease feature extraction is disabled unless --allow-threshold-features is set.",
            f"HU threshold used: {cfg.threshold_fallback_hu}",
        ],
    )


def _build_roi_from_label_files(
    image: sitk.Image,
    label_files: list[Path],
    cfg: ROIConfig,
    out_dir: Path,
    method: str,
    all_label_files: list[Path],
) -> ROIResult:
    spacing_xyz = image.GetSpacing()
    spacing_ijk = tuple(reversed(spacing_xyz))

    combined_mask = None
    for lf in label_files:
        try:
            lmask = sitk.GetArrayFromImage(sitk.ReadImage(str(lf))).astype(bool)
            # Drop scattered false-positive specks so the ROI bbox is built from
            # the real teeth/bone, not noise far away (which would inflate it).
            lmask = _largest_component(lmask)
            combined_mask = lmask if combined_mask is None else (combined_mask | lmask)
        except Exception as exc:
            log.warning("Could not read label file %s: %s", lf, exc)

    if combined_mask is None or not combined_mask.any():
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used=method,
            warnings=["status: fov_missing_dentition — no dental/jaw labels detected."],
            errors=["No dental or jaw labels found. FOV may not include dentition."],
        )

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    try:
        bbox = BoundingBox.from_mask(combined_mask)
        bbox_expanded = bbox.expand_mm(cfg.margin_mm, spacing_ijk).clip_to_shape(arr.shape)
    except ValueError as exc:
        return ROIResult(
            success=False,
            roi_image=None,
            roi_mask=None,
            bbox_voxel=None,
            bbox_physical=None,
            roi_quality="failed",
            method_used=method,
            errors=[str(exc)],
        )

    roi_arr = crop_array(arr, bbox_expanded)
    roi_img = sitk.GetImageFromArray(roi_arr)
    roi_img.SetSpacing(image.GetSpacing())
    roi_img.SetDirection(image.GetDirection())
    # Origin = physical position of voxel (min_x, min_y, min_z) in the source
    # image. Must go through TransformIndexToPhysicalPoint so the direction
    # matrix is applied — naive `origin + min_ijk * spacing` is wrong whenever
    # the direction is not the identity (e.g. LPS NIfTI with -1,-1,+1).
    min_xyz = [int(v) for v in bbox_expanded.min_ijk[::-1]]
    roi_img.SetOrigin(image.TransformIndexToPhysicalPoint(min_xyz))

    mask_arr = combined_mask.astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(image)

    fov = _estimate_fov_from_labels(all_label_files)
    _save_roi_outputs(roi_img, mask_img, bbox_expanded, image, out_dir)

    return ROIResult(
        success=True,
        roi_image=roi_img,
        roi_mask=mask_img,
        bbox_voxel=bbox_expanded.to_dict(),
        bbox_physical=bbox_expanded.to_physical(image.GetOrigin(), spacing_ijk),
        fov_completeness=fov,
        roi_quality="good",
        method_used=method,
    )


def _estimate_fov_from_labels(label_files: list[Path]) -> dict:
    names = [f.stem.lower() for f in label_files]
    has_upper = any("upper" in n or "maxilla" in n for n in names)
    has_lower = any("lower" in n or "mandible" in n for n in names)
    has_mandible = any("mandible" in n for n in names)
    has_maxilla = any("maxilla" in n or "upper_skull" in n for n in names)
    return {
        "has_upper_dentition": has_upper,
        "has_lower_dentition": has_lower,
        "has_mandible": has_mandible,
        "has_maxilla": has_maxilla,
        "partial_fov": not (has_upper and has_lower),
        "left_right_coverage_estimate": "unknown",
        "inferior_superior_coverage_estimate": "unknown",
    }


def _estimate_fov(label_files) -> dict:
    return {
        "has_upper_dentition": "unknown",
        "has_lower_dentition": "unknown",
        "has_mandible": "unknown",
        "has_maxilla": "unknown",
        "partial_fov": "unknown",
        "left_right_coverage_estimate": "unknown",
        "inferior_superior_coverage_estimate": "unknown",
    }


def _save_roi_outputs(
    roi_img: sitk.Image,
    mask_img: sitk.Image,
    bbox: BoundingBox,
    orig_image: sitk.Image,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(roi_img, str(out_dir / "dentition_roi.nii.gz"), useCompression=True)
    sitk.WriteImage(mask_img, str(out_dir / "roi_mask.nii.gz"), useCompression=True)
    spacing_xyz = orig_image.GetSpacing()
    spacing_ijk = tuple(reversed(spacing_xyz))
    bbox_data = {
        **bbox.to_dict(),
        **bbox.to_physical(orig_image.GetOrigin(), spacing_ijk),
    }
    (out_dir / "roi_bbox.json").write_text(json.dumps(bbox_data, indent=2))
    log.info("ROI saved to %s", out_dir)
