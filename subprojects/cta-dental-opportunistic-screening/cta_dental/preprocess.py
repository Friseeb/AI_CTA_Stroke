"""Preprocessing: reorientation, isotropic resampling, HU clipping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .config import PreprocessingConfig
from .logging_utils import get_logger

log = get_logger("preprocess")

_VALID_ORIENTATIONS = {"RAS", "LPS", "LAS", "RPS", "LAS", "AIR", "PIL"}


@dataclass
class PreprocessResult:
    resampled: sitk.Image           # isotropic resampled, reoriented
    clipped: sitk.Image             # HU-clipped copy for neural input
    original: sitk.Image            # unmodified original
    meta: dict                      # provenance record


def reorient(image: sitk.Image, orientation: str = "RAS") -> sitk.Image:
    orientation = orientation.upper()
    orienter = sitk.DICOMOrientImageFilter()
    try:
        # SimpleITK ≥ 2.3 uses string codes directly
        orienter.SetDesiredCoordinateOrientation(orientation)
    except Exception:
        log.warning("DICOMOrientImageFilter string API unavailable; skipping reorientation.")
        return image
    return orienter.Execute(image)


def resample_isotropic(
    image: sitk.Image,
    target_spacing_mm: float,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    target_spacing = np.array([target_spacing_mm] * 3)
    new_size = np.round(original_size * original_spacing / target_spacing).astype(int)
    new_size = [max(1, int(s)) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000.0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def clip_hu(image: sitk.Image, hu_min: float, hu_max: float) -> sitk.Image:
    clamp = sitk.ClampImageFilter()
    clamp.SetLowerBound(hu_min)
    clamp.SetUpperBound(hu_max)
    return clamp.Execute(sitk.Cast(image, sitk.sitkFloat32))


def preprocess(
    image: sitk.Image,
    cfg: PreprocessingConfig,
    dicom_meta: Optional[dict] = None,
) -> PreprocessResult:
    log.info("Reorienting to %s …", cfg.orientation)
    oriented = reorient(image, cfg.orientation)

    log.info("Resampling to %.2f mm isotropic …", cfg.target_spacing_mm)
    resampled = resample_isotropic(oriented, cfg.target_spacing_mm)

    log.info("Creating HU-clipped copy [%.0f, %.0f] …", cfg.hu_clip_min, cfg.hu_clip_max)
    clipped = clip_hu(resampled, cfg.hu_clip_min, cfg.hu_clip_max)

    orig_sp = list(image.GetSpacing())
    tgt_sp = list(resampled.GetSpacing())
    meta = {
        "original_spacing_xyz_mm": orig_sp,
        "target_spacing_xyz_mm": tgt_sp,
        "original_size_xyz": list(image.GetSize()),
        "resampled_size_xyz": list(resampled.GetSize()),
        "orientation": cfg.orientation,
        "hu_clip_min": cfg.hu_clip_min,
        "hu_clip_max": cfg.hu_clip_max,
    }
    if dicom_meta:
        for k in ("series_instance_uid", "convolution_kernel", "protocol_name", "modality", "series_description"):
            if k in dicom_meta:
                meta[k] = dicom_meta[k]

    log.info(
        "Preprocessing complete. Original spacing %s → target %s. Shape %s → %s.",
        orig_sp, tgt_sp, list(image.GetSize()), list(resampled.GetSize()),
    )
    return PreprocessResult(resampled=resampled, clipped=clipped, original=image, meta=meta)
