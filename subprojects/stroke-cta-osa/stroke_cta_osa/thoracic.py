"""Optional thoracic / cardiac / epicardial fat features.

Stub. The repo does not have epicardial-fat segmentation infrastructure
ready to consume, so this module accepts pre-computed masks and emits
volume + mean HU summaries only. No deep learning here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import ThoracicConfig
from .logging_utils import get_logger
from .types import CTAImage

log = get_logger("thoracic")

_NAN = float("nan")


def compute_thoracic_features(
    image: CTAImage,
    cfg: ThoracicConfig,
    fat_hu_min: float,
    fat_hu_max: float,
) -> dict:
    if not cfg.enabled:
        return _missing("disabled")

    out: dict = {"thoracic_available": True}

    out.update(_mask_stats(
        cfg.mediastinal_mask_path, "mediastinal_fat", image, fat_hu_min, fat_hu_max
    ))
    out.update(_mask_stats(
        cfg.epicardial_mask_path, "epicardial_adipose_tissue", image, fat_hu_min, fat_hu_max
    ))
    # Pericardial alias: same mask. Researchers may want both column names.
    out["pericardial_fat_volume_ml"] = out.get("epicardial_adipose_tissue_volume_ml", _NAN)

    valid = [
        v for v in (out.get("epicardial_adipose_tissue_mean_hu"),
                    out.get("mediastinal_fat_mean_hu"))
        if isinstance(v, float) and v == v  # not NaN
    ]
    out["thoracic_fat_mean_hu"] = round(float(np.mean(valid)), 2) if valid else _NAN
    return out


def _mask_stats(
    mask_path: str | None, prefix: str, image: CTAImage,
    fat_hu_min: float, fat_hu_max: float,
) -> dict:
    if not mask_path or not Path(mask_path).is_file():
        return {f"{prefix}_volume_ml": _NAN, f"{prefix}_mean_hu": _NAN}
    try:
        m = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)) > 0
    except Exception as exc:
        log.warning("%s read failed: %s", prefix, exc)
        return {f"{prefix}_volume_ml": _NAN, f"{prefix}_mean_hu": _NAN}
    if m.shape != image.shape_zyx or not m.any():
        return {f"{prefix}_volume_ml": _NAN, f"{prefix}_mean_hu": _NAN}
    fat = m & (image.array >= fat_hu_min) & (image.array <= fat_hu_max)
    n = int(fat.sum())
    if n == 0:
        return {f"{prefix}_volume_ml": 0.0, f"{prefix}_mean_hu": _NAN}
    return {
        f"{prefix}_volume_ml": round(n * image.voxel_volume_mm3 / 1000.0, 3),
        f"{prefix}_mean_hu": round(float(image.array[fat].mean()), 2),
    }


def _missing(reason: str) -> dict:
    return {
        "thoracic_available": False,
        "thoracic_reason": reason,
        "mediastinal_fat_volume_ml": _NAN,
        "mediastinal_fat_mean_hu": _NAN,
        "epicardial_adipose_tissue_volume_ml": _NAN,
        "epicardial_adipose_tissue_mean_hu": _NAN,
        "pericardial_fat_volume_ml": _NAN,
        "thoracic_fat_mean_hu": _NAN,
    }
