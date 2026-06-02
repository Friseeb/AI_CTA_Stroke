"""Optional carotid / pericarotid fat features.

This module deliberately does NOT include a carotid segmentation model.
We expose a clean adapter that takes external carotid + optional plaque
masks (e.g. produced by a sibling pipeline) and emits the pericarotid fat
shell features. When no masks are provided every pericarotid_* column is
NaN with `perivascular_available=False`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .config import PerivascularConfig
from .geometry import mm_to_voxels
from .logging_utils import get_logger
from .types import CTAImage

log = get_logger("perivascular")

_NAN = float("nan")


def compute_perivascular_features(
    image: CTAImage,
    cfg: PerivascularConfig,
    fat_hu_min: float,
    fat_hu_max: float,
) -> dict:
    if not cfg.enabled:
        return _missing(reason="disabled")
    if not (cfg.carotid_mask_path and Path(cfg.carotid_mask_path).is_file()):
        return _missing(reason="no_carotid_mask")

    try:
        carotid = sitk.GetArrayFromImage(sitk.ReadImage(cfg.carotid_mask_path))
    except Exception as exc:
        log.warning("Carotid mask read failed: %s", exc)
        return _missing(reason="carotid_mask_read_failed")
    if carotid.shape != image.shape_zyx:
        log.warning("Carotid mask shape %s != CTA %s; skipping perivascular features.",
                    carotid.shape, image.shape_zyx)
        return _missing(reason="shape_mismatch")

    # Split left/right by column midline
    sx, _, _ = image.spacing_xyz_mm
    midline_x = image.array.shape[2] // 2
    left = carotid > 0
    left[:, :, midline_x:] = False
    right = (carotid > 0) & ~left

    shell_vox = mm_to_voxels(cfg.pericarotid_shell_mm, sx)
    shell_left = ndimage.binary_dilation(left, iterations=shell_vox) & ~left
    shell_right = ndimage.binary_dilation(right, iterations=shell_vox) & ~right

    arr_hu = image.array
    fat = (arr_hu >= fat_hu_min) & (arr_hu <= fat_hu_max)
    pf_left = fat & shell_left
    pf_right = fat & shell_right

    def stats(mask: np.ndarray, prefix: str) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {f"{prefix}_volume_ml": _NAN, f"{prefix}_mean_hu": _NAN, f"{prefix}_voxel_count": 0}
        vol_ml = n * image.voxel_volume_mm3 / 1000.0
        return {
            f"{prefix}_volume_ml": round(vol_ml, 3),
            f"{prefix}_mean_hu": round(float(arr_hu[mask].mean()), 2),
            f"{prefix}_voxel_count": n,
        }

    out = {"perivascular_available": True,
           "pericarotid_shell_mm_used": cfg.pericarotid_shell_mm}
    out.update(stats(pf_left, "pericarotid_fat_left"))
    out.update(stats(pf_right, "pericarotid_fat_right"))
    out.update(stats(pf_left | pf_right, "pericarotid_fat"))
    lv = out["pericarotid_fat_left_volume_ml"]
    rv = out["pericarotid_fat_right_volume_ml"]
    out["pericarotid_fat_asymmetry"] = (
        round((rv - lv) / (rv + lv), 3) if (isinstance(lv, float) and isinstance(rv, float)
                                            and (lv + rv) > 0) else _NAN
    )

    # Plaque burden hook: if a plaque mask is provided, just emit voxel
    # count + volume. We don't compute Hounsfield-based composition here.
    if cfg.plaque_mask_path and Path(cfg.plaque_mask_path).is_file():
        try:
            plaque = sitk.GetArrayFromImage(sitk.ReadImage(cfg.plaque_mask_path)) > 0
            if plaque.shape == image.shape_zyx:
                n_pl = int(plaque.sum())
                out["carotid_calcification_present"] = bool(n_pl > 0)
                out["carotid_plaque_volume_ml"] = round(n_pl * image.voxel_volume_mm3 / 1000.0, 3)
            else:
                out["carotid_calcification_present"] = False
                out["carotid_plaque_volume_ml"] = _NAN
        except Exception as exc:
            log.warning("Plaque mask read failed: %s", exc)
            out["carotid_calcification_present"] = False
            out["carotid_plaque_volume_ml"] = _NAN
    else:
        out["carotid_calcification_present"] = False
        out["carotid_plaque_volume_ml"] = _NAN

    return out


def _missing(reason: str) -> dict:
    return {
        "perivascular_available": False,
        "perivascular_reason": reason,
        "pericarotid_shell_mm_used": _NAN,
        "pericarotid_fat_left_volume_ml": _NAN,
        "pericarotid_fat_left_mean_hu": _NAN,
        "pericarotid_fat_right_volume_ml": _NAN,
        "pericarotid_fat_right_mean_hu": _NAN,
        "pericarotid_fat_volume_ml": _NAN,
        "pericarotid_fat_mean_hu": _NAN,
        "pericarotid_fat_asymmetry": _NAN,
        "carotid_calcification_present": False,
        "carotid_plaque_volume_ml": _NAN,
    }
