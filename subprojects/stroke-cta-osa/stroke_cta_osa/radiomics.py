"""Optional PyRadiomics integration.

When `cfg.radiomics.enabled` is False, or when pyradiomics is not importable,
this module returns a dict with only `radiomics_available=False` plus a
reason. Otherwise it returns the standard radiomic feature set per requested
ROI, prefixed by `rad_<roi>_`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import RadiomicsConfig
from .logging_utils import get_logger
from .types import CTAImage

log = get_logger("radiomics")


def _try_import_pyradiomics():
    try:
        from radiomics import featureextractor  # type: ignore
        return featureextractor
    except Exception:
        return None


def compute_radiomics(
    image: CTAImage,
    cfg: RadiomicsConfig,
    masks: dict[str, np.ndarray],
) -> dict:
    if not cfg.enabled:
        return {"radiomics_available": False, "radiomics_reason": "disabled"}
    fe = _try_import_pyradiomics()
    if fe is None:
        return {"radiomics_available": False, "radiomics_reason": "pyradiomics_not_installed"}

    import SimpleITK as sitk
    base_image = sitk.GetImageFromArray(image.array.astype(np.float32))
    base_image.SetSpacing(image.spacing_xyz_mm)
    base_image.SetOrigin(image.origin_xyz_mm)
    base_image.SetDirection(image.direction_3x3)

    params = {
        "setting": {
            "binWidth": cfg.bin_width_hu,
            "label": cfg.label_value,
            "interpolator": "sitkNearestNeighbor",
            "normalize": False,
        },
        "imageType": {"Original": {}},
        "featureClass": {
            "firstorder": [],
            "shape": [],
            "glcm": [],
            "glrlm": [],
            "glszm": [],
        },
    }
    extractor = fe.RadiomicsFeatureExtractor(params)

    out: dict = {"radiomics_available": True, "radiomics_engine": "pyradiomics"}
    for roi in cfg.rois:
        mask = masks.get(roi)
        if mask is None or not mask.any():
            out[f"rad_{roi}_available"] = False
            continue
        mask_uint = mask.astype(np.uint8) * cfg.label_value
        mask_img = sitk.GetImageFromArray(mask_uint)
        mask_img.SetSpacing(image.spacing_xyz_mm)
        mask_img.SetOrigin(image.origin_xyz_mm)
        mask_img.SetDirection(image.direction_3x3)
        try:
            feats = extractor.execute(base_image, mask_img)
        except Exception as exc:
            log.warning("Radiomics failed on %s: %s", roi, exc)
            out[f"rad_{roi}_available"] = False
            out[f"rad_{roi}_error"] = str(exc)[:200]
            continue
        out[f"rad_{roi}_available"] = True
        for k, v in feats.items():
            # Skip diagnostics columns
            if k.startswith("diagnostics_"):
                continue
            col = f"rad_{roi}_{k.replace('original_', '')}"
            try:
                out[col] = float(v)
            except (TypeError, ValueError):
                continue
    return out
