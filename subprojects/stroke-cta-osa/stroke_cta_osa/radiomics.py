"""Optional PyRadiomics integration.

Now supports nine ROI families:

  * airway, tongue, posterior_tongue, soft_palate, lateral_wall,
  * cervical_fat, parapharyngeal_fat, retropharyngeal_fat,
  * combined_airway_soft_tissue.

Behaviour:
  * `cfg.enabled = False` → returns `{"radiomics_available": False,
    "radiomics_reason": "disabled"}`.
  * pyradiomics not importable → returns `{..., "reason":
    "pyradiomics_not_installed"}`.
  * any ROI failing extraction is logged as a warning but never breaks the
    others; per-ROI presence is in `rad_<roi>_available`.

Feature names are normalised so they're prefixed by `rad_<roi>_` and lose
the redundant `original_<class>_` prefix from PyRadiomics — making the
feature column names short enough to model on.
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
        return {"radiomics_available": False,
                "radiomics_reason": "pyradiomics_not_installed"}

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
            "ngtdm": [],
            "gldm": [],
        },
    }
    try:
        extractor = fe.RadiomicsFeatureExtractor(params)
    except Exception as exc:
        return {"radiomics_available": False,
                "radiomics_reason": f"extractor_init_failed:{exc}"}

    out: dict = {
        "radiomics_available": True,
        "radiomics_engine": "pyradiomics",
        "radiomics_rois_configured": ";".join(cfg.rois),
    }
    for roi in cfg.rois:
        mask = masks.get(roi)
        if mask is None or not np.asarray(mask).any():
            out[f"rad_{roi}_available"] = False
            continue
        try:
            feats = _extract_one(extractor, sitk, image, np.asarray(mask), cfg.label_value)
        except Exception as exc:
            log.warning("Radiomics failed on %s: %s", roi, exc)
            out[f"rad_{roi}_available"] = False
            out[f"rad_{roi}_error"] = str(exc)[:200]
            continue
        out[f"rad_{roi}_available"] = True
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            col = f"rad_{roi}_{k.replace('original_', '')}"
            try:
                out[col] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def _extract_one(extractor, sitk, image: CTAImage, mask: np.ndarray, label: int) -> dict:
    mask_uint = mask.astype(np.uint8) * label
    mask_img = sitk.GetImageFromArray(mask_uint)
    mask_img.SetSpacing(image.spacing_xyz_mm)
    mask_img.SetOrigin(image.origin_xyz_mm)
    mask_img.SetDirection(image.direction_3x3)
    base_image = sitk.GetImageFromArray(image.array.astype(np.float32))
    base_image.SetSpacing(image.spacing_xyz_mm)
    base_image.SetOrigin(image.origin_xyz_mm)
    base_image.SetDirection(image.direction_3x3)
    return extractor.execute(base_image, mask_img)
