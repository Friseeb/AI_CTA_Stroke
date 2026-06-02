"""De-identification and pixel defacing support.

Pixel defacing modes:
  none         — do nothing to the image.
  mask_only    — compute face/privacy region mask but do NOT alter analysis image.
  posthoc      — create defaced copy only for export/QC; analysis uses original.
  pre          — deface BEFORE analysis; warns loudly; keeps protected copy.

The analysis image is NEVER destructively altered unless mode='pre' and the user
explicitly selected it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import SimpleITK as sitk

from .config import DefaceConfig
from .logging_utils import get_logger

log = get_logger("deid")

_PRE_WARNING = (
    "WARNING: deface-mode='pre' selected. Segmentation performance may degrade "
    "because facial soft-tissue context used by models will be removed or altered. "
    "The undefaced working NIfTI is preserved in the protected intermediate folder. "
    "This flag is intended for privacy-preservation workflows only."
)


def scrub_metadata(meta: dict) -> dict:
    """Remove any PHI keys from a metadata sidecar dict."""
    phi_keys = {
        "patient_name", "patient_id", "accession_number",
        "patient_birth_date", "patient_age_raw",
        "study_date", "study_time", "referring_physician",
    }
    return {k: v for k, v in meta.items() if k.lower() not in phi_keys}


def compute_face_mask(image: sitk.Image) -> Optional[sitk.Image]:
    """Very simple anatomical face mask: anterior 1/3 of the volume in LR/AP extent.

    This is a placeholder implementation. For production use, wire in CTA-DEFACE
    via the configured executable (see apply_deface_external).
    Returns a binary mask image (1=face region) or None on failure.
    """
    try:
        size = image.GetSize()
        arr_size = list(reversed(size))  # ijk
        mask_arr = __import__("numpy").zeros(arr_size, dtype=__import__("numpy").uint8)
        # Anterior third heuristic: in RAS, anterior = high-index in AP axis.
        # Without a proper model this is illustrative only.
        ap_cutoff = int(arr_size[1] * 2 / 3)
        mask_arr[:, ap_cutoff:, :] = 1
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(image)
        return mask_img
    except Exception as exc:
        log.warning("Face mask computation failed: %s", exc)
        return None


def apply_deface_external(
    image_path: Path,
    output_path: Path,
    executable: Optional[str],
) -> bool:
    """Call CTA-DEFACE or CT-Defacer as external CLI. Returns True on success."""
    exe = executable or shutil.which("cta-deface") or shutil.which("ct-defacer")
    if exe is None:
        log.warning("No defacing executable found. Skipping external defacing.")
        return False
    cmd = [exe, str(image_path), str(output_path)]
    log.info("Running external deface: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Deface failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return False
    return True


def run_deface(
    image: sitk.Image,
    cfg: DefaceConfig,
    out_dir: Path,
    protected_dir: Path,
    image_path: Path,
) -> dict:
    """
    Execute the configured defacing strategy.

    Returns a dict with keys:
        mode, face_mask_path, defaced_export_path, protected_original_path
    """
    result: dict = {"mode": cfg.mode}

    if cfg.mode == "none":
        log.info("Deface mode=none. No defacing applied.")
        return result

    if cfg.mode == "mask_only":
        log.info("Deface mode=mask_only. Computing face mask (no image alteration).")
        mask = compute_face_mask(image)
        if mask is not None:
            mask_path = out_dir / "face_mask.nii.gz"
            sitk.WriteImage(mask, str(mask_path), useCompression=True)
            result["face_mask_path"] = str(mask_path)
        return result

    if cfg.mode == "posthoc":
        log.info("Deface mode=posthoc. Creating defaced export copy only.")
        defaced_path = out_dir / "defaced_export.nii.gz"
        success = apply_deface_external(image_path, defaced_path, cfg.executable)
        if not success:
            log.warning("Posthoc deface failed; falling back to mask_only.")
            mask = compute_face_mask(image)
            if mask is not None:
                mask_path = out_dir / "face_mask.nii.gz"
                sitk.WriteImage(mask, str(mask_path), useCompression=True)
                result["face_mask_path"] = str(mask_path)
        else:
            result["defaced_export_path"] = str(defaced_path)
        return result

    if cfg.mode == "pre":
        warnings.warn(_PRE_WARNING, UserWarning, stacklevel=2)
        log.warning(_PRE_WARNING)
        protected_dir.mkdir(parents=True, exist_ok=True)
        protected_path = protected_dir / "original_undefaced.nii.gz"
        sitk.WriteImage(image, str(protected_path), useCompression=True)
        result["protected_original_path"] = str(protected_path)
        defaced_path = out_dir / "analysis_defaced.nii.gz"
        success = apply_deface_external(image_path, defaced_path, cfg.executable)
        if not success:
            log.error("pre deface failed; analysis will use original image as fallback.")
            result["pre_deface_failed"] = True
        else:
            result["defaced_analysis_path"] = str(defaced_path)
        return result

    raise ValueError(f"Unknown deface mode: {cfg.mode}")
