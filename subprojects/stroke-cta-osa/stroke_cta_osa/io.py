"""Input-output: DICOM folder / zip / NIfTI ➜ CTAImage.

The hot path uses SimpleITK because the dental subproject already does, so
the two stacks stay consistent. NIfTI is preferred whenever it exists: it
already has the geometry baked in and avoids running our PHI scrubber on
DICOM tags that may or may not be present.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .dicom_utils import (
    check_adult, derive_ids_from_dicom, infer_contrast,
    safe_hash, scrub_dicom_metadata,
)
from .logging_utils import get_logger
from .types import CTAImage

log = get_logger("io")


# --- Public helpers ---------------------------------------------------------

def load_input(
    path: Path,
    age_floor_years: int = 18,
    sidecar_path: Optional[Path] = None,
) -> tuple[CTAImage, dict]:
    """Load any supported input (DICOM dir, DICOM zip, NIfTI) → (CTAImage, scrubbed_meta).

    Returns a `(CTAImage, meta)` tuple. `meta` is already scrubbed: callers
    can persist it without re-checking PHI.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    if p.is_dir():
        image, raw_meta = _load_dicom_series(p)
        kind = "dicom_dir"
    elif p.suffix.lower() == ".zip":
        image, raw_meta = _load_dicom_zip(p)
        kind = "dicom_zip"
    elif p.suffix.lower() in (".nii", ".gz") or p.name.endswith(".nii.gz"):
        image, raw_meta = _load_nifti(p, sidecar_path)
        kind = "nifti"
    else:
        raise ValueError(f"Unsupported input type for {p}")

    age_status = check_adult(raw_meta, age_floor_years)
    if age_status == "pediatric":
        raise RuntimeError(
            "Patient is pediatric (age below configured floor) — refusing to process."
        )

    scrubbed = scrub_dicom_metadata(raw_meta)
    study_id, scan_id = derive_ids_from_dicom(raw_meta, p)

    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    cta = CTAImage(
        array=arr.astype(np.int16, copy=False),
        spacing_xyz_mm=tuple(float(s) for s in image.GetSpacing()),  # type: ignore[arg-type]
        origin_xyz_mm=tuple(float(o) for o in image.GetOrigin()),    # type: ignore[arg-type]
        direction_3x3=tuple(float(d) for d in image.GetDirection()),
        source_path=p,
        study_id=study_id,
        scan_id=scan_id,
        orientation_code="LPS",  # ITK NIfTI / DICOM native is LPS
        is_contrast_enhanced=infer_contrast(raw_meta),
        sidecar={"input_kind": kind, **scrubbed},
    )
    log.info("Loaded CTA %s (%s) shape=%s spacing=%s",
             study_id, kind, cta.shape_zyx, cta.spacing_xyz_mm)
    return cta, scrubbed


def to_sitk_image(cta: CTAImage) -> sitk.Image:
    """Reconstruct a SimpleITK Image from a CTAImage (for writing masks back)."""
    img = sitk.GetImageFromArray(cta.array)
    img.SetSpacing(cta.spacing_xyz_mm)
    img.SetOrigin(cta.origin_xyz_mm)
    img.SetDirection(cta.direction_3x3)
    return img


def save_mask(mask_zyx: np.ndarray, reference: CTAImage, out_path: Path) -> None:
    """Write a binary or label mask in the same geometry as `reference`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    img.SetSpacing(reference.spacing_xyz_mm)
    img.SetOrigin(reference.origin_xyz_mm)
    img.SetDirection(reference.direction_3x3)
    sitk.WriteImage(img, str(out_path), useCompression=True)


# --- Internal loaders -------------------------------------------------------

def _load_dicom_series(folder: Path) -> tuple[sitk.Image, dict]:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under {folder}")
    # Pick the largest series — most CTA cardiac/neck reformats put the
    # primary recon as the biggest series, and this also dodges localizer.
    chosen, n = None, -1
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(str(folder), sid)
        if len(files) > n:
            n = len(files)
            chosen = sid
    files = reader.GetGDCMSeriesFileNames(str(folder), chosen)
    reader.SetFileNames(files)
    image = reader.Execute()
    meta = _read_first_dicom_tags(files[0])
    meta["_series_file_count"] = len(files)
    return image, meta


def _load_dicom_zip(zip_path: Path) -> tuple[sitk.Image, dict]:
    tmp = Path(tempfile.mkdtemp(prefix="stroke_cta_osa_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        # find the first dir that has DICOM files
        for root, _, files in os.walk(tmp):
            if any(_looks_like_dicom(Path(root) / f) for f in files[:20]):
                return _load_dicom_series(Path(root))
        raise RuntimeError(f"Zip {zip_path.name} contains no DICOMs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_nifti(nifti_path: Path, sidecar_path: Optional[Path]) -> tuple[sitk.Image, dict]:
    image = sitk.ReadImage(str(nifti_path))
    meta: dict = {}
    # Sidecar conventions: explicit path, then <stem>.json next to the file.
    candidate = sidecar_path
    if candidate is None:
        candidate = nifti_path.with_suffix("").with_suffix(".json")
    if candidate and Path(candidate).is_file():
        try:
            meta = json.loads(Path(candidate).read_text())
        except Exception as exc:
            log.warning("Could not parse sidecar %s: %s", candidate.name, exc)
            meta = {}
    return image, meta


def _read_first_dicom_tags(file_path: str) -> dict:
    """Pull a small set of acquisition tags from the first slice without
    pulling in pydicom unless it's already in the environment.

    Falls back to an empty dict if neither pydicom nor a sitk metadata reader
    are usable — downstream code treats missing meta as a soft signal.
    """
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.LoadPrivateTagsOff()
        reader.ReadImageInformation()
        meta: dict = {}
        for key in reader.GetMetaDataKeys():
            try:
                meta[_dicom_tag_to_name(key)] = reader.GetMetaData(key)
            except Exception:
                continue
        return meta
    except Exception:
        try:
            import pydicom  # type: ignore
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            out = {}
            for elem in ds:
                try:
                    out[elem.keyword] = elem.value
                except Exception:
                    continue
            return out
        except Exception:
            return {}


# Subset of the GDCM-style "GGGG|EEEE" → keyword mapping we actually care
# about. Anything else is dropped by scrub_dicom_metadata() anyway.
_DICOM_KEY_MAP = {
    "0020|000d": "StudyInstanceUID",
    "0020|000e": "SeriesInstanceUID",
    "0008|0060": "Modality",
    "0018|0050": "SliceThickness",
    "0028|0030": "PixelSpacing",
    "0018|0088": "SpacingBetweenSlices",
    "0008|0070": "Manufacturer",
    "0008|1090": "ManufacturerModelName",
    "0018|0015": "BodyPartExamined",
    "0018|0060": "KVP",
    "0018|1151": "XRayTubeCurrent",
    "0018|1150": "ExposureTime",
    "0018|1210": "ConvolutionKernel",
    "0018|1100": "ReconstructionDiameter",
    "0018|0010": "ContrastBolusAgent",
    "0008|0008": "ImageType",
    "0010|1010": "PatientAge",
    "0010|0040": "PatientSex",
}


def _dicom_tag_to_name(key: str) -> str:
    return _DICOM_KEY_MAP.get(key.lower(), key)


def _looks_like_dicom(p: Path) -> bool:
    if p.suffix.lower() in (".dcm", ".ima"):
        return True
    try:
        with p.open("rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False
