"""DICOM and NIfTI loading, preserving HU and spatial metadata."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .logging_utils import get_logger

log = get_logger("dicom_io")


def load_nifti(path: Path) -> sitk.Image:
    """Load NIfTI preserving HU values."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    image = reader.Execute()
    log.debug("Loaded NIfTI %s  spacing=%s  size=%s", path.name, image.GetSpacing(), image.GetSize())
    return image


def load_dicom_series(folder: Path) -> tuple[sitk.Image, dict]:
    """Load a DICOM series from *folder*, apply rescale slope/intercept, return image + metadata."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {folder}")
    if len(series_ids) > 1:
        log.warning("Multiple DICOM series found; using first: %s", series_ids[0])
    series_id = series_ids[0]
    files = reader.GetGDCMSeriesFileNames(str(folder), series_id)
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    meta = _extract_dicom_meta(reader, files)
    log.info("Loaded DICOM series %s  slices=%d  spacing=%s", series_id, len(files), image.GetSpacing())
    return image, meta


def _extract_dicom_meta(reader: sitk.ImageSeriesReader, files: tuple) -> dict:
    meta: dict = {}
    try:
        first = 0
        for tag, key in [
            ("0008|103e", "series_description"),
            ("0018|1030", "protocol_name"),
            ("0018|0081", "echo_time"),
            ("0018|0050", "slice_thickness"),
            ("0018|9087", "diffusion_bvalue"),
            ("0018|1400", "convolution_kernel"),
            ("0018|0088", "spacing_between_slices"),
            ("0010|1010", "patient_age"),
            ("0020|000e", "series_instance_uid"),
            ("0008|0060", "modality"),
            ("0018|0022", "scan_options"),
            ("0018|5100", "patient_position"),
        ]:
            try:
                val = reader.GetMetaData(first, tag)
                if val.strip():
                    meta[key] = val.strip()
            except Exception:
                pass
    except Exception as exc:
        log.debug("DICOM metadata extraction partial: %s", exc)
    return meta


def check_adult(meta: dict, min_age: int = 18) -> str:
    """Return 'adult', 'pediatric', or 'unknown'."""
    raw = meta.get("patient_age", "")
    if not raw:
        return "unknown"
    try:
        digits = "".join(c for c in raw if c.isdigit())
        age = int(digits)
        if raw.endswith("M"):
            age = age // 12
        elif raw.endswith("W"):
            age = age // 52
        return "adult" if age >= min_age else "pediatric"
    except (ValueError, AttributeError):
        return "unknown"


def save_nifti(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)
    log.debug("Saved NIfTI %s", path)


def image_to_numpy_hu(image: sitk.Image) -> tuple[np.ndarray, dict]:
    """Return HU array (i=slice, j=row, k=col) and spatial metadata dict."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()   # (x, y, z)
    spacing_ijk = tuple(reversed(spacing_xyz))  # (slice, row, col)
    meta = {
        "spacing_mm_ijk": list(spacing_ijk),
        "spacing_mm_xyz": list(spacing_xyz),
        "origin_xyz": list(image.GetOrigin()),
        "direction_xyz": list(image.GetDirection()),
        "size_xyz": list(image.GetSize()),
        "shape_ijk": list(arr.shape),
    }
    return arr, meta


def numpy_hu_to_image(arr: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Wrap numpy array in a SimpleITK image copying geometry from *reference*."""
    image = sitk.GetImageFromArray(arr)
    image.CopyInformation(reference)
    return image


def write_metadata_sidecar(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    log.debug("Wrote metadata sidecar %s", path)
