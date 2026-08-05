"""Safe binary NIfTI loading at the FLOWCAT boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .coordinate_systems import assert_geometry_compatible, validate_orientation
from .exceptions import FlowcatGeometryError, FlowcatSchemaError
from .schemas import ImageGeometry


@dataclass(frozen=True, slots=True)
class BinaryNifti:
    data: np.ndarray
    geometry: ImageGeometry
    source_file: str


def geometry_from_nifti(image_or_path: Any) -> ImageGeometry:
    """Extract three-dimensional NIfTI geometry in explicit RAS+ world units."""

    image = nib.load(str(image_or_path)) if isinstance(image_or_path, (str, Path)) else image_or_path
    if not hasattr(image, "shape") or not hasattr(image, "affine"):
        raise TypeError("Expected a nibabel NIfTI image or a NIfTI path")
    if len(image.shape) != 3:
        raise FlowcatGeometryError(f"FLOWCAT NIfTI must be 3D, got shape {image.shape!r}")
    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise FlowcatGeometryError("NIfTI affine must be a finite 4x4 matrix")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise FlowcatGeometryError("NIfTI affine is singular")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine))
    if not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise FlowcatGeometryError(f"Invalid NIfTI spacing: {spacing!r}")
    orientation = tuple(str(value) for value in nib.aff2axcodes(affine))
    validate_orientation(orientation)
    affine_tuple = tuple(tuple(float(value) for value in row) for row in affine)
    return ImageGeometry(
        shape_ijk=tuple(int(value) for value in image.shape),
        spacing_mm=spacing,
        affine_ras_mm=affine_tuple,  # type: ignore[arg-type]
        orientation=orientation,  # type: ignore[arg-type]
    )


def load_binary_nifti(
    path: str | Path,
    *,
    reference_geometry: ImageGeometry | None = None,
    affine_tolerance_mm: float = 1e-4,
) -> BinaryNifti:
    """Load a strictly binary, finite 3D NIfTI and preserve its affine.

    Values may be stored as integer or floating point, but every voxel must be
    numerically 0 or 1. This prevents accidental use of probability or
    multi-label maps as an arterial mask.
    """

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"FLOWCAT segmentation not found: {file_path}")
    try:
        image = nib.load(str(file_path))
    except Exception as exc:
        raise FlowcatSchemaError(f"Could not read FLOWCAT NIfTI {file_path}: {exc}") from exc
    geometry = geometry_from_nifti(image)
    if reference_geometry is not None:
        assert_geometry_compatible(geometry, reference_geometry, atol_mm=affine_tolerance_mm)

    data = np.asanyarray(image.dataobj)
    if not np.issubdtype(data.dtype, np.number) and data.dtype != np.bool_:
        raise FlowcatSchemaError(f"Binary NIfTI has non-numeric dtype {data.dtype}")
    if not np.isfinite(data).all():
        raise FlowcatSchemaError("Binary NIfTI contains NaN or infinite values")
    binary_values = np.logical_or(np.isclose(data, 0.0), np.isclose(data, 1.0))
    if not bool(binary_values.all()):
        values = np.unique(data)
        preview = values[:10].tolist()
        raise FlowcatSchemaError(
            f"FLOWCAT segmentation must be binary 0/1; observed values include {preview!r}"
        )
    return BinaryNifti(np.asarray(data != 0, dtype=bool), geometry, str(file_path.resolve()))
