"""QC utilities for vessel segmentation outputs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi


@dataclass
class QCThresholds:
    min_mask_voxels: int = 200
    max_mask_voxels: int = 2_000_000
    min_centerline_voxels: int = 50
    max_centerline_voxels: int = 200_000


@dataclass
class QCResult:
    mask_voxels: int
    mask_volume_mm3: float
    bbox_mm: List[float]
    centerline_voxels: int
    centerline_length_mm: float
    flags: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "mask_voxels": self.mask_voxels,
            "mask_volume_mm3": self.mask_volume_mm3,
            "bbox_mm": self.bbox_mm,
            "centerline_voxels": self.centerline_voxels,
            "centerline_length_mm": self.centerline_length_mm,
            "flags": self.flags,
        }

    def to_flat_dict(self) -> Dict[str, object]:
        return {
            "mask_voxels": self.mask_voxels,
            "mask_volume_mm3": round(self.mask_volume_mm3, 2),
            "bbox_x_mm": self.bbox_mm[0],
            "bbox_y_mm": self.bbox_mm[1],
            "bbox_z_mm": self.bbox_mm[2],
            "centerline_voxels": self.centerline_voxels,
            "centerline_length_mm": round(self.centerline_length_mm, 2),
            "flags": ";".join(self.flags),
        }


def _voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def _bbox_size_mm(mask: np.ndarray, voxel_sizes) -> List[float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return [0.0, 0.0, 0.0]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = (maxs - mins + 1) * voxel_sizes
    return [float(spans[0]), float(spans[1]), float(spans[2])]


def _centerline_length(centerline_mask: np.ndarray, voxel_sizes) -> float:
    coords = np.argwhere(centerline_mask)
    if len(coords) < 2:
        return 0.0
    coords_mm = coords * voxel_sizes
    diffs = np.diff(coords_mm, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_qc_metrics(
    mask_img: nib.Nifti1Image,
    centerline_img: Optional[nib.Nifti1Image] = None,
    thresholds: Optional[QCThresholds] = None,
) -> QCResult:
    thresholds = thresholds or QCThresholds()

    mask_data = mask_img.get_fdata() > 0
    voxel_vol = _voxel_volume_mm3(mask_img)
    voxel_sizes = np.array(nib.affines.voxel_sizes(mask_img.affine))

    mask_voxels = int(mask_data.sum())
    mask_volume_mm3 = mask_voxels * voxel_vol
    bbox_mm = _bbox_size_mm(mask_data, voxel_sizes)

    centerline_voxels = 0
    centerline_length_mm = 0.0
    if centerline_img is not None:
        cl_data = centerline_img.get_fdata() > 0
        centerline_voxels = int(cl_data.sum())
        centerline_length_mm = _centerline_length(cl_data, voxel_sizes)

    flags: List[str] = []
    if mask_voxels < thresholds.min_mask_voxels:
        flags.append("mask_too_small")
    if mask_voxels > thresholds.max_mask_voxels:
        flags.append("mask_too_large")
    if centerline_img is not None:
        if centerline_voxels < thresholds.min_centerline_voxels:
            flags.append("centerline_too_small")
        if centerline_voxels > thresholds.max_centerline_voxels:
            flags.append("centerline_too_large")
    if math.isclose(centerline_length_mm, 0.0) and centerline_img is not None:
        flags.append("centerline_zero_length")

    return QCResult(
        mask_voxels=mask_voxels,
        mask_volume_mm3=mask_volume_mm3,
        bbox_mm=bbox_mm,
        centerline_voxels=centerline_voxels,
        centerline_length_mm=centerline_length_mm,
        flags=flags,
    )


def simple_centerline(mask_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Approximate a centerline using distance ridge extraction.

    This is intentionally lightweight and avoids external dependencies. It is
    meant for QC previews and can be replaced with a full skeletonization
    method when available.
    """
    mask = mask_img.get_fdata() > 0
    if mask.sum() == 0:
        return nib.Nifti1Image(np.zeros_like(mask, dtype=np.uint8), mask_img.affine, mask_img.header)

    dist = ndi.distance_transform_edt(mask)
    ridge = dist == ndi.maximum_filter(dist, size=3)
    ridge = ridge & mask
    return nib.Nifti1Image(ridge.astype(np.uint8), mask_img.affine, mask_img.header)
