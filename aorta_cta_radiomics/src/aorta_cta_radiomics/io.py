"""NIfTI image and mask I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageVolume:
    """SimpleITK-backed volume plus NumPy array in z, y, x order."""

    image: object
    array: np.ndarray
    path: Path

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.image.GetSpacing())

    @property
    def origin_xyz(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.image.GetOrigin())

    @property
    def direction(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.image.GetDirection())

    @property
    def size_xyz(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.image.GetSize())


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for NIfTI I/O. Install the conda environment "
            "or run `pip install SimpleITK`."
        ) from exc
    return sitk


def read_volume(path: str | Path) -> ImageVolume:
    """Read an image volume with SimpleITK."""
    sitk = _sitk()
    volume_path = Path(path)
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)
    return ImageVolume(image=image, array=array, path=volume_path)


def read_mask(path: str | Path) -> ImageVolume:
    """Read a binary mask and convert non-zero values to True."""
    volume = read_volume(path)
    return ImageVolume(
        image=volume.image,
        array=volume.array > 0,
        path=volume.path,
    )


def same_physical_space(
    moving: object,
    reference: object,
    spacing_tol: float = 1e-5,
    origin_tol: float = 1e-4,
    direction_tol: float = 1e-5,
) -> bool:
    """Check size, spacing, origin, and direction compatibility."""
    if tuple(moving.GetSize()) != tuple(reference.GetSize()):
        return False
    spacing_ok = np.allclose(moving.GetSpacing(), reference.GetSpacing(), atol=spacing_tol)
    origin_ok = np.allclose(moving.GetOrigin(), reference.GetOrigin(), atol=origin_tol)
    direction_ok = np.allclose(moving.GetDirection(), reference.GetDirection(), atol=direction_tol)
    return bool(spacing_ok and origin_ok and direction_ok)


def resample_mask_to_image(mask_image: object, reference_image: object) -> object:
    """Resample a mask into image space with nearest-neighbor interpolation."""
    sitk = _sitk()
    return sitk.Resample(
        mask_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def load_image_and_mask(
    image_path: str | Path,
    mask_path: str | Path,
    resample_mask_if_needed: bool = True,
) -> tuple[ImageVolume, ImageVolume, bool]:
    """Load CTA image and aorta mask, resampling the mask if requested."""
    sitk = _sitk()
    image = read_volume(image_path)
    mask = read_mask(mask_path)
    was_resampled = False

    if not same_physical_space(mask.image, image.image):
        if not resample_mask_if_needed:
            raise ValueError(
                "Image and mask do not share shape/spacing/origin/direction. "
                "Enable mask resampling or provide a mask in image space."
            )
        resampled = resample_mask_to_image(mask.image, image.image)
        mask = ImageVolume(
            image=resampled,
            array=sitk.GetArrayFromImage(resampled) > 0,
            path=Path(mask_path),
        )
        was_resampled = True

    if image.array.shape != mask.array.shape:
        raise ValueError(
            f"Image and mask arrays differ after loading: {image.array.shape} vs {mask.array.shape}."
        )
    return image, mask, was_resampled


def write_mask_like(mask: np.ndarray, reference_image: object, output_path: str | Path) -> Path:
    """Write a binary mask using the spatial metadata of a reference image."""
    sitk = _sitk()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(path))
    return path


def write_label_like(labels: np.ndarray, reference_image: object, output_path: str | Path) -> Path:
    """Write an integer label map using the spatial metadata of a reference image."""
    sitk = _sitk()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(labels.astype(np.uint16))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(path))
    return path


def voxel_to_physical(
    zyx: np.ndarray,
    reference_image: object | None,
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Convert z, y, x voxel coordinates to physical x, y, z coordinates."""
    coords = np.asarray(zyx, dtype=float)
    xyz = coords[:, [2, 1, 0]]
    if reference_image is None:
        return xyz * np.asarray(spacing_xyz, dtype=float)

    physical = [reference_image.TransformIndexToPhysicalPoint(tuple(int(v) for v in row)) for row in xyz]
    return np.asarray(physical, dtype=float)
