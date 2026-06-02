"""Synthetic CTA fixtures used across the test suite.

We build a small, deterministic, 1mm-isotropic volume with:
  * a "body" ellipsoid filled with soft-tissue HU
  * a "subcutaneous" fat shell at -100 HU just under the body surface
  * a "pharyngeal airway" vertical tube at -800 HU through the middle
  * a "parapharyngeal fat" pair of slabs flanking the airway at -120 HU
  * a "retropharyngeal fat" slab posterior to the airway at -110 HU

Everything is large enough (a few hundred voxels for fat, ~50 voxels for
airway CSA) that downstream features have non-trivial values and tests can
assert ranges instead of exact equality (which is robust to refactors).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import SimpleITK as sitk

from stroke_cta_osa.types import CTAImage


# Volume shape (z, y, x). z=80 → 80 mm cranio-caudal.
SHAPE = (80, 80, 80)
SPACING = (1.0, 1.0, 1.0)


def _build_synth_array() -> np.ndarray:
    arr = np.full(SHAPE, -1000, dtype=np.int16)  # air outside

    # Body ellipsoid centered at (z=40, y=40, x=40) with radii (35, 35, 35)
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    cz, cy, cx = 40, 40, 40
    body = ((zz - cz) / 38) ** 2 + ((yy - cy) / 35) ** 2 + ((xx - cx) / 35) ** 2 <= 1.0
    arr[body] = 40  # soft tissue HU

    # Subcutaneous fat shell ~6mm thick (≈6 voxels) at body surface
    inner = ((zz - cz) / 32) ** 2 + ((yy - cy) / 29) ** 2 + ((xx - cx) / 29) ** 2 <= 1.0
    sub_fat = body & ~inner
    arr[sub_fat] = -100  # fat HU

    # Pharyngeal airway: a vertical 5-voxel-radius tube through z=15..70
    tube_xy = ((yy - 44) ** 2 + (xx - 40) ** 2) <= 25  # (1, y, x)
    z_band = (zz >= 15) & (zz < 70)                      # (z, 1, 1)
    tube = np.broadcast_to(tube_xy, SHAPE) & np.broadcast_to(z_band, SHAPE)
    arr[tube] = -800  # air

    # Parapharyngeal fat: two slabs left/right of the airway in z=25..55
    pp_left = (xx >= 28) & (xx < 33) & (yy >= 40) & (yy < 52)
    pp_right = (xx >= 47) & (xx < 52) & (yy >= 40) & (yy < 52)
    pp_z = (zz >= 25) & (zz < 55)
    pp_mask = (np.broadcast_to(pp_left, SHAPE) | np.broadcast_to(pp_right, SHAPE)) \
              & np.broadcast_to(pp_z, SHAPE)
    arr[pp_mask] = -120

    # Retropharyngeal fat: slab posterior to airway (y >= 50)
    rp_xy = (yy >= 50) & (yy < 56) & (xx >= 32) & (xx < 48)
    rp_z = (zz >= 30) & (zz < 60)
    rp_mask = np.broadcast_to(rp_xy, SHAPE) & np.broadcast_to(rp_z, SHAPE)
    arr[rp_mask] = -110

    return arr


@pytest.fixture(scope="session")
def synth_array() -> np.ndarray:
    return _build_synth_array()


@pytest.fixture
def synth_cta(synth_array, tmp_path) -> CTAImage:
    return CTAImage(
        array=synth_array.copy(),
        spacing_xyz_mm=SPACING,
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=tmp_path / "synth.nii.gz",
        study_id="stu_synth",
        scan_id="scn_synth",
        orientation_code="RAS",
        is_contrast_enhanced=False,
        sidecar={"input_kind": "nifti"},
    )


@pytest.fixture
def synth_nifti_path(synth_array, tmp_path) -> Path:
    """Persist the synthetic CTA to disk so I/O tests can hit it like real data."""
    img = sitk.GetImageFromArray(synth_array.astype(np.int16))
    img.SetSpacing(SPACING)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out = tmp_path / "synth.nii.gz"
    sitk.WriteImage(img, str(out), useCompression=True)
    return out


@pytest.fixture
def synth_airway_mask_path(synth_array, tmp_path) -> Path:
    """Externally-provided airway mask matching the synthetic tube."""
    arr = (synth_array == -800).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SPACING)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out = tmp_path / "airway_mask.nii.gz"
    sitk.WriteImage(img, str(out), useCompression=True)
    return out
