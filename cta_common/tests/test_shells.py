"""Tests for cta_common.shells (EDT peri-vessel shell generation)."""

import numpy as np
import pytest

from cta_common.shells import (
    combined_periaortic_shell,
    create_aorta_wall_band_masks,
    external_shell,
    internal_boundary_shell,
    local_shell_around_mask,
)

SPACING = (1.0, 1.0, 1.0)  # (x, y, z)


def _solid_cube(n=21, half=4):
    a = np.zeros((n, n, n), dtype=bool)
    c = n // 2
    a[c - half:c + half + 1, c - half:c + half + 1, c - half:c + half + 1] = True
    return a


def test_external_shell_is_outside_only_and_within_band():
    mask = _solid_cube()
    shell = external_shell(mask, SPACING, inner_mm=0.0, outer_mm=2.0)
    assert shell.any()
    assert not (shell & mask).any()  # strictly outside the mask


def test_external_shell_requires_outer_gt_inner():
    with pytest.raises(ValueError):
        external_shell(_solid_cube(), SPACING, inner_mm=3.0, outer_mm=2.0)


def test_external_shell_empty_mask_returns_empty():
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert not external_shell(empty, SPACING, 0.0, 2.0).any()


def test_internal_boundary_shell_inside_only():
    mask = _solid_cube()
    inner = internal_boundary_shell(mask, SPACING, depth_mm=1.0)
    assert inner.any()
    assert not (inner & ~mask).any()  # entirely within the mask
    assert not inner[mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2]  # core excluded


def test_wall_band_is_internal_or_external_union():
    mask = _solid_cube()
    bands = create_aorta_wall_band_masks(mask, SPACING, internal_mm=1.0, external_mm=2.0)
    assert set(bands) == {"aorta_wall_internal", "aorta_wall_external", "aorta_wall_band"}
    assert np.array_equal(bands["aorta_wall_band"], bands["aorta_wall_internal"] | bands["aorta_wall_external"])


def test_combined_shell_matches_external_plus_internal():
    mask = _solid_cube()
    combined = combined_periaortic_shell(mask, SPACING, outer_mm=3.0, internal_mm=1.0)
    expected = external_shell(mask, SPACING, 0.0, 3.0) | internal_boundary_shell(mask, SPACING, 1.0)
    assert np.array_equal(combined, expected)


def test_local_shell_excludes_base_mask():
    seed = np.zeros((21, 21, 21), dtype=bool)
    seed[10, 10, 10] = True
    exclusion = np.zeros_like(seed)
    exclusion[10, 10, 11] = True  # an adjacent excluded voxel
    shell = local_shell_around_mask(seed, exclusion, SPACING, outer_mm=2.0)
    assert shell.any()
    assert not (shell & exclusion).any()
    assert not shell[10, 10, 10]  # seed itself not in the outside shell


def test_anisotropic_spacing_band_thicker_along_fine_axis():
    # finer z spacing -> EDT distance grows slower in z -> more z voxels in band
    mask = _solid_cube()
    shell = external_shell(mask, (1.0, 1.0, 0.5), inner_mm=0.0, outer_mm=2.0)
    assert shell.any()
    assert not (shell & mask).any()
