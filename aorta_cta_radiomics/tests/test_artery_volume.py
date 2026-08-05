from __future__ import annotations

import numpy as np
import pytest

from aorta_cta_radiomics.artery_volume import (
    CenterlineSeed,
    UNCERTAIN_LABEL,
    branch_aware_connected_labels,
    generate_anatomical_artery_volume,
    nearest_centerline_labels,
)


def _cylinder(shape=(41, 41, 41), radius=4.0):
    grid = np.indices(shape)
    return ((grid[0] - 20) ** 2 + (grid[1] - 20) ** 2) <= radius**2


def test_nearest_centerline_recovers_straight_cylinder_label():
    lumen = _cylinder()
    points = np.column_stack(
        [np.full(31, 20.0), np.full(31, 20.0), np.arange(5.0, 36.0)]
    )
    branch = CenterlineSeed(
        branch_id="rcca",
        label_id=3,
        label_name="RCCA",
        points_ras_mm=points,
        radii_mm=np.full(len(points), 4.0),
    )
    labels, confidence = nearest_centerline_labels(lumen, np.eye(4), [branch])
    assert np.all(labels[lumen] == 3)
    assert np.all(confidence[lumen] > 0)
    assert not np.any(labels[~lumen])


def test_connected_propagation_does_not_cross_disconnected_components():
    lumen = np.zeros((30, 30, 30), dtype=bool)
    lumen[3:8, 3:8, 3:20] = True
    lumen[20:25, 20:25, 3:20] = True
    branch = CenterlineSeed(
        branch_id="lcca",
        label_id=4,
        label_name="LCCA",
        points_ras_mm=np.array([[5, 5, 4], [5, 5, 18]], dtype=float),
        radii_mm=np.array([2.0, 2.0]),
    )
    labels, metadata = branch_aware_connected_labels(lumen, np.eye(4), [branch])
    assert np.all(labels[3:8, 3:8, 3:20] == 4)
    assert np.all(labels[20:25, 20:25, 3:20] == UNCERTAIN_LABEL)
    assert metadata["unseeded_voxels"] == int(lumen[20:25, 20:25, 3:20].sum())


def test_bifurcation_is_retained_as_explicit_uncertainty():
    shape = (51, 51, 51)
    grid = np.indices(shape)
    trunk = ((grid[0] - 25) ** 2 + (grid[1] - 25) ** 2 <= 3**2) & (
        grid[2] <= 25
    )
    right = ((grid[0] - (25 + 0.35 * (grid[2] - 25))) ** 2 + (grid[1] - 25) ** 2 <= 3**2) & (
        grid[2] >= 25
    )
    left = ((grid[0] - (25 - 0.35 * (grid[2] - 25))) ** 2 + (grid[1] - 25) ** 2 <= 3**2) & (
        grid[2] >= 25
    )
    lumen = trunk | right | left
    z_trunk = np.arange(5.0, 26.0)
    z_child = np.arange(25.0, 46.0)
    branches = [
        CenterlineSeed(
            "rcca",
            3,
            "RCCA",
            np.column_stack([np.full(len(z_trunk), 25.0), np.full(len(z_trunk), 25.0), z_trunk]),
            np.full(len(z_trunk), 3.0),
        ),
        CenterlineSeed(
            "rica",
            9,
            "RICA",
            np.column_stack([25 + 0.35 * (z_child - 25), np.full(len(z_child), 25.0), z_child]),
            np.full(len(z_child), 3.0),
            parent_branch_id="rcca",
        ),
        CenterlineSeed(
            "reca",
            11,
            "RECA",
            np.column_stack([25 - 0.35 * (z_child - 25), np.full(len(z_child), 25.0), z_child]),
            np.full(len(z_child), 3.0),
            parent_branch_id="rcca",
        ),
    ]
    result = generate_anatomical_artery_volume(
        lumen, np.eye(4), branches, bifurcation_radius_mm=3.0
    )
    assert result.bifurcation_exclusion_mask[25, 25, 25]
    assert result.labels[25, 25, 25] == UNCERTAIN_LABEL
    assert np.any(result.labels == 3)
    assert np.any(result.labels == 9)
    assert np.any(result.labels == 11)
    assert result.summary["uncertain_voxels"] > 0
    assert np.all(result.labels[~lumen] == 0)


def test_anisotropic_affine_is_used_for_radius_compatibility():
    lumen = np.zeros((11, 11, 11), dtype=bool)
    lumen[4:7, 4:7, 1:10] = True
    affine = np.diag([0.5, 0.5, 2.0, 1.0])
    points_ijk = np.column_stack(
        [np.full(9, 5.0), np.full(9, 5.0), np.arange(1.0, 10.0)]
    )
    points_ras = (
        np.column_stack([points_ijk, np.ones(len(points_ijk))]) @ affine.T
    )[:, :3]
    branch = CenterlineSeed(
        "lica", 10, "LICA", points_ras, np.full(len(points_ras), 1.5)
    )
    labels, _ = nearest_centerline_labels(lumen, affine, [branch])
    assert np.all(labels[lumen] == 10)


def test_invalid_label_and_singular_affine_fail_clearly():
    lumen = np.ones((3, 3, 3), dtype=bool)
    invalid = CenterlineSeed("x", 14, "uncertain", np.array([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(ValueError, match="1..13"):
        nearest_centerline_labels(lumen, np.eye(4), [invalid])
    valid = CenterlineSeed("x", 3, "RCCA", np.array([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(ValueError, match="singular"):
        nearest_centerline_labels(lumen, np.zeros((4, 4)), [valid])
