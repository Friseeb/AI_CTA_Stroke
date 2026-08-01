import numpy as np

from aorta_cta_radiomics.perivascular.exclusions import bifurcation_exclusion_mask
from aorta_cta_radiomics.perivascular.shells import (
    AdaptiveShellLaw,
    RadialBand,
    _adaptive_padded_mask,
    generate_adaptive_shell,
    generate_radial_shells,
)


def test_physical_shell_bands_respect_anisotropic_spacing_and_valid_denominator():
    lumen = np.zeros((11, 11, 11), dtype=bool)
    lumen[5, 5, 5] = True
    result = generate_radial_shells(
        lumen,
        (2.0, 1.0, 3.0),
        vessel_id="right_ica",
        bands=(RadialBand(0.0, 2.0), RadialBand(2.0, 5.0)),
    )
    inner = result.bands["shell_0_2mm"]
    outer = result.bands["shell_2_5mm"]
    assert inner.raw_mask[5, 5, 6]  # x neighbor = 2 mm
    assert inner.raw_mask[5, 6, 5]  # y neighbor = 1 mm
    assert not inner.raw_mask[6, 5, 5]  # z neighbor = 3 mm
    assert outer.raw_mask[6, 5, 5]
    assert np.array_equal(inner.filtered_mask, inner.valid_denominator_mask)
    assert inner.valid_volume_mm3 == inner.valid_voxels * 6.0
    assert not inner.is_boundary_truncated


def test_straight_digital_cylinder_has_expected_radius_shell_thickness_and_volume():
    shape = (71, 41, 41)
    zz, yy, xx = np.indices(shape)
    radius_mm = 4.0
    lumen = ((xx - 20) ** 2 + (yy - 20) ** 2 <= radius_mm**2) & (zz >= 10) & (zz <= 60)
    band = generate_radial_shells(
        lumen,
        (1, 1, 1),
        vessel_id="synthetic_straight_cylinder",
        bands=(RadialBand(0, 2),),
    ).bands["shell_0_2mm"]

    assert band.raw_mask[35, 20, 26]  # radius 4 mm + 2 mm shell
    assert not band.raw_mask[35, 20, 27]
    # Digital EDT/voxel-face geometry approximates the analytic lateral shell;
    # retain a tolerance for lattice and cylinder-end effects.
    analytic_lateral_volume = np.pi * ((radius_mm + 2.0) ** 2 - radius_mm**2) * 51.0
    assert np.isclose(band.raw_volume_mm3, analytic_lateral_volume, rtol=0.15)


def test_bifurcation_and_adjacent_artery_are_removed_but_raw_shell_is_preserved():
    shape = (31, 31, 31)
    zz, yy, xx = np.indices(shape)
    lumen = ((xx - 15) ** 2 + (yy - 15) ** 2 <= 2**2) & (zz >= 4) & (zz <= 26)
    other = ((zz - 15) ** 2 + (yy - 15) ** 2 <= 2**2) & (xx >= 4) & (xx <= 26)
    branch = bifurcation_exclusion_mask(shape, np.array([[15.0, 15.0, 15.0]]), (1, 1, 1), 3.0)

    result = generate_radial_shells(
        lumen,
        (1, 1, 1),
        vessel_id="left_cca",
        bands=(RadialBand(0, 3),),
        other_artery_mask=other,
        bifurcation_exclusion_mask=branch,
    ).bands["shell_0_3mm"]

    assert np.any(result.raw_mask & other)
    assert not np.any(result.filtered_mask & other)
    assert not np.any(result.filtered_mask & branch)
    assert "shell_overlaps_other_artery" in result.flags
    assert "shell_intersects_bifurcation" in result.flags
    assert result.valid_voxels < result.raw_voxels


def test_boundary_truncation_uses_observed_valid_denominator_not_expected_shell():
    lumen = np.zeros((15, 15, 15), dtype=bool)
    lumen[3:12, 5:10, 0:2] = True
    band = generate_radial_shells(
        lumen,
        (1, 1, 1),
        vessel_id="boundary_case",
        bands=(RadialBand(0, 5),),
    ).bands["shell_0_5mm"]

    assert band.is_boundary_truncated
    assert 0 < band.boundary_truncation_fraction < 1
    assert band.expected_complete_voxels > band.raw_voxels
    assert band.valid_voxels == int(band.valid_denominator_mask.sum())
    assert "shell_truncated_by_image_boundary" in band.flags


def test_capped_adaptive_shell_law_has_minimum_and_maximum_and_drives_shell_extent():
    law = AdaptiveShellLaw(alpha=0.75, minimum_mm=1.0, maximum_mm=4.0)
    assert law.extent_mm(0.5) == 1.0
    assert law.extent_mm(4.0) == 3.0
    assert law.extent_mm(20.0) == 4.0

    lumen = np.zeros((21, 21, 21), dtype=bool)
    lumen[10, 10, 10] = True
    result = generate_adaptive_shell(
        lumen,
        (1, 1, 1),
        diameter_mm=4.0,
        vessel_id="left_ica",
        law=law,
    )
    adaptive = result.bands["adaptive_shell"].raw_mask
    assert adaptive[10, 10, 13]  # 3 mm
    assert not adaptive[10, 10, 14]  # > locally capped alpha * diameter
    assert result.parameters["adaptive_law"]["formula"].startswith("clip")


def test_adaptive_shell_local_edt_matches_full_grid_reference_at_image_boundary(monkeypatch):
    from scipy import ndimage as ndi

    shape = (61, 63, 65)
    lumen = np.zeros(shape, dtype=bool)
    lumen[27:34, 29:36, 0:3] = True
    diameter = np.zeros(shape, dtype=np.float32)
    diameter[lumen] = np.linspace(2.0, 8.0, int(lumen.sum()), dtype=np.float32)
    spacing = (0.8, 1.1, 1.7)
    law = AdaptiveShellLaw(alpha=0.75, minimum_mm=1.7, maximum_mm=5.0)

    full_padded, _, full_crop, _ = _adaptive_padded_mask(lumen, spacing, diameter, law)
    expected_raw = full_padded[full_crop]
    expected_complete_voxels = int(full_padded.sum())

    original_edt = ndi.distance_transform_edt
    edt_shapes: list[tuple[int, ...]] = []

    def tracking_edt(array, *args, **kwargs):
        edt_shapes.append(tuple(array.shape))
        return original_edt(array, *args, **kwargs)

    monkeypatch.setattr(ndi, "distance_transform_edt", tracking_edt)
    result = generate_adaptive_shell(
        lumen,
        spacing,
        diameter,
        vessel_id="boundary_ica",
        law=law,
    ).bands["adaptive_shell"]

    assert np.array_equal(result.raw_mask, expected_raw)
    assert result.expected_complete_voxels == expected_complete_voxels
    assert result.boundary_missing_voxels > 0
    assert edt_shapes
    assert all(np.prod(call_shape) < 0.20 * np.prod(shape) for call_shape in edt_shapes)
