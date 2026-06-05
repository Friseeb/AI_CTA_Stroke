import numpy as np

from aorta_cta_radiomics.shells import combined_periaortic_shell, create_aorta_wall_band_masks, external_shell


def test_external_shell_uses_physical_distance_not_voxel_count():
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[3, 3, 3] = True
    spacing_xyz = (2.0, 1.0, 3.0)

    shell_0_2 = external_shell(mask, spacing_xyz, inner_mm=0.0, outer_mm=2.0)
    assert shell_0_2[3, 3, 4]  # x neighbor is 2 mm away
    assert shell_0_2[3, 4, 3]  # y neighbor is 1 mm away
    assert not shell_0_2[4, 3, 3]  # z neighbor is 3 mm away

    shell_2_5 = external_shell(mask, spacing_xyz, inner_mm=2.0, outer_mm=5.0)
    assert shell_2_5[4, 3, 3]  # 3 mm in z
    assert shell_2_5[3, 3, 5]  # 4 mm in x
    assert not shell_2_5[3, 4, 3]  # 1 mm in y


def test_combined_shell_includes_inner_boundary_and_external_shell():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[3:6, 3:6, 3:6] = True
    shell = combined_periaortic_shell(mask, (1.0, 1.0, 1.0), outer_mm=1.0, internal_mm=1.0)

    assert shell[2, 4, 4]  # external neighbor
    assert shell[3, 4, 4]  # internal boundary
    assert not shell[4, 4, 4]  # center is deeper than 1 mm from boundary


def test_aorta_wall_band_excludes_central_lumen_core():
    mask = np.zeros((11, 11, 11), dtype=bool)
    mask[3:8, 3:8, 3:8] = True

    wall_masks = create_aorta_wall_band_masks(mask, (1.0, 1.0, 1.0), internal_mm=1.0, external_mm=1.0)

    assert wall_masks["aorta_wall_internal"][3, 5, 5]
    assert wall_masks["aorta_wall_external"][2, 5, 5]
    assert wall_masks["aorta_wall_band"][3, 5, 5]
    assert wall_masks["aorta_wall_band"][2, 5, 5]
    assert not wall_masks["aorta_wall_band"][5, 5, 5]
