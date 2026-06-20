"""Physical-distance peri-aortic shell generation.

This module now re-exports the shared implementation from ``cta_common.shells``
(install with ``pip install -e cta_common``). It is kept as a stable import path
so existing ``from .shells import ...`` call sites — including the private
helpers used by ``fat_wall``/``fat_omics`` — keep working.
"""

from __future__ import annotations

from cta_common.shells import (  # noqa: F401
    combined_periaortic_shell,
    create_aorta_wall_band_masks,
    create_base_shells,
    external_shell,
    internal_boundary_shell,
    local_shell_around_mask,
    _brute_force_distance_to_false,
    _crop_around_mask,
    _distance_transform_edt,
    _sampling_zyx,
)

__all__ = [
    "combined_periaortic_shell",
    "create_aorta_wall_band_masks",
    "create_base_shells",
    "external_shell",
    "internal_boundary_shell",
    "local_shell_around_mask",
]
