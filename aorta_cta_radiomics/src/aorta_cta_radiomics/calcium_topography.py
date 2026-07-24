"""Luminal-side vs adventitial-side aortic wall calcium topography.

Aortic wall calcium is not anatomically uniform: intimal (luminal-side)
calcification tracks atherosclerotic/plaque-type disease, while medial or
adventitial (outer-wall-side) calcification tracks a degenerative
(Moenckeberg-type) process with a different risk profile. CTA cannot resolve
histologic wall layers directly, so this module builds two reference surfaces
from the segmented ``aorta_mask`` itself: the outer wall boundary (the mask's
own edge) and an inner lumen reference set ``wall_thickness_mm`` in from that
edge (default 2.0mm, matching normal aortic wall thickness and this
pipeline's existing ``aorta_wall_internal_mm`` convention). Each connected
calcium component is then scored by its position between those two surfaces.

Note this intentionally does *not* reuse the pipeline's
``lumen_core_mask``/``lumen_core_distance_mm`` (default 5mm): that mask is
built deep enough to guarantee a robust per-slice contrast-HU sample for
dynamic thresholding, not to mark the true lumen-wall interface. Using it here
would push most real wall calcium (which sits within ~2mm of the outer
boundary) toward "adventitial" by construction, regardless of true position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .calcification import density_factor_for_hu
from .features import feature_row
from .shells import _crop_around_mask, internal_boundary_shell


COMPONENT_TABLE_COLUMNS = [
    "component_id",
    "voxel_count",
    "volume_mm3",
    "max_hu",
    "mean_hu",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "dist_to_lumen_mm",
    "dist_to_outer_wall_mm",
    "relative_position",
    "classification",
]


@dataclass(frozen=True)
class CalciumTopographyResult:
    """Voxel masks and per-component detail for a luminal/adventitial split."""

    luminal_mask: np.ndarray
    adventitial_mask: np.ndarray
    indeterminate_mask: np.ndarray
    component_table: pd.DataFrame


def classify_calcium_topography(
    calcium_mask: np.ndarray,
    aorta_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    image: np.ndarray | None = None,
    wall_thickness_mm: float = 2.0,
    luminal_max_ratio: float = 0.4,
    adventitial_min_ratio: float = 0.6,
    crop_margin_mm: float = 5.0,
) -> CalciumTopographyResult:
    """Split connected calcium components into luminal-side vs adventitial-side.

    For each 26-connected calcium component this computes the minimum distance
    to the lumen reference surface (``dist_to_lumen_mm``, the aorta mask minus
    an inward ``wall_thickness_mm`` shell) and to the outer aorta mask boundary
    (``dist_to_outer_wall_mm``), then scores the component with
    ``relative_position = dist_to_lumen / (dist_to_lumen + dist_to_outer_wall)``.
    A value near 0 means the component sits against the contrast-filled lumen
    (luminal/intimal-side); a value near 1 means it sits against the outer
    wall/periaortic interface (adventitial-side). Because the aortic wall is
    often only a few voxels thick at typical CTA resolution, components with a
    score in between the two ratio cutoffs are labeled "indeterminate" rather
    than forced into either side.
    """
    calcium = np.asarray(calcium_mask, dtype=bool)
    aorta = np.asarray(aorta_mask, dtype=bool)
    if calcium.shape != aorta.shape:
        raise ValueError("calcium_mask and aorta_mask must have the same shape.")

    empty_table = pd.DataFrame(columns=COMPONENT_TABLE_COLUMNS)
    luminal = np.zeros_like(calcium, dtype=bool)
    adventitial = np.zeros_like(calcium, dtype=bool)
    indeterminate = np.zeros_like(calcium, dtype=bool)
    if not calcium.any():
        return CalciumTopographyResult(luminal, adventitial, indeterminate, empty_table)

    from scipy import ndimage as ndi

    sampling_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    voxel_volume_mm3 = float(np.prod(spacing_xyz))

    # Crop tightly around the aorta + calcium so the distance transforms below
    # run over a small region instead of the full CT volume (which can be
    # several hundred million voxels for a chest/abdomen CTA).
    _, slices = _crop_around_mask(aorta | calcium, spacing_xyz, margin_mm=float(crop_margin_mm))
    offset = np.array([s.start for s in slices], dtype=float)

    calcium_c = calcium[slices]
    aorta_c = aorta[slices]
    image_c = np.asarray(image)[slices] if image is not None else None

    wall_zone_c = internal_boundary_shell(aorta_c, spacing_xyz, depth_mm=float(wall_thickness_mm))
    lumen_reference_c = aorta_c & ~wall_zone_c

    dist_to_lumen_c = (
        ndi.distance_transform_edt(~lumen_reference_c, sampling=sampling_zyx)
        if lumen_reference_c.any()
        else np.full(calcium_c.shape, np.inf, dtype=float)
    )
    # Distance from voxels inside the aorta mask to the nearest voxel outside
    # it; 0 for voxels already at/outside the outer wall boundary.
    dist_to_outer_wall_c = ndi.distance_transform_edt(aorta_c, sampling=sampling_zyx)
    dist_to_outer_wall_c = np.where(aorta_c, dist_to_outer_wall_c, 0.0)

    structure = np.ones((3, 3, 3), dtype=bool)
    labels_c, num_components = ndi.label(calcium_c, structure=structure)

    luminal_c = np.zeros_like(calcium_c, dtype=bool)
    adventitial_c = np.zeros_like(calcium_c, dtype=bool)
    indeterminate_c = np.zeros_like(calcium_c, dtype=bool)

    rows: list[dict[str, object]] = []
    for component_id in range(1, int(num_components) + 1):
        component = labels_c == component_id
        voxel_count = int(component.sum())
        d_lumen = float(dist_to_lumen_c[component].min())
        d_outer = float(dist_to_outer_wall_c[component].min())
        denom = d_lumen + d_outer
        relative_position = float(d_lumen / denom) if denom > 0 else 0.5

        if relative_position <= luminal_max_ratio:
            classification = "luminal"
            luminal_c |= component
        elif relative_position >= adventitial_min_ratio:
            classification = "adventitial"
            adventitial_c |= component
        else:
            classification = "indeterminate"
            indeterminate_c |= component

        centroid = np.asarray(ndi.center_of_mass(component)) + offset
        max_hu = float(image_c[component].max()) if image_c is not None else float("nan")
        mean_hu = float(image_c[component].mean()) if image_c is not None else float("nan")
        rows.append(
            {
                "component_id": int(component_id),
                "voxel_count": voxel_count,
                "volume_mm3": float(voxel_count * voxel_volume_mm3),
                "max_hu": max_hu,
                "mean_hu": mean_hu,
                "centroid_z": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_x": float(centroid[2]),
                "dist_to_lumen_mm": d_lumen,
                "dist_to_outer_wall_mm": d_outer,
                "relative_position": relative_position,
                "classification": classification,
            }
        )

    luminal[slices] = luminal_c
    adventitial[slices] = adventitial_c
    indeterminate[slices] = indeterminate_c
    component_table = pd.DataFrame(rows, columns=COMPONENT_TABLE_COLUMNS)
    return CalciumTopographyResult(luminal, adventitial, indeterminate, component_table)


def summarize_calcium_topography(
    image: np.ndarray,
    result: CalciumTopographyResult,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    mask_name: str,
    threshold_label: str,
    software_version: str = __version__,
    dominance_margin_fraction: float = 0.2,
) -> pd.DataFrame:
    """Summarize per-side calcium burden plus a case-level dominance category.

    ``dominant_topography`` is a convenience two/three-group label for cohort
    analyses: "luminal_dominant" or "adventitial_dominant" when one side holds
    a clear majority of calcium volume (by default >=60/40), "mixed" when the
    two sides are comparable, and "none"/"indeterminate_only" for cases with
    no confidently-sided calcium.
    """
    image_array = np.asarray(image)
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    groups = {
        "luminal": result.luminal_mask,
        "adventitial": result.adventitial_mask,
        "indeterminate": result.indeterminate_mask,
    }
    total_volume_mm3 = float(sum(int(mask.sum()) for mask in groups.values()) * voxel_volume_mm3)

    rows: list[dict[str, object]] = []
    volume_by_group: dict[str, float] = {}
    for name, mask in groups.items():
        voxel_count = int(mask.sum())
        volume_mm3 = float(voxel_count * voxel_volume_mm3)
        volume_by_group[name] = volume_mm3
        values = image_array[mask]
        max_hu = float(values.max()) if values.size else float("nan")
        mean_hu = float(values.mean()) if values.size else float("nan")
        mass_proxy = float(values.sum() * voxel_volume_mm3) if values.size else 0.0
        agatston_like = float(volume_mm3 * density_factor_for_hu(max_hu))
        num_components = (
            int((result.component_table["classification"] == name).sum())
            if not result.component_table.empty
            else 0
        )
        rows.extend(
            [
                _row(case_id, name, "voxel_count", voxel_count, "voxels", threshold_label, mask_name, software_version),
                _row(case_id, name, "volume_mm3", volume_mm3, "mm3", threshold_label, mask_name, software_version),
                _row(case_id, name, "num_components", num_components, "components", threshold_label, mask_name, software_version),
                _row(case_id, name, "max_hu", max_hu, "HU", threshold_label, mask_name, software_version),
                _row(case_id, name, "mean_hu", mean_hu, "HU", threshold_label, mask_name, software_version),
                _row(case_id, name, "mass_proxy", mass_proxy, "HU*mm3", threshold_label, mask_name, software_version),
                _row(case_id, name, "agatston_like_not_ecg_gated", agatston_like, "arbitrary", threshold_label, mask_name, software_version),
                _row(case_id, name, "fraction_of_total_calcium", _safe_divide(volume_mm3, total_volume_mm3), "fraction", threshold_label, mask_name, software_version),
            ]
        )

    luminal_volume = volume_by_group["luminal"]
    adventitial_volume = volume_by_group["adventitial"]
    sided_total = luminal_volume + adventitial_volume
    ratio = (
        float(luminal_volume / adventitial_volume)
        if adventitial_volume > 0
        else (float("inf") if luminal_volume > 0 else float("nan"))
    )
    if total_volume_mm3 <= 0:
        dominant = "none"
    elif sided_total <= 0:
        dominant = "indeterminate_only"
    else:
        balance = (luminal_volume - adventitial_volume) / sided_total
        if balance >= dominance_margin_fraction:
            dominant = "luminal_dominant"
        elif balance <= -dominance_margin_fraction:
            dominant = "adventitial_dominant"
        else:
            dominant = "mixed"

    rows.extend(
        [
            _row(case_id, "aorta", "total_volume_mm3", total_volume_mm3, "mm3", threshold_label, mask_name, software_version),
            _row(case_id, "aorta", "luminal_to_adventitial_volume_ratio", ratio, "ratio", threshold_label, mask_name, software_version),
            _row(case_id, "aorta", "dominant_topography", dominant, "category", threshold_label, mask_name, software_version),
        ]
    )
    return pd.DataFrame(rows)


def _row(
    case_id: str,
    region: str,
    feature_name: str,
    value: object,
    units: str,
    threshold_label: str,
    mask_name: str,
    software_version: str,
) -> dict[str, object]:
    return feature_row(
        case_id=case_id,
        region=region,
        feature_group="calcium_topography",
        feature_name=feature_name,
        feature_value=value,
        units=units,
        threshold_if_applicable=threshold_label,
        mask_name=mask_name,
        software_version=software_version,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator) / denominator
