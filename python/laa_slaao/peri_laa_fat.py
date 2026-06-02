"""Peri-LAA fat envelope features.

Analogous to peri-coronary / epicardial adipose tissue analysis. The pipeline
is **two-step** so each stage can be reviewed in isolation:

  1. **Peri-LAA ROI** — `build_peri_laa_roi`: the anatomical envelope around
     the LAA where peri-LAA fat could plausibly live. Built from:
        * radial distance ≤ `roi_max_distance_mm` from the LAA surface
        * minus the LAA itself
        * minus the SLAAO `exclusion_mask` (negative prior: aorta / lungs /
          myocardium / coronaries / vertebrae / pericardium)
        * minus a partial-volume buffer around pure-air (HU < air_proximity_hu)
          and pure-contrast (HU > vessel_proximity_hu) voxels
        * optionally ∩ pericardium mask (caller-supplied)
     The ROI is saved as its own NIfTI so annotators can verify the geometry
     BEFORE inspecting fat content. This is the right unit of QC.

  2. **Fat within ROI** — `compute_peri_laa_fat`: partition the ROI into
     radial shells (0–2, 2–5, 5–10 mm by default) and report fat HU stats
     within each shell. The HU window only affects fat selection; it never
     changes the ROI geometry.

This split makes "the QC looked wrong" debuggable: if the ROI is wrong,
fix the exclusion/buffer; if the ROI is right but fat is wrong, fix the HU
window. They are no longer entangled in one boolean expression.

This module is intentionally narrow:
  * it does NOT segment the LAA — it consumes any LAA mask (consensus, expert,
    a single prior, etc.);
  * it does NOT learn an inflammation model — the fat HU window is fixed
    at the standard adipose range (−190..−30 HU);
  * it accepts an optional exclusion mask (typically the SLAAO negative prior:
    aorta / lungs / myocardium / coronaries / vertebrae / pericardium) so the
    shell does not leak into adjacent structures.

The output mask is multi-label (0 = background, 1..N = shells in radial
order) so it can be loaded into 3D Slicer as a single Segmentation node with
one named/colored segment per shell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_fill_holes,
    distance_transform_edt, generate_binary_structure, label as cc_label,
)


_NAN = float("nan")


def _shell_prefix(lo_mm: float, hi_mm: float) -> str:
    """Stable, filename-safe shell key.

    `0.5` becomes `0p5`, `10` stays `10`. We replace only the decimal point,
    never the file extension — using str(float) directly produced names like
    `..._shell0_2pniipgz` because the bare `.replace(".", "p")` ate the
    `.nii.gz` separator.
    """
    def fmt(x: float) -> str:
        return f"{x:g}".replace(".", "p")
    return f"peri_laa_fat_shell{fmt(lo_mm)}_{fmt(hi_mm)}"


# Default radial shells (inner_mm, outer_mm). Anything outside the LAA at the
# given mm-band is included; HU filtering happens after the shell is built.
DEFAULT_SHELLS_MM: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
)


@dataclass
class PeriLAAROI:
    """Result of step 1 — the peri-LAA anatomical ROI.

    Attributes:
        roi_mask:   bool array, True = voxel is inside the peri-LAA region.
        dist_mm:    float array, physical distance (mm) from each voxel to
                    the nearest LAA surface voxel. Useful downstream for
                    arbitrary shell partitioning without recomputing the EDT.
        provenance: dict with which inputs were applied (negative prior on/off,
                    pericardium intersection on/off, PV-buffer params, etc.)
                    so the ROI's construction is reproducible from outputs.
    """
    roi_mask: np.ndarray
    dist_mm: np.ndarray
    provenance: dict[str, float | str | int | bool]


@dataclass
class PeriLAAFatResult:
    """Result of step 2 — fat HU statistics partitioned by shell.

    Attributes:
        label_mask:  int16, 0 = background, 1..N = shell-band fat voxels.
        shells_mm:   list of (inner_mm, outer_mm) used (echoes the input).
        features:    flat metrics dict keyed `peri_laa_fat_shell{lo}_{hi}_*`
                     plus aggregate `peri_laa_fat_total_*`.
        roi:         the PeriLAAROI used (so callers can persist it alongside).
    """
    label_mask: np.ndarray
    shells_mm: list[tuple[float, float]]
    features: dict[str, float | str | int]
    roi: "PeriLAAROI"

    @property
    def exclusion_used(self) -> int:
        """Back-compat shim — legacy callers asked for this on the result.
        Total voxels carved off the maximal-shell geometry by every
        exclusion mechanism combined."""
        return int(self.roi.provenance.get("exclusion_voxels_total", 0))


@dataclass
class LAACenterline:
    """Centerline analysis of the LAA used to steer the extension.

    Attributes:
        ordered_points_zyx: ordered skeleton points (axis-0..2 indices),
            walked from proximal (LA-orifice) toward distal (apex).
        bend_index: index into `ordered_points_zyx` of the bend (point of
            maximum curvature on the centerline). May be None if no clear
            bend was detected.
        post_bend_tangent_zyx: unit vector in physical mm space pointing
            distally from the bend along the post-bend centerline segment.
            None if no centerline could be computed.
        bend_point_phys_mm: physical-mm position of the bend (axis order
            matches the input arrays). None if no bend was detected.
        bend_angle_deg: angular change at the bend, in degrees.
        notes: short string describing how the centerline was obtained
            (e.g. "skeleton_2_endpoints", "pca_fallback").
    """
    ordered_points_zyx: Optional[np.ndarray]
    bend_index: Optional[int]
    post_bend_tangent_zyx: Optional[np.ndarray]
    bend_point_phys_mm: Optional[np.ndarray]
    bend_angle_deg: float
    notes: str


def compute_laa_centerline_and_bend(
    laa_mask: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    la_body_mask: Optional[np.ndarray] = None,
    smoothing_window: int = 5,
    perifat_mask: Optional[np.ndarray] = None,
    lateral_axis_zyx: Optional[np.ndarray] = None,
) -> LAACenterline:
    """Skeletonise the LAA, walk endpoint-to-endpoint, and locate the bend.

    Algorithm:
        1. 3D skeleton of `laa_mask`.
        2. Find skeleton endpoints (skeleton voxels with exactly 1 skeleton
           neighbour in a 3×3×3 window).
        3. Pick the two endpoints whose physical-space separation is largest.
           Orient them: proximal = closer to `la_body_mask` centroid; distal =
           farther. If no LA-body mask is supplied we use the endpoint whose
           PCA-projection is most negative as proximal.
        4. Build an ordered path from proximal to distal by greedy nearest
           neighbour walking on the skeleton.
        5. Compute tangents along the path in PHYSICAL mm (anisotropy-safe),
           smooth with a running mean of `smoothing_window` points.
        6. Bend index = position along the path with maximum |angular change|
           between consecutive smoothed tangents. Bend angle = the value of
           that maximum, in degrees.
        7. Post-bend tangent = mean of the smoothed tangents on the distal
           side of the bend.

    Falls back to PCA on the LAA voxels when skeletonization yields fewer
    than 2 endpoints (rare; happens on tiny or pathologically-thick LAAs).
    """
    from skimage.morphology import skeletonize

    laa = laa_mask.astype(bool)
    spacing = np.array([float(v) for v in spacing_xyz_mm])

    def _orient_toward_perifat(tangent: np.ndarray, anchor_phys: np.ndarray) -> np.ndarray:
        """Flip `tangent` if it points away from the LEFT-perifat centroid.

        Anatomical reality: peri-LAA fat exists on the anterolateral
        ("left-of-LAA-relative-to-aorta") side of the LAA. The medial
        (aorta) side has little fat. When a `lateral_axis_zyx` vector
        is supplied (= LAA_centroid − aorta_centroid, unit-norm), the
        perifat is first filtered to the LEFT half-space — that subset's
        centroid is the orientation reference. Without a lateral axis we
        fall back to the full perifat centroid.
        """
        if perifat_mask is None or not perifat_mask.any():
            return tangent
        pf_phys = np.argwhere(perifat_mask) * spacing
        laa_phys_local = np.argwhere(laa) * spacing
        laa_center = laa_phys_local.mean(axis=0)
        laa_extent = np.linalg.norm(laa_phys_local.max(axis=0) - laa_phys_local.min(axis=0))
        d = np.linalg.norm(pf_phys - laa_center, axis=1)
        near = d < (laa_extent * 1.5 if laa_extent > 0 else d.max() + 1)
        if not near.any():
            return tangent
        pf_subset = pf_phys[near]
        # LEFT-filter: when a lateral axis is provided, keep only perifat
        # voxels with positive projection onto it (= LAA side, away from
        # aorta). If that subset is too small (<20% of nearby perifat), fall
        # back to the full nearby perifat so we don't trust noise.
        if lateral_axis_zyx is not None:
            offsets = pf_subset - laa_center
            proj = offsets @ lateral_axis_zyx
            left_subset = pf_subset[proj > 0]
            if left_subset.shape[0] > max(20, int(0.2 * pf_subset.shape[0])):
                pf_subset = left_subset
        pf_center = pf_subset.mean(axis=0)
        toward_pf = pf_center - anchor_phys
        n = np.linalg.norm(toward_pf)
        if n < 1e-6:
            return tangent
        toward_pf /= n
        if np.dot(tangent, toward_pf) < 0:
            return -tangent
        return tangent

    def _pca_fallback() -> LAACenterline:
        coords = np.argwhere(laa).astype(np.float64)
        if coords.shape[0] < 4:
            return LAACenterline(None, None, None, None, _NAN, "empty_or_tiny_laa")
        phys = coords * spacing
        center = phys.mean(axis=0)
        u, s, vh = np.linalg.svd(phys - center, full_matrices=False)
        axis = vh[0]
        if la_body_mask is not None and la_body_mask.any():
            la_phys = np.argwhere(la_body_mask) * spacing
            away = center - la_phys.mean(axis=0)
            if np.dot(axis, away) < 0:
                axis = -axis
        # Final orientation override: must point toward where the perifat is.
        axis = _orient_toward_perifat(axis, center)
        return LAACenterline(
            ordered_points_zyx=None,
            bend_index=None,
            post_bend_tangent_zyx=axis,
            bend_point_phys_mm=center,
            bend_angle_deg=_NAN,
            notes="pca_fallback_perifat_oriented" if perifat_mask is not None else "pca_fallback",
        )

    if not laa.any():
        return LAACenterline(None, None, None, None, _NAN, "empty_laa")

    skel = skeletonize(laa)
    if skel.sum() < 4:
        return _pca_fallback()

    # Find skeleton endpoints (voxels with exactly 1 skeleton neighbour in
    # a 3×3×3 window — i.e. degree-1 nodes of the skeleton graph).
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    skel_i = skel.astype(np.int32)
    deg = convolve(skel_i, kernel, mode="constant", cval=0) - skel_i
    endpoints = np.argwhere(skel & (deg == 1))
    if len(endpoints) < 2:
        return _pca_fallback()

    # Pick the two endpoints farthest apart in physical space.
    phys_ep = endpoints * spacing
    diffs = phys_ep[:, None, :] - phys_ep[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    proximal_vox = endpoints[i].copy()
    distal_vox = endpoints[j].copy()

    # Orient: proximal = closer to LA body
    if la_body_mask is not None and la_body_mask.any():
        la_phys = np.argwhere(la_body_mask) * spacing
        la_centroid = la_phys.mean(axis=0)
        if np.linalg.norm(phys_ep[i] - la_centroid) > np.linalg.norm(phys_ep[j] - la_centroid):
            proximal_vox, distal_vox = distal_vox, proximal_vox

    # Greedy nearest-neighbour walk from proximal to distal across the
    # skeleton voxels. Bounded to skeleton voxels only.
    skel_set = {tuple(v) for v in np.argwhere(skel)}
    path: list[tuple[int, int, int]] = [tuple(int(x) for x in proximal_vox)]
    current = path[0]
    visited = {current}
    while current != tuple(int(x) for x in distal_vox):
        # 26-neighbours of current within the skeleton
        z, y, x = current
        next_candidates: list[tuple[int, int, int]] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    n = (z + dz, y + dy, x + dx)
                    if n in skel_set and n not in visited:
                        next_candidates.append(n)
        if not next_candidates:
            break
        # Pick the candidate closest to the distal endpoint
        next_phys = np.array(next_candidates, dtype=np.float64) * spacing
        d_to_distal = np.linalg.norm(next_phys - phys_ep[j], axis=1)
        chosen = next_candidates[int(np.argmin(d_to_distal))]
        path.append(chosen)
        visited.add(chosen)
        current = chosen

    if len(path) < 4:
        return _pca_fallback()

    ordered = np.array(path, dtype=np.int64)         # (N, 3) in voxel index
    ordered_phys = ordered * spacing                  # (N, 3) in mm

    # Tangents in physical mm; smooth using a moving window.
    raw_tangents = np.diff(ordered_phys, axis=0)
    norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_tangents = raw_tangents / norms

    win = max(1, int(smoothing_window))
    if unit_tangents.shape[0] >= win:
        kernel_1d = np.ones(win) / win
        # Smooth each component independently
        smoothed = np.stack([
            np.convolve(unit_tangents[:, c], kernel_1d, mode="valid")
            for c in range(3)
        ], axis=1)
    else:
        smoothed = unit_tangents

    # Bend = max |angular change| between consecutive smoothed tangents.
    if smoothed.shape[0] < 2:
        # Centerline too short to bend — fall back to last-segment tangent.
        post_tangent = unit_tangents.mean(axis=0)
        post_tangent /= max(np.linalg.norm(post_tangent), 1e-9)
        bend_idx = None
        bend_angle = 0.0
        bend_point = ordered_phys[len(ordered_phys) // 2]
    else:
        dots = np.einsum("ij,ij->i", smoothed[:-1], smoothed[1:])
        dots = np.clip(dots, -1.0, 1.0)
        angles_rad = np.arccos(dots)
        bend_local = int(np.argmax(angles_rad))
        bend_angle = float(np.degrees(angles_rad[bend_local]))
        # `bend_local` indexes into `smoothed`, which lost (win-1) from each
        # end of `unit_tangents`. Map back to the centerline-path index.
        bend_idx = bend_local + (win // 2) + 1
        bend_idx = min(max(bend_idx, 0), len(ordered_phys) - 1)
        bend_point = ordered_phys[bend_idx]
        # Post-bend tangent = mean of distal-side smoothed tangents
        post_segment = smoothed[bend_local + 1:]
        if post_segment.shape[0] == 0:
            post_segment = smoothed[bend_local:]
        post_tangent = post_segment.mean(axis=0)
        post_tangent /= max(np.linalg.norm(post_tangent), 1e-9)

    # Perifat-based orientation override: even with a correct LA-body
    # proximal/distal labelling, the skeleton's "farthest endpoint" pick
    # can be wrong for chicken-wing LAAs that bend back toward the LA root.
    # The perifat centroid is the most reliable anatomical signal for
    # "which way the LAA tip points" because peri-LAA fat surrounds the tip.
    pre_pf_tangent = post_tangent.copy()
    post_tangent = _orient_toward_perifat(post_tangent, bend_point)
    perifat_flipped = bool(np.dot(pre_pf_tangent, post_tangent) < 0)

    return LAACenterline(
        ordered_points_zyx=ordered,
        bend_index=bend_idx,
        post_bend_tangent_zyx=post_tangent,
        bend_point_phys_mm=bend_point,
        bend_angle_deg=bend_angle,
        notes=(
            f"skeleton_path_len={len(ordered)}_endpoints=2"
            + ("_perifat_flipped" if perifat_flipped else "")
        ),
    )


@dataclass
class ExtendedLAAResult:
    """Result of step 3 — LAA grown outward to the peri-LAA fat envelope.

    Attributes:
        extended_laa:    bool array. Original LAA voxels + all ROI voxels
                         not classified as peri-LAA fat. Captures the LAA
                         lumen, the LAA wall (myocardium), and any thrombus /
                         filling defect lying between the lumen and the
                         epicardial-fat layer.
        perifat_combined: bool array. Union of every fat shell — the outside
                          layer that bounds the extension.
        features:        deltas (voxel counts, mL, HU) between the original
                         LAA and the extended LAA. Lets downstream code track
                         how much volume the extension added per case.
    """
    extended_laa: np.ndarray
    perifat_combined: np.ndarray
    features: dict[str, float | str | int]


def build_peri_laa_roi(
    ct_array: np.ndarray,
    laa_mask: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    roi_max_distance_mm: float = 10.0,
    exclusion_mask: Optional[np.ndarray] = None,
    pericardium_mask: Optional[np.ndarray] = None,
    air_proximity_hu: float = -300.0,
    vessel_proximity_hu: float = 100.0,
    pv_buffer_mm: float = 1.0,
) -> PeriLAAROI:
    """Step 1: build the anatomical peri-LAA region of interest.

    Args:
        ct_array: HU array. Axis order must match `spacing_xyz_mm`.
        laa_mask: bool/int array of the LAA cavity in the same geometry.
        spacing_xyz_mm: physical mm per axis, in the same order as the arrays.
        roi_max_distance_mm: outermost radial distance from the LAA surface
            included in the ROI. Defaults to 10 mm, matching the outermost
            shell of `DEFAULT_SHELLS_MM`.
        exclusion_mask: SLAAO negative prior — aorta / lungs / myocardium /
            coronaries / vertebrae / pericardium / etc. Subtracted from the
            ROI.
        pericardium_mask: optional pericardium silhouette. If supplied, the
            ROI is *intersected* with it, so the result lives strictly inside
            the pericardial sac — the anatomical definition of "peri-LAA fat".
        air_proximity_hu / vessel_proximity_hu / pv_buffer_mm: tighten the
            ROI by removing voxels within `pv_buffer_mm` of obvious air
            (HU < air_proximity_hu) or contrast-vessel (HU > vessel_proximity_hu)
            structures — handles partial-volume bleed that the negative prior
            misses.

    Returns: PeriLAAROI with `roi_mask`, `dist_mm`, and full provenance.
    """
    laa = laa_mask.astype(bool)
    if laa.shape != ct_array.shape:
        raise ValueError(
            f"LAA mask shape {laa.shape} != CT shape {ct_array.shape}"
        )
    if exclusion_mask is not None and exclusion_mask.shape != ct_array.shape:
        raise ValueError(
            f"exclusion mask shape {exclusion_mask.shape} != CT shape {ct_array.shape}"
        )
    if pericardium_mask is not None and pericardium_mask.shape != ct_array.shape:
        raise ValueError(
            f"pericardium mask shape {pericardium_mask.shape} != CT shape {ct_array.shape}"
        )

    spacing = tuple(float(v) for v in spacing_xyz_mm)
    dist_mm = distance_transform_edt(~laa, sampling=spacing)
    band_outer_geom = (dist_mm > 0) & (dist_mm <= roi_max_distance_mm)

    pv_seed = (ct_array < air_proximity_hu) | (ct_array > vessel_proximity_hu)
    pv_excl: Optional[np.ndarray] = None
    if pv_buffer_mm > 0 and pv_seed.any():
        pv_dist = distance_transform_edt(~pv_seed, sampling=spacing)
        pv_excl = pv_dist <= pv_buffer_mm

    roi = band_outer_geom.copy()
    excl_neg_count = 0
    excl_pv_count = 0
    excl_peri_count = 0

    if exclusion_mask is not None:
        excl_neg = exclusion_mask.astype(bool)
        excl_neg_count = int((roi & excl_neg).sum())
        roi &= ~excl_neg

    if pv_excl is not None:
        excl_pv_count = int((roi & pv_excl).sum())
        roi &= ~pv_excl

    if pericardium_mask is not None:
        peri = pericardium_mask.astype(bool)
        before = int(roi.sum())
        roi &= peri
        excl_peri_count = before - int(roi.sum())

    provenance: dict[str, float | str | int | bool] = {
        "roi_max_distance_mm": float(roi_max_distance_mm),
        "air_proximity_hu": float(air_proximity_hu),
        "vessel_proximity_hu": float(vessel_proximity_hu),
        "pv_buffer_mm": float(pv_buffer_mm),
        "negative_prior_used": exclusion_mask is not None,
        "pericardium_used": pericardium_mask is not None,
        "roi_voxel_count": int(roi.sum()),
        "exclusion_voxels_negative_prior": excl_neg_count,
        "exclusion_voxels_pv_buffer": excl_pv_count,
        "exclusion_voxels_pericardium_clip": excl_peri_count,
        "exclusion_voxels_total": excl_neg_count + excl_pv_count + excl_peri_count,
        "max_shell_geom_voxel_count": int(band_outer_geom.sum()),
    }
    return PeriLAAROI(roi_mask=roi, dist_mm=dist_mm, provenance=provenance)


def compute_peri_laa_fat(
    ct_array: np.ndarray,
    laa_mask: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    fat_hu_min: float = -190.0,
    fat_hu_max: float = -30.0,
    shells_mm: Iterable[tuple[float, float]] = DEFAULT_SHELLS_MM,
    exclusion_mask: Optional[np.ndarray] = None,
    pericardium_mask: Optional[np.ndarray] = None,
    air_proximity_hu: float = -300.0,
    vessel_proximity_hu: float = 100.0,
    pv_buffer_mm: float = 1.0,
    roi: Optional[PeriLAAROI] = None,
) -> PeriLAAFatResult:
    """Compute peri-LAA fat per radial shell.

    Args:
        ct_array: HU array. Axis order must match `spacing_xyz_mm`: by
            convention (nibabel `get_fdata`) that is **(x, y, z)** — i.e.
            ``ct_array.shape[i]`` corresponds to ``spacing_xyz_mm[i]``.
            If you came from SimpleITK (`GetArrayFromImage` returns (z,y,x)),
            either transpose the array or reverse the spacing before calling.
        laa_mask: bool/int array of the LAA cavity in the same geometry.
        spacing_xyz_mm: voxel spacing along (axis-0, axis-1, axis-2) of the
            arrays above, in millimetres. Must match the array's axis order.
        fat_hu_min / fat_hu_max: adipose HU window (default standard).
        shells_mm: iterable of (inner_mm, outer_mm) bands measured radially
            outward from the LAA surface. Bands must be monotonically
            increasing; overlap is not allowed (validated).
        exclusion_mask: optional bool mask of structures to subtract from
            every shell (typically the SLAAO negative prior: aorta / lungs /
            myocardium / coronaries / vertebrae / pericardium).

    Returns: PeriLAAFatResult.

    Notes:
      * Distance transform uses physical-mm sampling per axis so a 2 mm shell
        is 2 mm regardless of anisotropy.
      * Voxel volume = product of spacings; order-invariant.
      * If `roi` is supplied it is used directly; otherwise `build_peri_laa_roi`
        is called internally with the same arguments. This keeps the
        convenience entry-point but makes the two-step pipeline explicit.
    """
    shells_list = sorted({(float(lo), float(hi)) for lo, hi in shells_mm})
    for lo, hi in shells_list:
        if hi <= lo or lo < 0:
            raise ValueError(f"invalid shell band: ({lo}, {hi})")

    spacing = tuple(float(v) for v in spacing_xyz_mm)
    voxel_vol_mm3 = float(np.prod(spacing))
    outer_max_mm = shells_list[-1][1]

    if roi is None:
        roi = build_peri_laa_roi(
            ct_array=ct_array,
            laa_mask=laa_mask,
            spacing_xyz_mm=spacing,
            roi_max_distance_mm=outer_max_mm,
            exclusion_mask=exclusion_mask,
            pericardium_mask=pericardium_mask,
            air_proximity_hu=air_proximity_hu,
            vessel_proximity_hu=vessel_proximity_hu,
            pv_buffer_mm=pv_buffer_mm,
        )

    fat_voxels = (ct_array >= fat_hu_min) & (ct_array <= fat_hu_max)
    label_mask = np.zeros(ct_array.shape, dtype=np.int16)
    features: dict[str, float | str | int] = {
        "peri_laa_fat_hu_min_used": float(fat_hu_min),
        "peri_laa_fat_hu_max_used": float(fat_hu_max),
        "peri_laa_fat_shells_mm": ";".join(f"{lo:g}-{hi:g}" for lo, hi in shells_list),
        "peri_laa_fat_max_shell_mm": float(outer_max_mm),
        "peri_laa_fat_voxel_volume_mm3": float(voxel_vol_mm3),
        "peri_laa_fat_pv_buffer_mm": float(roi.provenance["pv_buffer_mm"]),
        "peri_laa_fat_air_proximity_hu": float(roi.provenance["air_proximity_hu"]),
        "peri_laa_fat_vessel_proximity_hu": float(roi.provenance["vessel_proximity_hu"]),
        "peri_laa_roi_voxel_count": int(roi.provenance["roi_voxel_count"]),
        "peri_laa_roi_volume_ml": round(
            float(int(roi.provenance["roi_voxel_count"]) * voxel_vol_mm3) / 1000.0, 4
        ),
        "peri_laa_fat_pv_voxels_excluded":
            int(roi.provenance["exclusion_voxels_pv_buffer"]),
        "peri_laa_exclusion_voxels_used":
            int(roi.provenance["exclusion_voxels_total"]),
        "peri_laa_roi_pericardium_used": bool(roi.provenance["pericardium_used"]),
        "peri_laa_roi_negative_prior_used": bool(roi.provenance["negative_prior_used"]),
    }

    # Per-shell metrics — each band is a (lo, hi] mm slice of the ROI's
    # distance field. ROI already had every exclusion applied in step 1.
    dist_mm = roi.dist_mm
    for shell_idx, (lo_mm, hi_mm) in enumerate(shells_list, start=1):
        if lo_mm == 0.0:
            shell_geom = roi.roi_mask & (dist_mm > 0) & (dist_mm <= hi_mm)
        else:
            shell_geom = roi.roi_mask & (dist_mm > lo_mm) & (dist_mm <= hi_mm)
        shell_fat = shell_geom & fat_voxels
        label_mask[shell_fat] = shell_idx

        prefix = _shell_prefix(lo_mm, hi_mm)
        features.update(_block(prefix, ct_array, shell_fat, voxel_vol_mm3))
        features[f"{prefix}_geom_volume_ml"] = round(
            float(int(shell_geom.sum()) * voxel_vol_mm3) / 1000.0, 4
        )

    agg = label_mask > 0
    features.update(_block("peri_laa_fat_total", ct_array, agg, voxel_vol_mm3))

    return PeriLAAFatResult(
        label_mask=label_mask, shells_mm=shells_list,
        features=features, roi=roi,
    )


def extend_laa_to_perifat(
    laa_mask: np.ndarray,
    fat_result: PeriLAAFatResult,
    ct_array: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float],
    perifat_closing_mm: float = 2.0,
    max_added_volume_ml: float = 25.0,
    max_growth_mm: Optional[float] = None,
    exclusion_mask: Optional[np.ndarray] = None,
    centerline_aware: bool = True,
    la_body_mask: Optional[np.ndarray] = None,
    centerline_smoothing_window: int = 5,
    extension_hu_min: float = -100.0,
    extension_hu_max: float = 500.0,
    aorta_mask: Optional[np.ndarray] = None,
    anterior_axis_zyx: Optional[np.ndarray] = None,
    left_axis_zyx: Optional[np.ndarray] = None,
    hard_exclusion_mask: Optional[np.ndarray] = None,
) -> ExtendedLAAResult:
    """Step 3: extend the LAA outward to the peri-LAA fat boundary.

    The ROI from step 1 is deliberately **not** used here. Instead we treat
    the peri-LAA fat itself as a shell, close that shell using its own
    geometry, and fill what's enclosed. Concretely:

      1. ``perifat = union of every fat shell`` (output of step 2).
      2. ``perifat_closed = binary_closing(perifat, radius = perifat_closing_mm)``
         to bridge small gaps in the fat envelope.
      3. ``filled = binary_fill_holes(perifat_closed)``
         — the closed shell PLUS its interior pocket.
      4. ``interior = filled & ~perifat_closed`` — the pocket itself.
      5. ``extended_LAA = LAA ∪ (interior CC that contains the LAA)`` —
         only the connected-component of the interior that the original LAA
         actually sits inside. This kills runaway fills in cases where the
         closing leaks (e.g. the perifat shell has a large hole on one side).

    A safety cap (``max_added_volume_ml``) refuses the extension if the
    interior pocket is implausibly large, returning the original LAA
    unchanged plus a warning flag in the features. Default 25 mL — LAA
    typical volume is < 15 mL.

    Args:
        laa_mask: bool/int array of the LAA cavity.
        fat_result: output of `compute_peri_laa_fat`. Only `label_mask` (the
            fat shells) is used; the ROI mask is ignored on purpose.
        ct_array: HU volume — only for reporting HU stats on the added region.
        spacing_xyz_mm: per-axis spacing matching array order.
        perifat_closing_mm: closing radius in mm. Defaults to 4 mm.
            Increase if the perifat is patchy and the LAA appears partially
            unsealed; decrease if neighbour structures get absorbed.
        max_added_volume_ml: refuse the extension if the new volume would
            grow the LAA by more than this amount.

    Returns: ExtendedLAAResult.
    """
    laa = laa_mask.astype(bool)
    if laa.shape != fat_result.label_mask.shape:
        raise ValueError(
            f"LAA mask shape {laa.shape} != fat-label shape {fat_result.label_mask.shape}"
        )

    perifat = fat_result.label_mask > 0
    spacing = tuple(float(v) for v in spacing_xyz_mm)
    voxel_vol_mm3 = float(np.prod(spacing))

    # Hard distance bound. By default we use the *median* radial distance of
    # the perifat voxels from the LAA surface as the boundary — that's
    # statistically "where the fat actually sits", not the outermost
    # detected speck. This makes the extension match the dominant
    # perifat-shell position rather than the longest reach.
    if max_growth_mm is None:
        if perifat.any():
            laa_for_default = laa_mask.astype(bool)
            d_default = distance_transform_edt(~laa_for_default, sampling=spacing)
            max_growth_mm = float(np.median(d_default[perifat]))
        elif fat_result.shells_mm:
            max_growth_mm = float(max(hi for _, hi in fat_result.shells_mm))
        else:
            max_growth_mm = 5.0

    # Strategy: define the perifat boundary implicitly via a Voronoi split.
    # For every voxel, it's "on the LAA side" of the perifat iff its
    # Euclidean distance to the nearest LAA surface is *less than* its
    # distance to the nearest perifat voxel. This partition is well defined
    # even when the perifat is patchy and doesn't form a closed 3D surface —
    # which is the practical case for most real LAAs in this dataset.
    #
    # `perifat_closing_mm` thickens the perifat slightly before the split, so
    # a few isolated fat voxels don't act as tiny "magnets" that pull the
    # boundary inward.
    struct = generate_binary_structure(3, 1)
    closing_vox = max(0, int(np.ceil(
        perifat_closing_mm / max(min(spacing), 1e-6)
    )))
    if closing_vox > 0:
        perifat_seed = binary_closing(perifat, structure=struct, iterations=closing_vox)
    else:
        perifat_seed = perifat.copy()

    extension_failed = False
    if not perifat_seed.any():
        extension_failed = True
        extension_reason = "no_perifat"
        candidate_added = np.zeros_like(laa)
        fill_method = "n/a"
    elif not laa.any():
        extension_failed = True
        extension_reason = "empty_laa"
        candidate_added = np.zeros_like(laa)
        fill_method = "n/a"
    else:
        dist_to_laa = distance_transform_edt(~laa, sampling=spacing)
        dist_to_perifat = distance_transform_edt(~perifat_seed, sampling=spacing)
        # HU floor: the LAA, its wall, and any thrombus all sit well above
        # air HU. Anything below `extension_hu_min` (default -100) in the
        # fill region is either lung air, oral airway, sinus air, or
        # partial-volume of those — none of which is LAA tissue.
        on_hu = (ct_array >= float(extension_hu_min)) & (ct_array <= float(extension_hu_max))

        on_laa_side = (
            (~laa) & (~perifat_seed)
            & (dist_to_laa < dist_to_perifat)
            & (dist_to_laa <= float(max_growth_mm))
            & on_hu
        )
        if exclusion_mask is not None:
            on_laa_side = on_laa_side & ~exclusion_mask.astype(bool)

        # Centerline-aware half-space restriction. The LAA bends: growth
        # should only proceed in the centerline direction PAST the bend,
        # not radially in all directions (which fights the LAA's geometry
        # at the bend itself).
        # Derive the aorta-relative lateral axis once. The diagnostic
        # showed the full perifat centroid is nearly perpendicular to this
        # axis on real data — meaning plane 2 alone leaks toward the aorta.
        # We use the lateral axis (a) to *filter* the perifat to the LEFT
        # subset before computing plane 2, and (b) as plane 3 directly.
        lateral_axis: Optional[np.ndarray] = None
        aorta_centroid_phys: Optional[np.ndarray] = None
        laa_centroid_phys: Optional[np.ndarray] = None
        if aorta_mask is not None and aorta_mask.any() and laa.any():
            laa_idx = np.argwhere(laa)
            aorta_idx = np.argwhere(aorta_mask.astype(bool))
            laa_centroid_phys = (laa_idx * spacing).mean(axis=0)
            aorta_centroid_phys = (aorta_idx * spacing).mean(axis=0)
            raw_axis = laa_centroid_phys - aorta_centroid_phys
            n = float(np.linalg.norm(raw_axis))
            if n > 1e-6:
                lateral_axis = raw_axis / n

        centerline_used = False
        bend_angle = _NAN
        centerline_notes = ""
        if centerline_aware:
            try:
                cl = compute_laa_centerline_and_bend(
                    laa_mask=laa, spacing_xyz_mm=spacing,
                    la_body_mask=la_body_mask,
                    smoothing_window=centerline_smoothing_window,
                    perifat_mask=perifat,
                    lateral_axis_zyx=lateral_axis,
                )
                centerline_notes = cl.notes
                if (cl.post_bend_tangent_zyx is not None
                        and cl.bend_point_phys_mm is not None):
                    sh = laa.shape
                    zz, yy, xx = np.indices(sh, dtype=np.float32)

                    # Plane 1 — bend plane: keep voxels distal of the bend.
                    pz = zz * spacing[0] - cl.bend_point_phys_mm[0]
                    py = yy * spacing[1] - cl.bend_point_phys_mm[1]
                    px = xx * spacing[2] - cl.bend_point_phys_mm[2]
                    proj_bend = (pz * cl.post_bend_tangent_zyx[0]
                                 + py * cl.post_bend_tangent_zyx[1]
                                 + px * cl.post_bend_tangent_zyx[2])
                    distal_half = proj_bend > 0
                    on_laa_side = on_laa_side & distal_half

                    # Planes 2 & 3 are perifat / aorta-vector heuristics.
                    # They are useful only as a fallback when explicit
                    # anatomical axes (anterior + left from the affine)
                    # are NOT available — those planes can disagree with
                    # the real LPS directions (e.g. on sub-547 the LEFT-
                    # perifat centroid lies posteriorly, fighting plane 4).
                    # When the affine gave us honest anterior + left, the
                    # user's spec is "bend + anterior + left" exactly, so
                    # we skip planes 2 & 3 to avoid an empty intersection.
                    use_proxy_planes = not (
                        anterior_axis_zyx is not None
                        and left_axis_zyx is not None
                    )

                    laa_phys_idx = np.argwhere(laa)
                    perifat_phys_idx = np.argwhere(perifat)
                    if use_proxy_planes and perifat_phys_idx.size and laa_phys_idx.size:
                        laa_centroid = (
                            laa_centroid_phys if laa_centroid_phys is not None
                            else (laa_phys_idx * spacing).mean(axis=0)
                        )
                        pf_phys = perifat_phys_idx * spacing
                        if lateral_axis is not None:
                            offsets_pf = pf_phys - laa_centroid
                            proj_left = offsets_pf @ lateral_axis
                            left_pf = pf_phys[proj_left > 0]
                            if left_pf.shape[0] > max(
                                20, int(0.2 * pf_phys.shape[0])
                            ):
                                pf_phys = left_pf
                                centerline_notes = centerline_notes + "_left_perifat"
                        pf_centroid = pf_phys.mean(axis=0)
                        pf_axis = pf_centroid - laa_centroid
                        n = float(np.linalg.norm(pf_axis))
                        if n > 1e-6:
                            pf_axis /= n
                            pz2 = zz * spacing[0] - laa_centroid[0]
                            py2 = yy * spacing[1] - laa_centroid[1]
                            px2 = xx * spacing[2] - laa_centroid[2]
                            proj_pf = (pz2 * pf_axis[0]
                                       + py2 * pf_axis[1]
                                       + px2 * pf_axis[2])
                            on_laa_side = on_laa_side & (proj_pf > 0)

                    # Plane 3 — aorta-away plane through the LAA centroid.
                    # Same caveat as plane 2: it's a proxy for "patient left"
                    # via the actual aorta position. Skip when an explicit
                    # LPS left axis is supplied (plane 5 is then the honest
                    # version of the same idea).
                    if (use_proxy_planes
                            and lateral_axis is not None
                            and laa_centroid_phys is not None):
                        pz3 = zz * spacing[0] - laa_centroid_phys[0]
                        py3 = yy * spacing[1] - laa_centroid_phys[1]
                        px3 = xx * spacing[2] - laa_centroid_phys[2]
                        proj_lat = (pz3 * lateral_axis[0]
                                    + py3 * lateral_axis[1]
                                    + px3 * lateral_axis[2])
                        on_laa_side = on_laa_side & (proj_lat > 0)
                        centerline_notes = centerline_notes + "_aorta_plane"

                    # Planes 4 & 5 — empirically-derived ANTERIOR and LEFT
                    # half-spaces. We derive both from the data itself so
                    # the result is independent of LPS-vs-RAS NIfTI
                    # convention (which silently flipped earlier attempts):
                    #
                    #   LEFT     = LAA_centroid − aorta_centroid
                    #              (aorta is anatomically right of the LAA,
                    #               so this vector points patient-left by
                    #               construction; lateral_axis already holds it)
                    #
                    #   ANTERIOR = post_bend_tangent × LEFT
                    #              (perpendicular to bend & left, oriented
                    #               toward the perifat centroid because the
                    #               peri-LAA fat sits anterolaterally)
                    #
                    # When the caller explicitly passes anterior/left axes
                    # (e.g. derived from a trusted affine) we use those
                    # instead — but the empirical fallback is the default
                    # because it is robust to broken sform/qform tags like
                    # the one on sub-547.
                    if lateral_axis is not None and laa_centroid_phys is not None:
                        # ----- LEFT plane (empirical: lateral_axis) -----
                        left_vec = (np.asarray(left_axis_zyx, dtype=np.float64)
                                    if left_axis_zyx is not None
                                    else lateral_axis.copy())
                        ln = float(np.linalg.norm(left_vec))
                        if ln > 1e-6:
                            left_vec = left_vec / ln
                            pz5 = zz * spacing[0] - laa_centroid_phys[0]
                            py5 = yy * spacing[1] - laa_centroid_phys[1]
                            px5 = xx * spacing[2] - laa_centroid_phys[2]
                            proj_lft = (pz5 * left_vec[0]
                                        + py5 * left_vec[1]
                                        + px5 * left_vec[2])
                            on_laa_side = on_laa_side & (proj_lft > 0)
                            centerline_notes = centerline_notes + "_left_plane"

                        # ----- ANTERIOR plane -----
                        if anterior_axis_zyx is not None:
                            ant_vec = np.asarray(anterior_axis_zyx, dtype=np.float64)
                        else:
                            # cross(bend_tangent, left) → perpendicular to both
                            ant_vec = np.cross(cl.post_bend_tangent_zyx, left_vec)
                            # Orient toward the perifat centroid (which sits
                            # on the anterolateral side of the LAA).
                            pf_idx = np.argwhere(perifat)
                            if pf_idx.size:
                                pf_centroid_local = (pf_idx * spacing).mean(axis=0)
                                toward_pf = pf_centroid_local - laa_centroid_phys
                                if np.dot(ant_vec, toward_pf) < 0:
                                    ant_vec = -ant_vec
                        an = float(np.linalg.norm(ant_vec))
                        if an > 1e-6:
                            ant_vec = ant_vec / an
                            pz4 = zz * spacing[0] - laa_centroid_phys[0]
                            py4 = yy * spacing[1] - laa_centroid_phys[1]
                            px4 = xx * spacing[2] - laa_centroid_phys[2]
                            proj_ant = (pz4 * ant_vec[0]
                                        + py4 * ant_vec[1]
                                        + px4 * ant_vec[2])
                            on_laa_side = on_laa_side & (proj_ant > 0)
                            centerline_notes = centerline_notes + "_anterior_plane"

                    centerline_used = True
                    bend_angle = cl.bend_angle_deg
            except Exception as exc:
                # Centerline analysis failed — fall back to radial growth.
                pass
        # Restrict to the connected component touching the LAA so we don't
        # absorb unrelated "closer to LAA" lobes far away from it.
        seed_region = on_laa_side | laa
        labelled, _ = cc_label(seed_region, structure=struct)
        laa_label = int(np.bincount(labelled[laa].ravel()).argmax())
        if laa_label == 0:
            extension_failed = True
            extension_reason = "laa_disconnected_from_fill"
            candidate_added = np.zeros_like(laa)
        else:
            connected = labelled == laa_label
            candidate_added = connected & ~laa
            extension_reason = "ok"
        fill_method = "voronoi_partition"
    added_volume_ml = float(int(candidate_added.sum()) * voxel_vol_mm3) / 1000.0

    # Soft cap: keep the fill so the user can visualise it, but flag the
    # status so downstream code can decide whether to accept the mask.
    if (not extension_failed) and added_volume_ml > max_added_volume_ml:
        extension_reason = (
            f"warn_added_volume_{added_volume_ml:.1f}_ml_exceeds_cap_{max_added_volume_ml}"
        )

    extended = laa | candidate_added

    # ---- Hard exclusion (overrides LAA consensus) ----
    # Some masks (notably the coronary tree + a small pericoronary buffer)
    # MUST be removed from the final extended LAA even when they happen to
    # overlap the original LAA consensus — that overlap is a segmentation
    # error in the consensus, not anatomy. The soft `exclusion_mask` above
    # only filtered candidate growth; this one trims `extended` directly.
    hard_excl_removed = 0
    if hard_exclusion_mask is not None:
        hex_arr = hard_exclusion_mask.astype(bool)
        if hex_arr.shape == extended.shape:
            hard_excl_removed = int((extended & hex_arr).sum())
            extended = extended & ~hex_arr

    # ---- features ----
    n_laa = int(laa.sum())
    n_ext = int(extended.sum())
    n_added = int((extended & ~laa).sum())
    n_perifat = int(perifat.sum())

    features: dict[str, float | str | int] = {
        "extended_laa_voxel_count": n_ext,
        "extended_laa_volume_ml": round(float(n_ext * voxel_vol_mm3) / 1000.0, 4),
        "extended_laa_added_voxel_count": n_added,
        "extended_laa_added_volume_ml": round(float(n_added * voxel_vol_mm3) / 1000.0, 4),
        "extended_laa_growth_ratio":
            round(n_ext / n_laa, 4) if n_laa else _NAN,
        "peri_laa_fat_combined_voxel_count": n_perifat,
        "peri_laa_fat_combined_volume_ml":
            round(float(n_perifat * voxel_vol_mm3) / 1000.0, 4),
        "extended_laa_perifat_closing_mm": float(perifat_closing_mm),
        "extended_laa_max_added_volume_ml": float(max_added_volume_ml),
        "extended_laa_max_growth_mm": float(max_growth_mm),
        "extended_laa_extension_status": extension_reason,
        "extended_laa_extension_succeeded": (not extension_failed),
        "extended_laa_fill_method": fill_method,
        "extended_laa_centerline_aware": bool(centerline_used),
        "extended_laa_centerline_bend_angle_deg": float(bend_angle),
        "extended_laa_centerline_notes": str(centerline_notes),
        "extended_laa_hu_min": float(extension_hu_min),
        "extended_laa_hu_max": float(extension_hu_max),
        "extended_laa_hard_exclusion_voxels_removed": int(hard_excl_removed),
    }

    # HU stats on the added (non-fat) region — should track myocardium /
    # thrombus density (~+30..+80 HU). If it's wildly different the user can
    # spot leakage at a glance.
    added_only = extended & ~laa
    if added_only.any():
        hu = ct_array[added_only].astype(np.float32)
        features["extended_laa_added_mean_hu"] = round(float(hu.mean()), 2)
        features["extended_laa_added_median_hu"] = round(float(np.median(hu)), 2)
        features["extended_laa_added_p10_hu"] = round(float(np.percentile(hu, 10)), 2)
        features["extended_laa_added_p90_hu"] = round(float(np.percentile(hu, 90)), 2)
    else:
        for k in ("mean_hu", "median_hu", "p10_hu", "p90_hu"):
            features[f"extended_laa_added_{k}"] = _NAN

    return ExtendedLAAResult(
        extended_laa=extended.astype(bool),
        perifat_combined=perifat.astype(bool),
        features=features,
    )


def save_peri_laa_fat(
    result: PeriLAAFatResult,
    reference_affine: np.ndarray,
    out_dir: Path,
    case_id: str,
    write_per_shell_masks: bool = False,
    write_roi_mask: bool = True,
) -> dict[str, Path]:
    """Persist results.

    Writes:
      * `<case_id>_peri_laa_roi.nii.gz` — the ROI itself (step 1 output)
      * `<case_id>_peri_laa_fat_labels.nii.gz` — multi-label NIfTI (1..N per shell)
      * `<case_id>_peri_laa_fat_metrics.json` — flat metrics + shell map + ROI provenance
      * optionally `<case_id>_peri_laa_fat_shell{lo}_{hi}.nii.gz` — binary per shell

    Returns a dict mapping kind → path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if write_roi_mask:
        roi_img = nib.Nifti1Image(result.roi.roi_mask.astype(np.uint8), reference_affine)
        roi_path = out_dir / f"{case_id}_peri_laa_roi.nii.gz"
        nib.save(roi_img, roi_path)
        paths["roi"] = roi_path

    label_img = nib.Nifti1Image(result.label_mask.astype(np.int16), reference_affine)
    labels_path = out_dir / f"{case_id}_peri_laa_fat_labels.nii.gz"
    nib.save(label_img, labels_path)
    paths["labels"] = labels_path

    metrics_path = out_dir / f"{case_id}_peri_laa_fat_metrics.json"
    metrics_path.write_text(json.dumps({
        "case_id": case_id,
        "shells_mm": result.shells_mm,
        "features": result.features,
        "roi_provenance": result.roi.provenance,
        "label_value_to_shell_mm": {
            i + 1: list(band) for i, band in enumerate(result.shells_mm)
        },
        "disclaimer": (
            "RESEARCH PROTOTYPE — peri-LAA fat envelope features; not validated "
            "for clinical use."
        ),
    }, indent=2, default=str))
    paths["metrics"] = metrics_path

    if write_per_shell_masks:
        for idx, (lo, hi) in enumerate(result.shells_mm, start=1):
            mask_bin = (result.label_mask == idx).astype(np.uint8)
            mask_img = nib.Nifti1Image(mask_bin, reference_affine)
            # `_shell_prefix` is filename-safe (won't eat `.nii.gz`).
            name = f"{case_id}_{_shell_prefix(lo, hi)}.nii.gz"
            p = out_dir / name
            nib.save(mask_img, p)
            paths[f"shell_{idx}"] = p

    return paths


# ---------------------------------------------------------------------------
# File-driven entry point — single call from disk paths to disk outputs.
# ---------------------------------------------------------------------------

def _anatomical_axes_from_affine(
    affine: np.ndarray, spacing: tuple[float, float, float],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (anterior, left, superior) axes in our scaled-array-index space.

    NIfTI affines map array indices to LPS physical mm. Patient directions:
    L = +x_phys, P = +y_phys, S = +z_phys. We want:
        anterior = -y_phys   (toward chest wall)
        left     = +x_phys   (patient's left)
        superior = +z_phys   (toward head)

    The function transforms each unit LPS direction back into our
    "array_index × abs(spacing)" coordinate system used throughout the
    pipeline so the resulting axes can be dotted with our
    centroid-relative voxel offsets directly.

    Returns (None, None, None) if the affine is unusable (singular).
    """
    try:
        rot = np.asarray(affine[:3, :3], dtype=np.float64)
        inv = np.linalg.inv(rot)
    except Exception:
        return None, None, None
    abs_sp = np.array([float(v) for v in spacing], dtype=np.float64)
    anterior_lps = np.array([0.0, -1.0, 0.0])  # -y in LPS = anterior
    left_lps     = np.array([+1.0, 0.0, 0.0])  # +x in LPS = patient left
    superior_lps = np.array([0.0, 0.0, +1.0])  # +z in LPS = superior

    def _to_array_space(d_lps: np.ndarray) -> np.ndarray:
        arr = inv @ d_lps              # array-index direction
        scaled = arr * abs_sp           # rescale into our scaled-index space
        n = np.linalg.norm(scaled)
        return scaled / n if n > 1e-9 else scaled

    return (
        _to_array_space(anterior_lps),
        _to_array_space(left_lps),
        _to_array_space(superior_lps),
    )


def run_peri_laa_fat_from_paths(
    ct_path: Path,
    laa_mask_path: Path,
    out_dir: Path,
    case_id: str,
    fat_hu_min: float = -190.0,
    fat_hu_max: float = -30.0,
    shells_mm: Iterable[tuple[float, float]] = DEFAULT_SHELLS_MM,
    negative_prior_path: Optional[Path] = None,
    write_per_shell_masks: bool = False,
    air_proximity_hu: float = -300.0,
    vessel_proximity_hu: float = 100.0,
    pv_buffer_mm: float = 1.0,
    extend_laa: bool = False,
    extend_laa_closing_mm: float = 4.0,
    extend_laa_max_added_volume_ml: float = 25.0,
    extend_laa_hu_min: float = -100.0,
    extend_laa_hu_max: float = 500.0,
    positive_prior_path: Optional[Path] = None,
    extra_exclusion_paths: Optional[list[Path]] = None,
    aorta_mask_path: Optional[Path] = None,
    coronary_mask_paths: Optional[list[Path]] = None,
    pericoronary_buffer_mm: float = 1.5,
    calcification_hu_min: float = 500.0,
    calcification_neighborhood_mm: float = 10.0,
) -> tuple[PeriLAAFatResult, dict[str, Path]]:
    """Convenience wrapper used by the CLI scripts."""
    ct_nib = nib.load(str(ct_path))
    laa_nib = nib.load(str(laa_mask_path))
    ct_arr = np.asarray(ct_nib.get_fdata(), dtype=np.float32)
    laa_arr = np.asarray(laa_nib.get_fdata())

    if ct_arr.shape != laa_arr.shape:
        raise ValueError(
            f"CT shape {ct_arr.shape} != LAA mask shape {laa_arr.shape}"
        )

    # Pick the affine carefully. Some CT NIfTIs in this repo have an
    # `sform` that disagrees with `pixdim` (the ITK warning was real:
    # "unexpected scales in sform"). Using the CT's affine to save derived
    # masks then renders them at the wrong physical extent — the LAA-derived
    # ROI specifically looked stretched 2x along z in Slicer because the
    # CT's affine encoded z-scale 0.5 while pixdim said 0.25. The prior-fusion
    # outputs (LAA masks etc.) were written from a consistent affine, so we
    # adopt the LAA mask's affine as the reference for everything we emit.
    reference_affine = laa_nib.affine
    if not np.allclose(reference_affine[:3, :3], ct_nib.affine[:3, :3]):
        log_msg = (
            "CT affine differs from LAA mask affine — saving derived masks "
            "with the LAA mask's affine (assumed canonical for this case)."
        )
        # Emit via print so users see it even without a configured logger.
        print(f"[peri_laa_fat] {log_msg}")

    excl_arr = None
    if negative_prior_path is not None and Path(negative_prior_path).is_file():
        excl_nib = nib.load(str(negative_prior_path))
        excl_arr = np.asarray(excl_nib.get_fdata()) > 0
        if excl_arr.shape != ct_arr.shape:
            raise ValueError(
                f"negative-prior shape {excl_arr.shape} != CT shape {ct_arr.shape}"
            )

    # Spacing in axis-0..2 order — derived from the LAA mask's affine (its
    # diagonal magnitudes), not from `header.get_zooms()`, because the latter
    # can disagree with the sform on broken-NIfTI inputs (see affine note
    # above). Norm of each column of the rotation block gives the per-axis
    # voxel size.
    aff_rot = reference_affine[:3, :3]
    spacing = tuple(float(np.linalg.norm(aff_rot[:, i])) for i in range(3))
    if not np.allclose(spacing, ct_nib.header.get_zooms()[:3], atol=1e-3):
        print(f"[peri_laa_fat] Using affine-derived spacing {tuple(round(s,4) for s in spacing)} "
              f"(CT header.get_zooms says {tuple(round(float(s),4) for s in ct_nib.header.get_zooms()[:3])}).")

    result = compute_peri_laa_fat(
        ct_array=ct_arr,
        laa_mask=laa_arr,
        spacing_xyz_mm=spacing,  # type: ignore[arg-type]
        fat_hu_min=fat_hu_min,
        fat_hu_max=fat_hu_max,
        shells_mm=shells_mm,
        exclusion_mask=excl_arr,
        air_proximity_hu=air_proximity_hu,
        vessel_proximity_hu=vessel_proximity_hu,
        pv_buffer_mm=pv_buffer_mm,
    )
    paths = save_peri_laa_fat(
        result=result,
        reference_affine=reference_affine,
        out_dir=Path(out_dir),
        case_id=case_id,
        write_per_shell_masks=write_per_shell_masks,
    )

    if extend_laa:
        # The negative prior excludes myocardium / aorta / lungs but NOT the
        # LA body (LA + LAA are the *positive* prior). Without an LA-body
        # exclusion the Voronoi extension flows through the LAA orifice into
        # the LA cavity (HU ≈ 200 with contrast — clearly not LAA wall).
        # If a positive prior is supplied we derive LA-body = positive − LAA
        # and union it into the exclusion.
        ext_excl = excl_arr
        if positive_prior_path is not None and Path(positive_prior_path).is_file():
            pos_nib = nib.load(str(positive_prior_path))
            pos_arr = np.asarray(pos_nib.get_fdata()) > 0
            if pos_arr.shape == ct_arr.shape:
                la_body = pos_arr & ~laa_arr.astype(bool)
                ext_excl = la_body if ext_excl is None else (ext_excl | la_body)
        # Extra exclusion masks (e.g. standalone coronary_arteries.nii.gz).
        # Coronaries are only partly covered by the SLAAO negative prior
        # depending on which TS tasks were run, so we union the dedicated
        # mask in too — coronaries that brush against the LAA must NEVER
        # become "LAA wall".
        if extra_exclusion_paths:
            for p in extra_exclusion_paths:
                if not Path(p).is_file():
                    continue
                try:
                    m = np.asarray(nib.load(str(p)).get_fdata()) > 0
                except Exception:
                    continue
                if m.shape != ct_arr.shape:
                    continue
                ext_excl = m if ext_excl is None else (ext_excl | m)

        # Reuse the LA-body mask (positive_prior - LAA) we already derived
        # for the exclusion to also drive centerline orientation.
        la_body_for_centerline = None
        if positive_prior_path is not None and Path(positive_prior_path).is_file():
            pos_nib2 = nib.load(str(positive_prior_path))
            pos_arr2 = np.asarray(pos_nib2.get_fdata()) > 0
            if pos_arr2.shape == ct_arr.shape:
                la_body_for_centerline = pos_arr2 & ~laa_arr.astype(bool)

        # Load aorta mask if provided. Resample to CT grid when shape
        # differs (TotalSegmentator masks are commonly at the same
        # geometry but defensive resampling avoids silent misalignment).
        aorta_arr = None
        if aorta_mask_path is not None and Path(aorta_mask_path).is_file():
            try:
                aorta_nib_ = nib.load(str(aorta_mask_path))
                aorta_arr_raw = np.asarray(aorta_nib_.get_fdata()) > 0
                if aorta_arr_raw.shape == ct_arr.shape:
                    aorta_arr = aorta_arr_raw
                else:
                    from scipy.ndimage import zoom
                    factors = tuple(
                        ct_arr.shape[i] / aorta_arr_raw.shape[i] for i in range(3)
                    )
                    aorta_arr = zoom(
                        aorta_arr_raw.astype(np.uint8), factors, order=0,
                    ) > 0
            except Exception as exc:
                print(f"[peri_laa_fat] Could not read aorta mask: {exc}")

        # ---- Hard exclusion: coronaries (+ pericoronary buffer) and
        # calcifications (HU > calcification_hu_min, restricted to LAA
        # neighborhood). Applied to the final extended LAA AFTER the LAA
        # union — overrides the LAA consensus where it erroneously
        # overlaps these structures.
        from scipy.ndimage import binary_dilation
        hard_excl: Optional[np.ndarray] = None

        if coronary_mask_paths:
            cor_combined = np.zeros(ct_arr.shape, dtype=bool)
            for p in coronary_mask_paths:
                if not Path(p).is_file():
                    continue
                try:
                    cm = np.asarray(nib.load(str(p)).get_fdata()) > 0
                except Exception:
                    continue
                if cm.shape == ct_arr.shape:
                    cor_combined |= cm
            if cor_combined.any():
                if pericoronary_buffer_mm > 0:
                    n_vox = max(1, int(np.ceil(
                        pericoronary_buffer_mm / max(min(spacing), 1e-6)
                    )))
                    cor_combined = binary_dilation(cor_combined, iterations=n_vox)
                hard_excl = cor_combined if hard_excl is None else (hard_excl | cor_combined)

        if calcification_hu_min is not None and calcification_hu_min < float("inf"):
            # Restrict to a small neighbourhood around the LAA so we don't
            # touch unrelated bright calcium (vertebrae, sternum, etc.).
            n_vox_n = max(1, int(np.ceil(
                calcification_neighborhood_mm / max(min(spacing), 1e-6)
            )))
            laa_neigh = binary_dilation(
                laa_arr.astype(bool), iterations=n_vox_n,
            )
            calcium = (ct_arr > float(calcification_hu_min)) & laa_neigh
            if calcium.any():
                hard_excl = calcium if hard_excl is None else (hard_excl | calcium)

        # We deliberately do NOT pass affine-derived anterior/left axes
        # anymore; the extension derives them empirically from
        # bend_tangent × (LAA−aorta) and orients toward the perifat
        # centroid. That's robust to LPS-vs-RAS confusion in the source
        # NIfTI (which silently flipped the prior LPS-assumed derivation).
        ext = extend_laa_to_perifat(
            laa_mask=laa_arr.astype(bool),
            fat_result=result,
            ct_array=ct_arr,
            spacing_xyz_mm=spacing,
            perifat_closing_mm=extend_laa_closing_mm,
            max_added_volume_ml=extend_laa_max_added_volume_ml,
            exclusion_mask=ext_excl,
            centerline_aware=True,
            la_body_mask=la_body_for_centerline,
            extension_hu_min=extend_laa_hu_min,
            extension_hu_max=extend_laa_hu_max,
            aorta_mask=aorta_arr,
            hard_exclusion_mask=hard_excl,
        )
        # Persist the combined perifat (the "outside layer") and the
        # extended LAA so they can both be loaded into Slicer.
        combined_img = nib.Nifti1Image(
            ext.perifat_combined.astype(np.uint8), reference_affine,
        )
        combined_path = Path(out_dir) / f"{case_id}_peri_laa_fat_combined.nii.gz"
        nib.save(combined_img, combined_path)
        paths["fat_combined"] = combined_path

        ext_img = nib.Nifti1Image(
            ext.extended_laa.astype(np.uint8), reference_affine,
        )
        ext_path = Path(out_dir) / f"{case_id}_extended_laa.nii.gz"
        nib.save(ext_img, ext_path)
        paths["extended_laa"] = ext_path

        # Merge the extension metrics back into the metrics JSON so a
        # single file remains the per-case source of truth.
        metrics_path = paths["metrics"]
        meta = json.loads(metrics_path.read_text())
        meta["features"].update(ext.features)
        metrics_path.write_text(json.dumps(meta, indent=2, default=str))

    return result, paths


# ---------------------------------------------------------------------------
# Small helper — HU/volume block (mirrors stroke_cta_osa.fat._block style)
# but local to this subproject so we don't carry a cross-package import.
# ---------------------------------------------------------------------------

def _block(prefix: str, hu_arr: np.ndarray, mask: np.ndarray,
           voxel_vol_mm3: float) -> dict[str, float | int]:
    if not mask.any():
        return {
            f"{prefix}_volume_ml": 0.0,
            f"{prefix}_voxel_count": 0,
            f"{prefix}_mean_hu": _NAN,
            f"{prefix}_median_hu": _NAN,
            f"{prefix}_p10_hu": _NAN,
            f"{prefix}_p90_hu": _NAN,
            f"{prefix}_std_hu": _NAN,
        }
    n = int(mask.sum())
    hu = hu_arr[mask].astype(np.float32)
    return {
        f"{prefix}_volume_ml": round(float(n * voxel_vol_mm3) / 1000.0, 4),
        f"{prefix}_voxel_count": n,
        f"{prefix}_mean_hu": round(float(hu.mean()), 2),
        f"{prefix}_median_hu": round(float(np.median(hu)), 2),
        f"{prefix}_p10_hu": round(float(np.percentile(hu, 10)), 2),
        f"{prefix}_p90_hu": round(float(np.percentile(hu, 90)), 2),
        f"{prefix}_std_hu": round(float(hu.std()), 2),
    }
