"""Mandible + oral cavity features.

Tests cover:
  * external mask path (known volume → mL).
  * bone-HU threshold fallback: only the shape-scored arch wins.
  * mandibular plane resolution priority: bundle point+normal → 3 landmark
    points → mask extrema → None.
  * point-to-plane distance: hyoid → mandibular plane in physical mm.
  * oral cavity volume passed through.
"""

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

from stroke_cta_osa.landmark_schema import (
    LandmarkBundle, LandmarkPlane, LandmarkPoint,
)
from stroke_cta_osa.mandible import (
    MandibleConfig, OralCavityConfig, compute_mandible_features,
    _point_to_plane_distance, _resolve_mandible_mask,
    _resolve_mandibular_plane,
)
from stroke_cta_osa.types import CTAImage


# ---------------------------------------------------------------------------
# Synthetic mandible mask helpers
# ---------------------------------------------------------------------------

def _arch_mask(shape=(80, 80, 80)) -> np.ndarray:
    """A U-shaped arch: anterior bow at small y (front of jaw), opening
    posteriorly toward larger y, with L/R rami at the lateral extremes.

    The shape is wide in x (50 voxels), short in z (8 voxels), and curved
    in y so the inferior extrema are NON-collinear (so the mask-extrema
    plane fallback can build a non-degenerate plane).
    """
    mask = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    # Arch: in z=66..74, the y at the anterior bow follows a U opening to +y.
    # bow_y(x) = 20 + 0.02 * (x - 40)² — anterior at x=40 (y=20), posterior at
    # edges (y ≈ 32 for |x-40|=25).
    bow_y = 20 + 0.02 * (xx - 40) ** 2
    in_arch = (zz >= 66) & (zz < 74) & (yy >= bow_y) & (yy < bow_y + 6) \
              & (xx >= 15) & (xx < 65)
    mask |= np.broadcast_to(in_arch, shape)
    return mask


def _high_hu_image_with_mandible_and_spine(shape=(80, 80, 80)) -> CTAImage:
    """Synthetic CTA where:
      * mandible arch (wide x, narrow z) is at HU = 400
      * cervical spine (narrow x, tall z) is at HU = 350
      * background is 40 HU
    The shape-scoring fallback should pick the arch (x_extent / z_extent is
    much higher).
    """
    arr = np.full(shape, 40, dtype=np.int16)
    # mandible arch (uniform cuboid for simplicity; voxel count ≈ 8064 ≈ 8 mL).
    # z=66..74, y=20..38, x=12..68 (extents: z=8, y=18, x=56).
    arr[66:74, 20:38, 12:68] = 400
    # spine: z=10..70, y=55..60, x=37..43 (x_extent=6, z_extent=60)
    arr[10:70, 55:60, 37:43] = 350
    return CTAImage(
        array=arr,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/synth_bone.nii.gz"),
        study_id="stu_bone", scan_id="scn_bone", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )


def _bundle_with_mandible_landmarks():
    """Three points that define a flat plane (the floor of the mandible)."""
    b = LandmarkBundle(image_shape_zyx=(80, 80, 80))
    b.points["menton"] = LandmarkPoint(
        name="menton", voxel_zyx=(70, 20, 40),
        physical_mm=(40.0, 20.0, 70.0), source="external_json",
    )
    b.points["gonion_left"] = LandmarkPoint(
        name="gonion_left", voxel_zyx=(70, 35, 15),
        physical_mm=(15.0, 35.0, 70.0), source="external_json",
    )
    b.points["gonion_right"] = LandmarkPoint(
        name="gonion_right", voxel_zyx=(70, 35, 65),
        physical_mm=(65.0, 35.0, 70.0), source="external_json",
    )
    b.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=(40, 50, 40),
        physical_mm=(40.0, 50.0, 40.0), source="external_json",
    )
    return b


# ---------------------------------------------------------------------------
# Disabled / missing
# ---------------------------------------------------------------------------

def test_disabled_config_returns_empty(synth_cta):
    cfg = MandibleConfig(enabled=False)
    out = compute_mandible_features(synth_cta, cfg, None, LandmarkBundle())
    assert out["mandible_mask_available"] is False
    assert out["mandible_mask_method"] == "disabled"


def test_no_mask_and_no_fallback_returns_unavailable(synth_cta):
    cfg = MandibleConfig(allow_bone_threshold_fallback=False)
    out = compute_mandible_features(synth_cta, cfg, None, LandmarkBundle())
    assert out["mandible_mask_available"] is False
    assert out["mandible_mask_method"] == "absent_threshold_fallback_disabled"


def test_no_bone_voxels_in_threshold_fallback(synth_cta):
    """synth_cta has max HU around 40 — no bone-threshold candidates."""
    cfg = MandibleConfig(allow_bone_threshold_fallback=True, bone_hu_min=1000.0)
    out = compute_mandible_features(synth_cta, cfg, None, LandmarkBundle())
    assert out["mandible_mask_available"] is False
    assert "no_bone" in out["mandible_mask_method"] \
        or "no_plausible" in out["mandible_mask_method"]


# ---------------------------------------------------------------------------
# External mask path
# ---------------------------------------------------------------------------

def test_external_mask_volume(synth_cta):
    mask = _arch_mask()
    cfg = MandibleConfig()
    out = compute_mandible_features(synth_cta, cfg, mask, LandmarkBundle())
    assert out["mandible_mask_available"] is True
    assert out["mandible_mask_method"] == "external_mask"
    n_vox = int(mask.sum())
    assert out["mandible_volume_mm3"] == pytest.approx(n_vox * 1.0, rel=1e-3)
    assert out["mandible_volume_ml"] == pytest.approx(n_vox / 1000.0, rel=1e-3)


def test_external_mask_callback_invoked(synth_cta):
    mask = _arch_mask()
    saved = {}

    def cb(name: str, arr: np.ndarray) -> None:
        saved[name] = int(arr.sum())

    compute_mandible_features(
        synth_cta, MandibleConfig(), mask, LandmarkBundle(),
        save_masks_callback=cb,
    )
    assert "mandible" in saved
    assert saved["mandible"] == int(mask.sum())


# ---------------------------------------------------------------------------
# Bone-HU fallback
# ---------------------------------------------------------------------------

def test_bone_threshold_fallback_picks_arch_over_spine():
    img = _high_hu_image_with_mandible_and_spine()
    cfg = MandibleConfig(allow_bone_threshold_fallback=True,
                          bone_hu_min=250.0, bone_min_volume_ml=3.0)
    out = compute_mandible_features(img, cfg, None, LandmarkBundle())
    assert out["mandible_mask_available"] is True
    assert "bone_threshold_largest_cc" in out["mandible_mask_method"]
    # Arch has 8 × 18 × 56 = 8064 voxels; spine has 60 × 5 × 6 = 1800.
    # Score is x_extent / z_extent: arch=56/8=7.0, spine=6/60=0.1. Arch wins.
    assert out["mandible_volume_mm3"] == pytest.approx(8064.0, rel=0.05)


def test_bone_min_volume_filter_rejects_small_candidates():
    img = _high_hu_image_with_mandible_and_spine()
    cfg = MandibleConfig(allow_bone_threshold_fallback=True,
                          bone_hu_min=250.0, bone_min_volume_ml=100.0)
    out = compute_mandible_features(img, cfg, None, LandmarkBundle())
    assert out["mandible_mask_available"] is False


# ---------------------------------------------------------------------------
# Plane resolution priority
# ---------------------------------------------------------------------------

def test_plane_from_bundle_point_and_normal_wins():
    b = LandmarkBundle()
    b.planes["mandibular_plane"] = LandmarkPlane(
        name="mandibular_plane",
        point_phys_mm=(0.0, 0.0, 0.0),
        normal_phys_mm=(0.0, 0.0, 1.0),
        source="external_json",
    )
    fake_image = CTAImage(
        array=np.zeros((10, 10, 10), dtype=np.int16),
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/x.nii.gz"),
        study_id="s", scan_id="c", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )
    plane, method = _resolve_mandibular_plane(
        b, np.zeros((10, 10, 10), dtype=bool), fake_image,
    )
    assert method == "bundle_point_normal"
    pt, normal = plane
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])


def test_plane_from_three_landmark_points(synth_cta):
    b = _bundle_with_mandible_landmarks()
    plane, method = _resolve_mandibular_plane(
        b, np.zeros(synth_cta.shape_zyx, dtype=bool), synth_cta,
    )
    assert method == "from_menton_gonion_points"
    pt, normal = plane
    # Three points all at z=70 → plane should be ~horizontal (normal ‖ z-axis)
    assert abs(normal[2]) > 0.9


def test_plane_falls_back_to_mask_extrema(synth_cta):
    """No landmarks but a mask → fallback computes plane from extrema points."""
    mask = _arch_mask()
    b = LandmarkBundle()  # no plane, no points
    plane, method = _resolve_mandibular_plane(b, mask, synth_cta)
    assert plane is not None
    assert method == "mask_inferior_extrema_heuristic"


def test_plane_unavailable_without_mask_or_landmarks(synth_cta):
    b = LandmarkBundle()
    plane, method = _resolve_mandibular_plane(
        b, np.zeros(synth_cta.shape_zyx, dtype=bool), synth_cta,
    )
    assert plane is None
    assert method == "no_landmarks_or_extrema"


# ---------------------------------------------------------------------------
# Hyoid → plane distance
# ---------------------------------------------------------------------------

def test_point_to_plane_distance_axis_aligned():
    """A point 25 mm above an XY plane at z=0 → distance = 25 mm."""
    img = CTAImage(
        array=np.zeros((50, 50, 50), dtype=np.int16),
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/x.nii.gz"),
        study_id="s", scan_id="c", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )
    plane_point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    # Voxel (25, 0, 0) maps to physical (0, 0, 25) under identity affine
    d = _point_to_plane_distance(img, (25, 0, 0), (plane_point, plane_normal))
    assert d == pytest.approx(25.0, abs=1e-6)


def test_hyoid_to_mandibular_plane_distance_full_pipeline(synth_cta):
    mask = _arch_mask()
    bundle = _bundle_with_mandible_landmarks()
    out = compute_mandible_features(synth_cta, MandibleConfig(), mask, bundle)
    assert out["mandibular_plane_available"] is True
    # Hyoid at voxel (40, 50, 40), physical (40, 50, 40). Plane points all at
    # z=70 → plane is at z=70 in physical, so distance = |70 - 40| = 30.
    assert out["mandibular_plane_to_hyoid_distance_mm"] == pytest.approx(30.0, abs=1.0)
    assert out["hyoid_to_mandible_distance_mm"] == \
        out["mandibular_plane_to_hyoid_distance_mm"]


# ---------------------------------------------------------------------------
# Oral cavity
# ---------------------------------------------------------------------------

def test_oral_cavity_external_mask(synth_cta):
    mask = _arch_mask()
    oc = np.zeros(synth_cta.shape_zyx, dtype=bool)
    oc[40:60, 25:45, 20:60] = True  # 20 × 20 × 40 = 16000 voxels
    out = compute_mandible_features(
        synth_cta, MandibleConfig(), mask, LandmarkBundle(),
        oral_cavity_mask=oc, oral_cavity_cfg=OralCavityConfig(),
    )
    assert out["oral_cavity_mask_available"] is True
    assert out["oral_cavity_method"] == "external"
    assert out["oral_cavity_volume_ml"] == pytest.approx(16000.0 / 1000.0, rel=1e-3)


def test_oral_cavity_missing_when_no_mask(synth_cta):
    mask = _arch_mask()
    out = compute_mandible_features(
        synth_cta, MandibleConfig(), mask, LandmarkBundle(),
        oral_cavity_cfg=OralCavityConfig(),
    )
    assert out["oral_cavity_mask_available"] is False
    assert out["oral_cavity_method"] == "absent"
    assert math.isnan(out["oral_cavity_volume_ml"])
