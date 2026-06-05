"""Soft palate, uvula, palatine tonsils, lateral pharyngeal wall.

We exercise:
  * each external mask block independently (each is optional);
  * landmark-only soft-palate length fallback (PNS → uvula tip distance);
  * lateral wall thickness over a synthetic airway-in-body slice (the airway
    sits in the centre, body extends laterally, expected thickness is the gap
    between the two);
  * asymmetry index direction.
"""

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

from stroke_cta_osa.landmark_schema import (
    LandmarkBundle, LandmarkPoint, LandmarkZLevel,
)
from stroke_cta_osa.soft_palate import (
    SoftTissueConfig, compute_soft_palate_features, _lateral_wall_thickness,
)
from stroke_cta_osa.types import AirwayMaskInfo


def _bundle_with_rp_level(z=50, pns=(55, 30, 40), uvula=(60, 50, 40)):
    b = LandmarkBundle(image_shape_zyx=(80, 80, 80))
    b.z_levels["retropalatal_level"] = LandmarkZLevel(
        name="retropalatal_level", z_voxel=z,
    )
    b.points["posterior_nasal_spine"] = LandmarkPoint(
        name="posterior_nasal_spine", voxel_zyx=pns,
        physical_mm=(float(pns[2]), float(pns[1]), float(pns[0])),
    )
    b.points["uvula_tip"] = LandmarkPoint(
        name="uvula_tip", voxel_zyx=uvula,
        physical_mm=(float(uvula[2]), float(uvula[1]), float(uvula[0])),
    )
    return b


def _slab(shape, z_lo, z_hi, y_lo, y_hi, x_lo, x_hi) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = True
    return out


# ---------------------------------------------------------------------------
# Empty / disabled
# ---------------------------------------------------------------------------

def test_disabled_returns_empty_row(synth_cta):
    cfg = SoftTissueConfig(enabled=False)
    out = compute_soft_palate_features(synth_cta, cfg)
    assert out["soft_palate_mask_available"] is False
    assert math.isnan(out["soft_palate_volume_ml"])


def test_no_inputs_keeps_everything_unavailable(synth_cta):
    cfg = SoftTissueConfig()
    out = compute_soft_palate_features(synth_cta, cfg)
    assert out["soft_palate_mask_available"] is False
    assert out["uvula_visible"] is False
    assert out["palatine_tonsil_left_visible"] is False
    assert out["palatine_tonsil_right_visible"] is False
    assert math.isnan(out["soft_palate_length_mm"])


# ---------------------------------------------------------------------------
# Soft palate
# ---------------------------------------------------------------------------

def test_soft_palate_mask_populates_volume_length_thickness(synth_cta):
    sp = _slab(synth_cta.shape_zyx, 50, 60, 30, 38, 35, 50)
    cfg = SoftTissueConfig()
    out = compute_soft_palate_features(synth_cta, cfg, soft_palate_mask=sp)
    assert out["soft_palate_mask_available"] is True
    # 10 × 8 × 15 = 1200 voxels = 1.2 mL
    assert out["soft_palate_volume_ml"] == pytest.approx(1.2, rel=1e-3)
    # z extent = 10 voxels × 1 mm = 10 mm
    assert out["soft_palate_length_mm"] == pytest.approx(10.0, abs=0.5)
    # thickness max = y extent = 8 voxels × 1 mm = 8 mm
    assert out["soft_palate_thickness_max_mm"] == pytest.approx(8.0, abs=0.5)


def test_soft_palate_landmark_length_fallback(synth_cta):
    """No mask → length from PNS-to-uvula physical distance."""
    bundle = _bundle_with_rp_level(pns=(50, 30, 40), uvula=(60, 35, 40))
    # PNS phys = (40, 30, 50), uvula phys = (40, 35, 60)
    # distance = sqrt(0² + 5² + 10²) = sqrt(125) ≈ 11.18 mm
    cfg = SoftTissueConfig()
    out = compute_soft_palate_features(synth_cta, cfg, landmarks=bundle)
    assert out["soft_palate_mask_available"] is False
    assert out["soft_palate_length_mm"] == pytest.approx(11.18, abs=0.05)


def test_soft_palate_landmark_fallback_disabled(synth_cta):
    bundle = _bundle_with_rp_level()
    cfg = SoftTissueConfig(allow_landmark_length_fallback=False)
    out = compute_soft_palate_features(synth_cta, cfg, landmarks=bundle)
    assert math.isnan(out["soft_palate_length_mm"])


def test_soft_palate_callback_called(synth_cta):
    sp = _slab(synth_cta.shape_zyx, 50, 60, 30, 38, 35, 50)
    saved = []
    compute_soft_palate_features(synth_cta, SoftTissueConfig(),
                                  soft_palate_mask=sp,
                                  save_masks_callback=lambda n, m: saved.append(n))
    assert "soft_palate" in saved


# ---------------------------------------------------------------------------
# Uvula
# ---------------------------------------------------------------------------

def test_uvula_mask_populates_volume_and_dimensions(synth_cta):
    uv = _slab(synth_cta.shape_zyx, 58, 64, 33, 36, 38, 42)
    out = compute_soft_palate_features(synth_cta, SoftTissueConfig(),
                                        uvula_mask=uv)
    assert out["uvula_visible"] is True
    # 6 × 3 × 4 = 72 voxels = 0.072 mL
    assert out["uvula_volume_ml"] == pytest.approx(0.072, rel=1e-3)
    assert out["uvula_length_mm"] == pytest.approx(6.0, abs=0.5)
    assert out["uvula_width_mm"] == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# Palatine tonsils
# ---------------------------------------------------------------------------

def test_tonsil_left_only_populates_left_volume(synth_cta):
    tl = _slab(synth_cta.shape_zyx, 50, 60, 30, 40, 25, 35)
    out = compute_soft_palate_features(synth_cta, SoftTissueConfig(),
                                        palatine_tonsil_left_mask=tl)
    assert out["palatine_tonsil_left_visible"] is True
    assert out["palatine_tonsil_right_visible"] is False
    expected_ml = 10 * 10 * 10 / 1000.0
    assert out["palatine_tonsil_left_volume_ml"] == pytest.approx(expected_ml, rel=1e-3)
    # total should equal left when right is missing
    assert out["palatine_tonsil_total_volume_ml"] == pytest.approx(expected_ml, rel=1e-3)


def test_tonsil_both_sides_sum_to_total(synth_cta):
    tl = _slab(synth_cta.shape_zyx, 50, 60, 30, 40, 25, 35)
    tr = _slab(synth_cta.shape_zyx, 50, 60, 30, 40, 45, 55)
    out = compute_soft_palate_features(
        synth_cta, SoftTissueConfig(),
        palatine_tonsil_left_mask=tl, palatine_tonsil_right_mask=tr,
    )
    assert out["palatine_tonsil_left_visible"] is True
    assert out["palatine_tonsil_right_visible"] is True
    assert out["palatine_tonsil_total_volume_ml"] == pytest.approx(
        out["palatine_tonsil_left_volume_ml"]
        + out["palatine_tonsil_right_volume_ml"],
        abs=1e-4,
    )


# ---------------------------------------------------------------------------
# Lateral pharyngeal wall thickness
# ---------------------------------------------------------------------------

def _airway_at_centre(shape=(80, 80, 80)) -> np.ndarray:
    """Solid airway column z=40..60, y=35..45, x=35..45 (10x10x20)."""
    return _slab(shape, 40, 60, 35, 45, 35, 45)


def _body_filling(shape=(80, 80, 80)) -> np.ndarray:
    """Body spans wide laterally so there's wall thickness either side."""
    return _slab(shape, 30, 70, 20, 60, 15, 65)


def test_lateral_wall_thickness_finds_left_and_right(synth_cta):
    airway = _airway_at_centre()
    body = _body_filling()
    bundle = _bundle_with_rp_level(z=50)
    out = _lateral_wall_thickness(
        image=synth_cta, airway_mask=airway, body_mask=body,
        band_mm=15.0, window_mm=20.0, landmarks=bundle,
    )
    # Body extends to x=14 and x=64 (from 15..65 → max idx 64).
    # Airway L wall at x=35, R wall at x=44.
    # Left thickness: voxels x=34..20 where body=True → that's 15 voxels = 15 mm.
    # But band_voxels = 15 → search x_lo=20, x_hi=59. Counts body voxels along
    # the band excluding x in airway extent. Body covers x=15..64, so the
    # leftward sweep finds body in x ∈ [20..34] → 15 voxels = 15 mm.
    assert out["lateral_pharyngeal_wall_left_thickness_mm"] == pytest.approx(15.0, abs=1.0)
    assert out["lateral_pharyngeal_wall_right_thickness_mm"] == pytest.approx(15.0, abs=1.0)
    # Symmetric so asymmetry ≈ 0
    assert abs(out["lateral_pharyngeal_wall_asymmetry_index"]) < 0.05


def test_lateral_wall_thickness_returns_empty_without_landmarks(synth_cta):
    airway = _airway_at_centre()
    body = _body_filling()
    out = _lateral_wall_thickness(
        image=synth_cta, airway_mask=airway, body_mask=body,
        band_mm=15.0, window_mm=20.0, landmarks=None,
    )
    assert out == {}


def test_lateral_wall_thickness_returns_empty_without_airway_at_level(synth_cta):
    """If the airway mask is empty within the axial window, no measurements."""
    bundle = _bundle_with_rp_level(z=50)
    out = _lateral_wall_thickness(
        image=synth_cta,
        airway_mask=np.zeros(synth_cta.shape_zyx, dtype=bool),
        body_mask=_body_filling(),
        band_mm=15.0, window_mm=20.0, landmarks=bundle,
    )
    assert out == {}


def test_lateral_wall_thickness_asymmetry_sign():
    """Asymmetry should be positive when right wall is thicker than left."""
    shape = (80, 80, 80)
    airway = _slab(shape, 40, 60, 35, 45, 35, 45)
    # Body extends much further right than left
    body = _slab(shape, 30, 70, 20, 60, 30, 70)
    bundle = _bundle_with_rp_level(z=50)
    from stroke_cta_osa.types import CTAImage
    img = CTAImage(
        array=np.zeros(shape, dtype=np.int16),
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/x.nii.gz"),
        study_id="s", scan_id="c", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )
    out = _lateral_wall_thickness(
        image=img, airway_mask=airway, body_mask=body,
        band_mm=30.0, window_mm=20.0, landmarks=bundle,
    )
    # Body x ∈ [30..69]; airway x ∈ [35..44].
    # Left sweep: body voxels at x=34..30 = 5 → 5 mm
    # Right sweep: body voxels at x=45..69 = 25 → 25 mm
    # Right > Left so asymmetry > 0.
    assert out["lateral_pharyngeal_wall_right_thickness_mm"] > \
        out["lateral_pharyngeal_wall_left_thickness_mm"]
    assert out["lateral_pharyngeal_wall_asymmetry_index"] > 0


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

def test_compute_soft_palate_features_full_pipeline_passes_body_mask(synth_cta):
    """End-to-end smoke: mask + airway + body + landmarks → lateral wall block
    populates downstream too."""
    airway = _airway_at_centre()
    body = _body_filling()
    info = AirwayMaskInfo(mask_zyx=airway, method="external_mask",
                           confidence="medium", notes="")
    out = compute_soft_palate_features(
        synth_cta, SoftTissueConfig(),
        soft_palate_mask=_slab(synth_cta.shape_zyx, 50, 60, 30, 38, 35, 50),
        landmarks=_bundle_with_rp_level(z=50),
        airway=info, body_mask=body,
    )
    assert out["soft_palate_mask_available"] is True
    assert out["lateral_pharyngeal_wall_mean_thickness_mm"] > 0
