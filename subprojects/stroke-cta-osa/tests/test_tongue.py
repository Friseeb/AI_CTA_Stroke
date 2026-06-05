"""Tongue features on synthetic data.

Synthetic geometry: we build a known tongue ellipsoid mask with a known
volume (so the volume → mL conversion is testable), drop in a contrived
posterior third (for HU stats), and exercise the two fallback paths:

  * mask present, no landmarks  → global volume + posterior third works
  * no mask, landmarks + airway → landmark posterior box works
  * no mask, no landmarks       → every tongue feature missing, qc_pass False

We also verify the tongue/mandible and tongue/oral-cavity volume ratios
when the caller passes in those volumes.
"""

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

from stroke_cta_osa.landmark_schema import (
    LandmarkBundle, LandmarkPoint, LandmarkZLevel,
)
from stroke_cta_osa.tongue import (
    TongueConfig, compute_tongue_features,
    _landmark_posterior_tongue_box, _posterior_third_of_mask,
    _tongue_base_band, _tongue_base_airway_displacements,
)
from stroke_cta_osa.types import AirwayMaskInfo, CTAImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tongue_mask(shape=(80, 80, 80)) -> np.ndarray:
    """Ellipsoid mask placed in clean body soft tissue, avoiding airway tube
    (centred at y=44, x=40) and L/R parapharyngeal fat slabs.

    Centred at (z=50, y=30, x=30) with radii (6, 6, 6): volume ≈ (4/3)*π*6³
    ≈ 905 mm³.
    """
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = 50, 30, 30
    rz, ry, rx = 6, 6, 6
    return ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _tube_mask(shape=(80, 80, 80)) -> np.ndarray:
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    tube_xy = ((yy - 50) ** 2 + (xx - 40) ** 2) <= 25
    z_band = (zz >= 8) & (zz < 76)
    return np.broadcast_to(tube_xy & z_band, shape).copy()


def _bundle_with_levels(rg_z=40, base_z=48, hyoid_zyx=(20, 50, 40)):
    b = LandmarkBundle(image_shape_zyx=(80, 80, 80))
    b.z_levels["retroglossal_level"] = LandmarkZLevel(
        name="retroglossal_level", z_voxel=rg_z, source="external_json",
    )
    b.z_levels["tongue_base_level"] = LandmarkZLevel(
        name="tongue_base_level", z_voxel=base_z, source="external_json",
    )
    b.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=hyoid_zyx, source="external_json",
    )
    return b


# ---------------------------------------------------------------------------
# Empty-row + disabled config
# ---------------------------------------------------------------------------

def test_disabled_config_returns_empty_row(synth_cta):
    cfg = TongueConfig(enabled=False)
    out = compute_tongue_features(synth_cta, cfg, None, LandmarkBundle())
    assert out["tongue_qc_failure_reasons"] == "tongue_module_disabled"
    assert out["tongue_mask_available"] is False
    assert math.isnan(out["tongue_volume_ml"])


def test_missing_mask_and_no_fallback_returns_failure(synth_cta):
    cfg = TongueConfig(allow_posterior_roi_fallback=False)
    out = compute_tongue_features(synth_cta, cfg, None, _bundle_with_levels())
    assert out["tongue_mask_available"] is False
    assert out["tongue_qc_failure_reasons"] == "no_tongue_mask_and_fallback_disabled"
    assert math.isnan(out["tongue_volume_ml"])


def test_missing_mask_and_no_landmarks_failure(synth_cta):
    cfg = TongueConfig()
    out = compute_tongue_features(synth_cta, cfg, None, LandmarkBundle())
    # No airway either → no anchor → failure
    assert out["tongue_mask_available"] is False
    assert "no_tongue_mask_and_no_landmark_box" in str(out["tongue_qc_failure_reasons"])
    assert out["tongue_posterior_roi_available"] is False


# ---------------------------------------------------------------------------
# Mask-driven path
# ---------------------------------------------------------------------------

def test_tongue_volume_from_mask(synth_cta):
    mask = _tongue_mask()
    cfg = TongueConfig()
    n_vox = int(mask.sum())
    expected_ml = n_vox * 1.0 / 1000.0  # 1 mm³ voxels
    out = compute_tongue_features(synth_cta, cfg, mask, _bundle_with_levels())
    assert out["tongue_mask_available"] is True
    assert out["tongue_qc_pass"] is True
    assert out["tongue_volume_ml"] == pytest.approx(expected_ml, rel=1e-3)
    # ellipsoid volume formula sanity check: should be within 10% of formula
    formula_ml = (4.0 / 3.0) * math.pi * 6 * 6 * 6 / 1000.0
    assert abs(out["tongue_volume_ml"] - formula_ml) / formula_ml < 0.10


def test_tongue_hu_stats_from_mask(synth_cta):
    """The synth CTA body voxels are 40 HU; the tongue ellipsoid is fully inside
    the body, so mean_hu should be exactly 40 with std 0."""
    mask = _tongue_mask()
    cfg = TongueConfig()
    out = compute_tongue_features(synth_cta, cfg, mask, _bundle_with_levels())
    assert out["tongue_mean_hu"] == pytest.approx(40.0, abs=1.0)
    assert out["tongue_median_hu"] == pytest.approx(40.0, abs=1.0)
    # All voxels at 40 HU; low_hu_fraction with default threshold 30 should be 0.
    assert out["tongue_low_hu_fraction"] == 0.0


def test_tongue_posterior_third_is_subset_of_mask(synth_cta):
    """The posterior-third sub-ROI must contain non-zero volume and its volume
    must be roughly 1/3 of the full mask."""
    mask = _tongue_mask()
    posterior, method = _posterior_third_of_mask(mask)
    assert posterior.any()
    assert "posterior_third_y" in method
    ratio = posterior.sum() / mask.sum()
    # An ellipsoid sliced at the posterior third should give ~10–20% of total
    # (the central slab is the widest).
    assert 0.05 < ratio < 0.45


def test_tongue_posterior_hu_block_populated(synth_cta):
    mask = _tongue_mask()
    cfg = TongueConfig()
    out = compute_tongue_features(synth_cta, cfg, mask, _bundle_with_levels())
    assert out["tongue_posterior_roi_available"] is True
    assert out["tongue_posterior_volume_ml"] > 0
    # Same HU as global since body == 40 HU everywhere.
    assert out["tongue_posterior_mean_hu"] == pytest.approx(40.0, abs=1.0)


def test_tongue_base_band_with_landmark(synth_cta):
    mask = _tongue_mask()
    bundle = _bundle_with_levels(base_z=52)  # within tongue z extent (44..56)
    band = _tongue_base_band(synth_cta, mask, bundle)
    assert band.any()
    zs = np.where(band.any(axis=(1, 2)))[0]
    # 10 mm window around z=52 at 1 mm spacing
    assert int(zs.min()) >= 42
    assert int(zs.max()) <= 62


def test_tongue_base_band_without_landmark_uses_inferior_third(synth_cta):
    """Without a landmark, the band should be the inferior 1/3 of the mask z extent."""
    mask = _tongue_mask()
    bundle = LandmarkBundle()  # no levels
    band = _tongue_base_band(synth_cta, mask, bundle)
    assert band.any()
    full_zs = np.where(mask.any(axis=(1, 2)))[0]
    band_zs = np.where(band.any(axis=(1, 2)))[0]
    # inferior third → larger-z slab
    assert int(band_zs.min()) >= int(full_zs.min()) + (full_zs.size * 2 // 3) - 1


def test_tongue_base_area_at_retroglossal_with_airway(synth_cta):
    """When base band overlaps the retroglossal slice, area is populated."""
    mask = _tongue_mask()
    tube = _tube_mask()
    airway = AirwayMaskInfo(mask_zyx=tube, method="external_mask",
                            confidence="medium", notes="")
    # Put rg_z within both tongue (z=44..56) and tube (8..75) extents
    bundle = _bundle_with_levels(rg_z=50, base_z=52)
    cfg = TongueConfig()
    out = compute_tongue_features(synth_cta, cfg, mask, bundle, airway=airway)
    assert out["tongue_base_area_at_retroglossal_level_mm2"] > 0
    # Airway adjacency populates too
    assert out["retroglossal_airway_area_adjacent_to_tongue_base_mm2"] > 0


def test_tongue_to_mandible_and_oral_cavity_ratios(synth_cta):
    mask = _tongue_mask()
    cfg = TongueConfig()
    n_vox = int(mask.sum())
    tongue_ml = n_vox / 1000.0
    out = compute_tongue_features(
        synth_cta, cfg, mask, _bundle_with_levels(),
        mandible_volume_ml=20.0, oral_cavity_volume_ml=50.0,
    )
    # Ratios are rounded to 4 decimals → tolerate 1e-3 absolute.
    assert out["tongue_to_mandible_volume_ratio"] == \
        pytest.approx(tongue_ml / 20.0, abs=1e-3)
    assert out["tongue_to_oral_cavity_volume_ratio"] == \
        pytest.approx(tongue_ml / 50.0, abs=1e-3)


def test_tongue_save_masks_callback_invoked(synth_cta):
    mask = _tongue_mask()
    saved = {}

    def cb(name: str, arr: np.ndarray) -> None:
        saved[name] = arr.sum()

    out = compute_tongue_features(
        synth_cta, TongueConfig(), mask, _bundle_with_levels(),
        save_masks_callback=cb,
    )
    assert "tongue" in saved
    assert "tongue_posterior" in saved
    assert saved["tongue"] == int(mask.sum())
    assert saved["tongue_posterior"] > 0


# ---------------------------------------------------------------------------
# Landmark-fallback path (no mask)
# ---------------------------------------------------------------------------

def test_landmark_posterior_box_with_airway_and_landmarks(synth_cta):
    tube = _tube_mask()
    airway = AirwayMaskInfo(mask_zyx=tube, method="external_mask",
                            confidence="medium", notes="")
    bundle = _bundle_with_levels(rg_z=40, base_z=45)
    box, method, confidence = _landmark_posterior_tongue_box(
        synth_cta, bundle, airway,
    )
    assert box is not None
    assert box.any()
    assert confidence == "low"
    assert "landmark_box" in method


def test_landmark_posterior_box_without_airway_returns_none(synth_cta):
    bundle = _bundle_with_levels()
    box, method, confidence = _landmark_posterior_tongue_box(
        synth_cta, bundle, airway=None,
    )
    assert box is None
    assert method == "no_anchor"


def test_fallback_path_populates_posterior_hu_stats(synth_cta):
    """No tongue mask, but airway + landmarks → landmark posterior box gives
    HU stats and qc_pass = True (because no failure reasons accumulated)."""
    tube = _tube_mask()
    airway = AirwayMaskInfo(mask_zyx=tube, method="external_mask",
                            confidence="medium", notes="")
    bundle = _bundle_with_levels(rg_z=40, base_z=45)
    cfg = TongueConfig()
    out = compute_tongue_features(synth_cta, cfg, None, bundle, airway=airway)
    assert out["tongue_mask_available"] is False
    assert out["tongue_posterior_roi_available"] is True
    assert out["tongue_roi_confidence"] == "low"
    assert out["tongue_posterior_volume_ml"] > 0
    assert out["tongue_qc_pass"] is True


# ---------------------------------------------------------------------------
# Displacement helper
# ---------------------------------------------------------------------------

def test_tongue_base_displacements_against_airway(synth_cta):
    mask = _tongue_mask()
    bundle = _bundle_with_levels(base_z=52)
    base_band = _tongue_base_band(synth_cta, mask, bundle)
    tube = _tube_mask()
    disp = _tongue_base_airway_displacements(synth_cta, base_band, tube)
    # Tongue mask centred at y=30, radius 6 → max y ≈ 36;
    # tube centred at y=50, radius 5 → min y ≈ 45.
    assert not math.isnan(disp["tongue_base_posterior_displacement_mm"])
    # tongue base maximum z is around 56; airway min z is 8 → diff > 0
    assert disp["tongue_base_inferior_displacement_mm"] > 0


def test_tongue_base_displacements_with_empty_inputs(synth_cta):
    out = _tongue_base_airway_displacements(
        synth_cta,
        np.zeros((80, 80, 80), dtype=bool),
        np.zeros((80, 80, 80), dtype=bool),
    )
    assert math.isnan(out["tongue_base_posterior_displacement_mm"])
    assert math.isnan(out["tongue_base_inferior_displacement_mm"])
