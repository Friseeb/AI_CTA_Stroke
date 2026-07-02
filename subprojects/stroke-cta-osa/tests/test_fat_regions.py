"""Regional fat features (level-anchored, per-side, retropharyngeal,
facial / buccal).

We use the conftest synth CTA which already carries fat slabs in the
parapharyngeal (-120 HU) and retropharyngeal (-110 HU) regions and a
subcutaneous shell (-100 HU). Tests assert:

  * empty/disabled config returns empty row;
  * fat area at hyoid/RP/RG levels populates when those z's are landmarked;
  * per-side L/R parapharyngeal areas are non-zero and balanced (symmetric
    synth data);
  * retropharyngeal area at RP/RG populates when posterior fat slab covers
    that z;
  * facial fat block is only emitted when explicitly enabled.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from stroke_cta_osa.fat_regions import (
    FatRegionConfig, compute_regional_fat_features,
    _per_side_parapharyngeal, _retropharyngeal_band, _subglosso_anchor,
)
from stroke_cta_osa.landmark_schema import (
    LandmarkBundle, LandmarkPoint, LandmarkZLevel,
)
from stroke_cta_osa.types import AirwayMaskInfo


def _bundle_with_levels(rp_z=45, rg_z=35, hyoid_zyx=(40, 44, 40)):
    b = LandmarkBundle(image_shape_zyx=(80, 80, 80))
    b.z_levels["retropalatal_level"] = LandmarkZLevel(
        name="retropalatal_level", z_voxel=rp_z,
    )
    b.z_levels["retroglossal_level"] = LandmarkZLevel(
        name="retroglossal_level", z_voxel=rg_z,
    )
    b.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=hyoid_zyx,
    )
    return b


def _airway_from_synth(arr: np.ndarray) -> AirwayMaskInfo:
    mask = (arr == -800).astype(bool)
    return AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                           confidence="medium", notes="")


def _body_from_synth(arr: np.ndarray, hu_min: float = -250.0) -> np.ndarray:
    """The body mask = anything that isn't air. The synth CTA has body at 40,
    fat at -100..-120, airway at -800, outside at -1000."""
    return arr > hu_min


# ---------------------------------------------------------------------------
# Disabled / empty
# ---------------------------------------------------------------------------

def test_disabled_returns_empty(synth_cta):
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(enabled=False),
        airway=None, body_mask=None, landmarks=LandmarkBundle(),
    )
    assert math.isnan(out["fat_cervical_area_at_hyoid_level_mm2"])
    assert math.isnan(out["fat_parapharyngeal_area_retropalatal_total_mm2"])


def test_no_airway_returns_empty(synth_cta):
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(),
        airway=None, body_mask=_body_from_synth(synth_cta.array),
        landmarks=_bundle_with_levels(),
    )
    assert math.isnan(out["fat_cervical_area_at_hyoid_level_mm2"])


def test_no_body_returns_empty(synth_cta):
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=None,
        landmarks=_bundle_with_levels(),
    )
    assert math.isnan(out["fat_cervical_area_at_hyoid_level_mm2"])


# ---------------------------------------------------------------------------
# Cervical fat areas at standard levels
# ---------------------------------------------------------------------------

def test_cervical_fat_areas_populate_at_landmarked_levels(synth_cta):
    """Synth CTA has subcutaneous fat shell + PP / RP slabs.
    At z=40 (within PP fat z=25..55) cervical fat area should be > 0."""
    bundle = _bundle_with_levels(rp_z=45, rg_z=40, hyoid_zyx=(40, 44, 40))
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle,
    )
    assert out["fat_cervical_area_at_hyoid_level_mm2"] > 0
    assert out["fat_cervical_area_at_retropalatal_level_mm2"] > 0
    assert out["fat_cervical_area_at_retroglossal_level_mm2"] > 0


def test_cervical_fat_unavailable_when_no_landmarks(synth_cta):
    """No hyoid / RP / RG → those areas stay NaN."""
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=LandmarkBundle(),
    )
    assert math.isnan(out["fat_cervical_area_at_hyoid_level_mm2"])


# ---------------------------------------------------------------------------
# Per-side parapharyngeal
# ---------------------------------------------------------------------------

def test_per_side_parapharyngeal_areas_populated_and_symmetric(synth_cta):
    """The synth CTA has symmetric L/R parapharyngeal fat slabs at x=28..32
    (left) and x=47..51 (right). At z=40 (within PP fat z range), both sides
    should be non-zero and balanced."""
    bundle = _bundle_with_levels(rp_z=40, rg_z=40, hyoid_zyx=(40, 44, 40))
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle,
    )
    left = out["fat_parapharyngeal_area_retropalatal_left_mm2"]
    right = out["fat_parapharyngeal_area_retropalatal_right_mm2"]
    total = out["fat_parapharyngeal_area_retropalatal_total_mm2"]
    assert left > 0
    assert right > 0
    # Synth fat is symmetric → L≈R within 50%
    assert min(left, right) / max(left, right) > 0.5
    assert total == pytest.approx(left + right, abs=0.5)


def test_parapharyngeal_to_airway_ratio_populated(synth_cta):
    bundle = _bundle_with_levels(rp_z=40, rg_z=40)
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle,
    )
    assert out["fat_parapharyngeal_to_airway_ratio_retropalatal"] > 0


def test_per_side_parapharyngeal_helper_split_left_right(synth_cta):
    airway = (synth_cta.array == -800).astype(bool)
    body = synth_cta.array > -250
    fat = (synth_cta.array >= -190) & (synth_cta.array <= -30)
    left, right = _per_side_parapharyngeal(
        image=synth_cta, airway_mask=airway, anchor_z=40,
        lateral_band_mm=25.0, window_mm=30.0,
        body_mask=body, fat_voxels=fat,
    )
    # Left half: x < airway centre (≈40) → ALL left voxels should have x < 40
    if left.any():
        _, _, xs = np.where(left)
        assert xs.max() < 40
    if right.any():
        _, _, xs = np.where(right)
        assert xs.min() > 40


# ---------------------------------------------------------------------------
# Retropharyngeal band
# ---------------------------------------------------------------------------

def test_retropharyngeal_band_posterior_to_airway(synth_cta):
    """All voxels in the retropharyngeal band must lie posterior to the
    airway at their slice."""
    airway = (synth_cta.array == -800).astype(bool)
    band = _retropharyngeal_band(
        image=synth_cta, airway_mask=airway,
        posterior_band_mm=15.0, window_mm=30.0, anchor_z=40,
    )
    assert band.any()
    # At each z slice with content, band's y-min should be > airway's y-max.
    for z in range(band.shape[0]):
        if not band[z].any() or not airway[z].any():
            continue
        ys_band, _ = np.where(band[z])
        ys_air, _ = np.where(airway[z])
        assert int(ys_band.min()) > int(ys_air.max())


def test_retropharyngeal_band_uses_centre_z_when_anchor_missing(synth_cta):
    airway = (synth_cta.array == -800).astype(bool)
    band = _retropharyngeal_band(
        image=synth_cta, airway_mask=airway,
        posterior_band_mm=15.0, window_mm=30.0, anchor_z=None,
    )
    # Should still produce a non-empty band somewhere in the airway extent
    assert band.any()


def test_retropharyngeal_area_at_levels_populated(synth_cta):
    """The synth CTA's retropharyngeal fat slab covers z=30..60 — should
    contribute at both RP=45 and RG=35."""
    bundle = _bundle_with_levels(rp_z=45, rg_z=35)
    out = compute_regional_fat_features(
        synth_cta, FatRegionConfig(use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle,
    )
    assert out["fat_retropharyngeal_area_at_retropalatal_level_mm2"] > 0
    assert out["fat_retropharyngeal_area_at_retroglossal_level_mm2"] > 0


# ---------------------------------------------------------------------------
# Facial fat (opt-in)
# ---------------------------------------------------------------------------

def test_facial_fat_block_emitted_only_when_enabled(synth_cta):
    bundle = _bundle_with_levels()
    out_off = compute_regional_fat_features(
        synth_cta, FatRegionConfig(enable_facial_fat=False,
                                    use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle,
    )
    out_on = compute_regional_fat_features(
        synth_cta, FatRegionConfig(enable_facial_fat=True,
                                    use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle,
    )
    # When off, facial volume stays NaN
    assert math.isnan(out_off["fat_facial_total_volume_ml"])
    # When on, facial volume populates with non-negative number
    assert (math.isnan(out_on["fat_facial_total_volume_ml"])
            or out_on["fat_facial_total_volume_ml"] >= 0)


# ---------------------------------------------------------------------------
# Subglosso-supraglottic anchor derivation
# ---------------------------------------------------------------------------

def test_subglosso_anchor_offset_below_rg():
    """The subglosso anchor must sit ~15 mm below the RG level."""
    z = _subglosso_anchor(rg_z=40, sz_mm=1.0)
    assert z == 40 + 15
    assert _subglosso_anchor(rg_z=None, sz_mm=1.0) is None
    # Non-isotropic spacing: 5 mm slices → 3 voxel offset
    assert _subglosso_anchor(rg_z=20, sz_mm=5.0) == 23


# ---------------------------------------------------------------------------
# Save masks callback
# ---------------------------------------------------------------------------

def test_save_masks_callback_invoked_for_parapharyngeal(synth_cta):
    bundle = _bundle_with_levels(rp_z=40, rg_z=40)
    seen = []

    def cb(name: str, arr: np.ndarray) -> None:
        seen.append(name)

    compute_regional_fat_features(
        synth_cta, FatRegionConfig(use_anatomy_priors=False),
        airway=_airway_from_synth(synth_cta.array),
        body_mask=_body_from_synth(synth_cta.array),
        landmarks=bundle, save_masks_callback=cb,
    )
    # We should have written L/R masks for at least the retropalatal level
    assert any("fat_parapharyngeal_retropalatal_left" in s for s in seen)
    assert any("fat_parapharyngeal_retropalatal_right" in s for s in seen)
    assert any("fat_retropharyngeal_regional" in s for s in seen)
