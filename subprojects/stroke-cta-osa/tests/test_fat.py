"""Cervical, parapharyngeal, and retropharyngeal fat computation."""

import math

import numpy as np
import pytest

from stroke_cta_osa.config import FatConfig, HUConfig
from stroke_cta_osa.fat import compute_fat_features
from stroke_cta_osa.shared_schema import SharedAirwayLandmarks
from stroke_cta_osa.types import AirwayMaskInfo


def _airway_info(synth_cta):
    mask = (synth_cta.array == -800).astype(bool)
    return AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                          confidence="medium", notes="")


def test_total_cervical_fat_nonzero(synth_cta):
    feats = compute_fat_features(
        synth_cta, _airway_info(synth_cta), SharedAirwayLandmarks(),
        HUConfig(), FatConfig(),
        airway_min_csa_z_index=40,
    )
    assert feats["fat_cervical_volume_ml"] > 0
    assert -190 <= feats["fat_cervical_mean_hu"] <= -30


def test_subcutaneous_and_deep_split(synth_cta):
    feats = compute_fat_features(
        synth_cta, _airway_info(synth_cta), SharedAirwayLandmarks(),
        HUConfig(), FatConfig(),
        airway_min_csa_z_index=40,
    )
    # Both compartments populated, subcutaneous fraction in (0, 1)
    assert feats["fat_subcutaneous_cervical_volume_ml"] > 0
    sub_frac = feats["fat_subcutaneous_fraction_of_neck_area"]
    assert 0 < sub_frac < 1


def test_parapharyngeal_with_airway(synth_cta):
    feats = compute_fat_features(
        synth_cta, _airway_info(synth_cta), SharedAirwayLandmarks(),
        HUConfig(), FatConfig(parapharyngeal_axial_window_mm=30.0),
        airway_min_csa_z_index=40,
    )
    assert feats["fat_parapharyngeal_total_volume_ml"] > 0
    assert feats["fat_parapharyngeal_left_volume_ml"] > 0
    assert feats["fat_parapharyngeal_right_volume_ml"] > 0
    # Synthetic data is symmetric → asymmetry index near zero
    assert abs(feats["fat_parapharyngeal_asymmetry_index"]) < 0.05
    assert feats["fat_parapharyngeal_to_airway_ratio"] > 0


def test_retropharyngeal_present(synth_cta):
    feats = compute_fat_features(
        synth_cta, _airway_info(synth_cta), SharedAirwayLandmarks(),
        HUConfig(), FatConfig(),
        airway_min_csa_z_index=40,
    )
    assert feats["fat_retropharyngeal_volume_ml"] > 0
    assert feats["fat_retropharyngeal_mean_thickness_mm"] > 0


def test_fat_features_without_airway_emits_nans(synth_cta):
    feats = compute_fat_features(
        synth_cta, None, SharedAirwayLandmarks(),
        HUConfig(), FatConfig(),
        airway_min_csa_z_index=None,
    )
    # Cervical fat still computed
    assert feats["fat_cervical_volume_ml"] > 0
    # Parapharyngeal collapses to NaN
    assert math.isnan(feats["fat_parapharyngeal_total_volume_ml"])
    assert feats["fat_parapharyngeal_roi_method"] == "unavailable_no_airway"


def test_landmark_anchored_areas(synth_cta):
    landmarks = SharedAirwayLandmarks(
        posterior_nasal_spine=(45, 44, 40),
        epiglottis_tip=(30, 44, 40),
    )
    feats = compute_fat_features(
        synth_cta, _airway_info(synth_cta), landmarks,
        HUConfig(), FatConfig(),
        airway_min_csa_z_index=40,
    )
    assert feats["fat_parapharyngeal_area_retropalatal_mm2"] >= 0
    assert feats["fat_parapharyngeal_area_retroglossal_mm2"] >= 0
