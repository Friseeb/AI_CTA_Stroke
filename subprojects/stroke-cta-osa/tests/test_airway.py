"""Airway adapter + geometry behaviour on synthetic data."""

import math

import numpy as np
import pytest

from stroke_cta_osa.adapters import (CTAFallbackAirwayAdapter,
                                     ExternalMaskAdapter, NullAirwayAdapter)
from stroke_cta_osa.airway import compute_airway_features
from stroke_cta_osa.shared_schema import SharedAirwayLandmarks
from stroke_cta_osa.types import AirwayMaskInfo


def test_fallback_finds_synthetic_airway(synth_cta):
    adapter = CTAFallbackAirwayAdapter(
        air_hu_max=-500, min_component_volume_ml=0.05, closing_mm=0.0,
    )
    info = adapter.get_airway_mask(synth_cta)
    assert info is not None
    assert info.method == "threshold_connected_component"
    assert info.mask_zyx.any()
    # Should overlap heavily with the ground-truth tube (HU == -800)
    gt = synth_cta.array == -800
    overlap = (info.mask_zyx & gt).sum() / max(gt.sum(), 1)
    assert overlap > 0.7


def test_external_mask_adapter_roundtrip(synth_cta, synth_airway_mask_path):
    adapter = ExternalMaskAdapter(mask_path=synth_airway_mask_path)
    assert adapter.is_available()
    info = adapter.get_airway_mask(synth_cta)
    assert info is not None
    assert info.method == "external_mask"
    assert int(info.mask_zyx.sum()) > 100


def test_null_adapter_returns_none(synth_cta):
    adapter = NullAirwayAdapter()
    assert adapter.is_available()
    assert adapter.get_airway_mask(synth_cta) is None
    assert adapter.get_landmarks(synth_cta).hyoid is None


def test_airway_features_with_mask(synth_cta):
    mask = (synth_cta.array == -800).astype(bool)
    info = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                          confidence="medium", notes="")
    feats = compute_airway_features(
        synth_cta, info, SharedAirwayLandmarks(),
    ).features
    assert feats["airway_mask_available"] is True
    assert feats["airway_volume_ml"] > 0
    # Tube is 56 slices tall (z=15..70) → length ~56 mm at 1 mm slices
    assert 50 < feats["airway_length_mm"] < 65
    # Min CSA from a 5-vox-radius circle is ≈ 81 mm² (pi*r² + discretisation)
    assert 50 < feats["airway_min_csa_mm2"] < 110
    # Region features are NaN when no landmarks supplied
    assert math.isnan(feats["retropalatal_csa_mm2"])
    assert math.isnan(feats["retroglossal_csa_mm2"])


def test_airway_features_without_mask(synth_cta):
    feats = compute_airway_features(
        synth_cta, None, SharedAirwayLandmarks(),
    ).features
    assert feats["airway_mask_available"] is False
    assert math.isnan(feats["airway_volume_ml"])
    assert math.isnan(feats["airway_min_csa_mm2"])
    assert feats["airway_region_method"] == "unavailable"


def test_airway_features_with_landmarks_populate_regions(synth_cta):
    mask = (synth_cta.array == -800).astype(bool)
    info = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                          confidence="medium", notes="")
    landmarks = SharedAirwayLandmarks(
        posterior_nasal_spine=(60, 44, 40),  # upper z
        epiglottis_tip=(25, 44, 40),          # lower z
        hyoid=(20, 44, 40),
    )
    feats = compute_airway_features(synth_cta, info, landmarks,
                                    retropalatal_window_mm=10.0,
                                    retroglossal_window_mm=10.0).features
    assert feats["airway_region_method"] == "landmarked"
    assert feats["retropalatal_csa_mm2"] > 0
    assert feats["retroglossal_csa_mm2"] > 0
    assert feats["retropalatal_volume_ml"] > 0
