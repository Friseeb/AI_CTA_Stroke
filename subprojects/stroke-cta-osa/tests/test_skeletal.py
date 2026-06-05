"""Skeletal / hyoid geometry features.

We construct a minimal LandmarkBundle with the points we need and assert:
  * distances are computed in physical mm regardless of voxel spacing;
  * missing landmarks → NaN for the dependent feature;
  * the hyoid → airway posterior wall distance uses the airway's posterior y
    at the hyoid's z;
  * cervicomandibular ring area is a bbox proxy of the mandible's inferior
    rim;
  * mandibular_plane_to_hyoid_distance_mm passed in is forwarded into the
    output.
"""

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

from stroke_cta_osa.landmark_schema import LandmarkBundle, LandmarkPoint
from stroke_cta_osa.skeletal import (
    SkeletalConfig, compute_skeletal_features,
    _cervicomandibular_ring_area,
    _distance_hyoid_to_airway_posterior_wall,
)
from stroke_cta_osa.types import AirwayMaskInfo, CTAImage


def _bundle_with_points(**points) -> LandmarkBundle:
    b = LandmarkBundle(image_shape_zyx=(80, 80, 80))
    for name, voxel in points.items():
        b.points[name] = LandmarkPoint(
            name=name, voxel_zyx=voxel,
            physical_mm=(float(voxel[2]), float(voxel[1]), float(voxel[0])),
            source="external_json",
        )
    return b


def test_disabled_returns_empty_row(synth_cta):
    out = compute_skeletal_features(
        synth_cta, SkeletalConfig(enabled=False), LandmarkBundle(),
    )
    assert out["hyoid_detected"] is False
    assert math.isnan(out["hyoid_to_c3_distance_mm"])


def test_empty_bundle_keeps_all_features_missing(synth_cta):
    out = compute_skeletal_features(
        synth_cta, SkeletalConfig(), LandmarkBundle(),
    )
    assert out["hyoid_detected"] is False
    assert math.isnan(out["hyoid_to_c2_distance_mm"])
    assert math.isnan(out["hyoid_to_c3_distance_mm"])
    assert math.isnan(out["hyoid_to_c4_distance_mm"])
    assert math.isnan(out["neck_length_mm"])


def test_hyoid_centroid_populated_from_landmark(synth_cta):
    bundle = _bundle_with_points(hyoid_centroid=(20, 44, 40))
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    assert out["hyoid_detected"] is True
    # voxel (20, 44, 40) → physical (40, 44, 20) under identity affine
    assert out["hyoid_centroid_x_mm"] == 40.0
    assert out["hyoid_centroid_y_mm"] == 44.0
    assert out["hyoid_centroid_z_mm"] == 20.0


def test_hyoid_to_c3_distance(synth_cta):
    """Two points 10 mm apart in z → distance = 10 mm."""
    bundle = _bundle_with_points(
        hyoid_centroid=(20, 44, 40),
        c3_centroid=(30, 44, 40),
    )
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    assert out["hyoid_to_c3_distance_mm"] == pytest.approx(10.0, abs=0.01)


def test_hyoid_to_c2_c3_c4_independent():
    bundle = _bundle_with_points(
        hyoid_centroid=(20, 44, 40),
        c2_centroid=(10, 50, 40),  # ~11.7 mm
        c3_centroid=(20, 50, 40),  # 6.0 mm
        c4_centroid=(35, 50, 40),  # ~16.16 mm
    )
    img = CTAImage(
        array=np.zeros((80, 80, 80), dtype=np.int16),
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/x.nii.gz"),
        study_id="s", scan_id="c", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )
    out = compute_skeletal_features(img, SkeletalConfig(), bundle)
    assert out["hyoid_to_c2_distance_mm"] == pytest.approx(
        float(np.sqrt(10 ** 2 + 6 ** 2)), abs=0.05)
    assert out["hyoid_to_c3_distance_mm"] == pytest.approx(6.0, abs=0.05)
    assert out["hyoid_to_c4_distance_mm"] == pytest.approx(
        float(np.sqrt(15 ** 2 + 6 ** 2)), abs=0.05)


def test_neck_length_equals_hard_palate_to_hyoid_distance(synth_cta):
    bundle = _bundle_with_points(
        hyoid_centroid=(40, 44, 40),
        posterior_nasal_spine=(60, 30, 40),
    )
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    expected = float(np.sqrt(20 ** 2 + 14 ** 2 + 0))
    assert out["hard_palate_to_hyoid_distance_mm"] == pytest.approx(expected, abs=0.05)
    assert out["neck_length_mm"] == out["hard_palate_to_hyoid_distance_mm"]


def test_laryngeal_descent_equals_hyoid_to_c4(synth_cta):
    bundle = _bundle_with_points(
        hyoid_centroid=(20, 44, 40),
        c4_centroid=(35, 44, 40),
    )
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    assert out["laryngeal_descent_mm"] == out["hyoid_to_c4_distance_mm"]


def test_vertical_position_relative_to_mandible(synth_cta):
    """Hyoid z=20, menton z=70 → vertical (z) diff = -50 mm."""
    bundle = _bundle_with_points(
        hyoid_centroid=(20, 44, 40),
        menton=(70, 30, 40),
    )
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    assert out["hyoid_vertical_position_relative_to_mandible_mm"] == \
        pytest.approx(-50.0, abs=0.01)


def test_ap_position_relative_to_cervical_spine(synth_cta):
    """Hyoid y=44, C3 y=55 → AP diff = -11 mm (hyoid is anterior to C3)."""
    bundle = _bundle_with_points(
        hyoid_centroid=(20, 44, 40),
        c3_centroid=(20, 55, 40),
    )
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    assert out["hyoid_ap_position_relative_to_cervical_spine_mm"] == \
        pytest.approx(-11.0, abs=0.01)


# ---------------------------------------------------------------------------
# Hyoid → posterior pharyngeal wall distance
# ---------------------------------------------------------------------------

def test_distance_hyoid_to_airway_posterior_wall_basic(synth_cta):
    """Airway slice at z=20 has y posterior = 49 (radius 5 around y=44).
    Hyoid at y=10 → distance = 49 - 10 = 39 mm."""
    mask = np.zeros(synth_cta.shape_zyx, dtype=bool)
    mask[20, 40:50, 35:45] = True  # y posterior = 49
    d = _distance_hyoid_to_airway_posterior_wall(synth_cta, (20, 10, 40), mask)
    assert d == pytest.approx(39.0, abs=0.01)


def test_distance_hyoid_to_airway_posterior_wall_z_out_of_bounds(synth_cta):
    mask = np.zeros(synth_cta.shape_zyx, dtype=bool)
    mask[20:30, 40:50, 35:45] = True
    d = _distance_hyoid_to_airway_posterior_wall(synth_cta, (999, 10, 40), mask)
    assert d is None


def test_distance_hyoid_to_airway_posterior_wall_empty_slice(synth_cta):
    mask = np.zeros(synth_cta.shape_zyx, dtype=bool)
    d = _distance_hyoid_to_airway_posterior_wall(synth_cta, (20, 10, 40), mask)
    assert d is None


def test_distance_through_compute_pipeline(synth_cta):
    mask = np.zeros(synth_cta.shape_zyx, dtype=bool)
    mask[20, 40:50, 35:45] = True
    info = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                           confidence="medium", notes="")
    bundle = _bundle_with_points(hyoid_centroid=(20, 10, 40))
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle, airway=info)
    assert out["hyoid_to_posterior_pharyngeal_wall_distance_mm"] == \
        pytest.approx(39.0, abs=0.01)


# ---------------------------------------------------------------------------
# Cervicomandibular ring
# ---------------------------------------------------------------------------

def test_cervicomandibular_ring_uses_mandible_inferior_bbox(synth_cta):
    mand = np.zeros(synth_cta.shape_zyx, dtype=bool)
    # mandible inferior rim: z=68..73, y=20..38, x=12..68
    mand[68:74, 20:38, 12:68] = True
    bundle = _bundle_with_points(hyoid_centroid=(40, 44, 40))
    out = compute_skeletal_features(
        synth_cta, SkeletalConfig(), bundle,
        mandible_mask=mand,
    )
    assert "cervicomandibular_ring_area_mm2" in out
    assert out["cervicomandibular_ring_area_mm2"] > 0
    assert out["cervicomandibular_ring_method"] == "mandible_inferior_bbox_proxy"
    # bbox area at inferior rim ≈ y_extent * x_extent = 18 × 56 = 1008 mm²
    assert out["cervicomandibular_ring_area_mm2"] == pytest.approx(1008.0, abs=20.0)


def test_cervicomandibular_ring_returns_none_when_no_mandible(synth_cta):
    area, method = _cervicomandibular_ring_area(
        synth_cta, (40, 44, 40),
        np.zeros(synth_cta.shape_zyx, dtype=bool),
        airway=None,
    )
    assert area is None
    assert method == "no_mandible"


# ---------------------------------------------------------------------------
# Forwarded mandibular plane → hyoid distance
# ---------------------------------------------------------------------------

def test_mandibular_plane_distance_forwarded_from_caller(synth_cta):
    bundle = _bundle_with_points(hyoid_centroid=(20, 44, 40))
    out = compute_skeletal_features(
        synth_cta, SkeletalConfig(), bundle,
        mandibular_plane_to_hyoid_distance_mm=23.45,
    )
    assert out["mandibular_plane_to_hyoid_distance_mm"] == pytest.approx(23.45, abs=0.01)
    assert out["mandibular_plane_available"] is True


def test_mandibular_plane_distance_unavailable_when_not_passed(synth_cta):
    bundle = _bundle_with_points(hyoid_centroid=(20, 44, 40))
    out = compute_skeletal_features(synth_cta, SkeletalConfig(), bundle)
    assert math.isnan(out["mandibular_plane_to_hyoid_distance_mm"])
    assert out["mandibular_plane_available"] is False
