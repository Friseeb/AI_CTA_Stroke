"""Landmarks: schema round-trips, validation, voxel↔physical, heuristic
provider chain.

We pin down the load/save contract (every unknown name is silently dropped,
no exception), the validator's warning behaviour, the voxel-to-physical
helper for non-identity origins/direction, and the priority order in
`build_landmark_bundle` (explicit JSON wins; dental fills gaps; heuristic
only fires when airway is present and tall enough).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stroke_cta_osa.landmark_schema import (
    LandmarkBundle, LandmarkPlane, LandmarkPoint, LandmarkZLevel,
    PLANE_LANDMARKS, POINT_LANDMARKS, Z_LEVEL_LANDMARKS,
)
from stroke_cta_osa.landmarks import (
    build_landmark_bundle, estimate_from_airway, fill_physical_coords,
    get_hyoid_position, get_mandibular_plane, get_retroglossal_level,
    get_retropalatal_level, get_tongue_base_level,
    infer_region_levels_from_landmarks, load_landmarks, save_landmarks,
    transform_landmarks_between_image_spaces, validate_landmarks,
    voxel_to_physical, _physical_to_voxel,
)
from stroke_cta_osa.types import AirwayMaskInfo, CTAImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_bundle_for_synth() -> LandmarkBundle:
    b = LandmarkBundle(case_id="stu_synth",
                       coord_system="voxel_zyx",
                       image_shape_zyx=(80, 80, 80))
    b.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=(20, 44, 40),
        source="external_json", confidence=0.9,
    )
    b.points["menton"] = LandmarkPoint(
        name="menton", voxel_zyx=(70, 30, 40), physical_mm=(40.0, 30.0, 70.0),
        source="external_json", confidence=0.9,
    )
    b.points["gonion_left"] = LandmarkPoint(
        name="gonion_left", voxel_zyx=(65, 35, 25), physical_mm=(25.0, 35.0, 65.0),
        source="external_json", confidence=0.9,
    )
    b.points["gonion_right"] = LandmarkPoint(
        name="gonion_right", voxel_zyx=(65, 35, 55), physical_mm=(55.0, 35.0, 65.0),
        source="external_json", confidence=0.9,
    )
    b.z_levels["retropalatal_level"] = LandmarkZLevel(
        name="retropalatal_level", z_voxel=55, source="external_json",
        confidence=0.8,
    )
    b.z_levels["retroglossal_level"] = LandmarkZLevel(
        name="retroglossal_level", z_voxel=30, source="external_json",
        confidence=0.8,
    )
    b.z_levels["tongue_base_level"] = LandmarkZLevel(
        name="tongue_base_level", z_voxel=35, source="external_json",
        confidence=0.7,
    )
    b.planes["mandibular_plane"] = LandmarkPlane(
        name="mandibular_plane",
        point_names=("menton", "gonion_left", "gonion_right"),
        source="external_json",
    )
    return b


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

def test_canonical_names_are_unique_across_groups():
    all_names = set(POINT_LANDMARKS) | set(Z_LEVEL_LANDMARKS) | set(PLANE_LANDMARKS)
    total = len(POINT_LANDMARKS) + len(Z_LEVEL_LANDMARKS) + len(PLANE_LANDMARKS)
    assert len(all_names) == total, \
        "landmark names must not collide across point/z-level/plane groups"


def test_landmark_groups_have_expected_anchors():
    """A handful of named landmarks must remain in the canonical set
    because downstream features hardcode them."""
    for required in ("hyoid_centroid", "menton",
                     "gonion_left", "gonion_right",
                     "posterior_nasal_spine", "epiglottis_tip"):
        assert required in POINT_LANDMARKS
    for required in ("retropalatal_level", "retroglossal_level",
                     "tongue_base_level"):
        assert required in Z_LEVEL_LANDMARKS
    assert "mandibular_plane" in PLANE_LANDMARKS


# ---------------------------------------------------------------------------
# Load / save round-trip
# ---------------------------------------------------------------------------

def test_save_then_load_roundtrip(tmp_path: Path):
    b = _build_bundle_for_synth()
    p = save_landmarks(b, tmp_path / "lm.json")
    assert p.is_file()

    loaded = load_landmarks(p)
    assert loaded.case_id == "stu_synth"
    assert loaded.image_shape_zyx == (80, 80, 80)
    assert "hyoid_centroid" in loaded.points
    assert loaded.points["hyoid_centroid"].voxel_zyx == (20, 44, 40)
    assert loaded.z_levels["retropalatal_level"].z_voxel == 55
    assert loaded.planes["mandibular_plane"].point_names == \
        ("menton", "gonion_left", "gonion_right")


def test_loader_silently_drops_unknown_names(tmp_path: Path):
    """Forward compatibility: unknown landmark keys are skipped, not crashed."""
    data = {
        "case_id": "stu_x", "coord_system": "mixed",
        "image_shape_zyx": [80, 80, 80], "image_affine": None,
        "points": {
            "hyoid_centroid": {"voxel_zyx": [20, 44, 40],
                                "physical_mm": None,
                                "source": "external_json",
                                "confidence": 0.9},
            "totally_made_up_landmark": {"voxel_zyx": [1, 2, 3]},
        },
        "z_levels": {"retropalatal_level": {"z_voxel": 55},
                     "bogus_level": {"z_voxel": 10}},
        "planes": {"mandibular_plane": {"point_names":
                                          ["menton", "gonion_left",
                                           "gonion_right"]},
                   "bogus_plane": {"point_names": ["a", "b", "c"]}},
        "notes": "",
    }
    path = tmp_path / "lm.json"
    path.write_text(json.dumps(data))

    loaded = load_landmarks(path)
    assert "hyoid_centroid" in loaded.points
    assert "totally_made_up_landmark" not in loaded.points
    assert "retropalatal_level" in loaded.z_levels
    assert "bogus_level" not in loaded.z_levels
    assert "mandibular_plane" in loaded.planes
    assert "bogus_plane" not in loaded.planes


def test_load_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_landmarks(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_clean_bundle_returns_no_warnings(synth_cta):
    b = _build_bundle_for_synth()
    # Fill physical mm so every check passes
    fill_physical_coords(b, synth_cta)
    warnings = validate_landmarks(b, image=synth_cta)
    assert warnings == []


def test_validate_flags_shape_mismatch(synth_cta):
    b = _build_bundle_for_synth()
    b.image_shape_zyx = (99, 99, 99)
    warnings = validate_landmarks(b, image=synth_cta)
    assert any("image_shape_zyx mismatch" in w for w in warnings)


def test_validate_flags_point_out_of_bounds(synth_cta):
    b = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    b.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=(999, 0, 0),
    )
    warnings = validate_landmarks(b, image=synth_cta)
    assert any("outside image shape" in w for w in warnings)


def test_validate_flags_empty_point():
    b = LandmarkBundle()
    b.points["hyoid_centroid"] = LandmarkPoint(name="hyoid_centroid")
    warnings = validate_landmarks(b)
    assert any("neither voxel_zyx nor physical_mm" in w for w in warnings)


def test_validate_flags_plane_references_missing_points():
    b = LandmarkBundle()
    b.planes["mandibular_plane"] = LandmarkPlane(
        name="mandibular_plane",
        point_names=("menton", "gonion_left", "gonion_right"),
    )
    warnings = validate_landmarks(b)
    assert any("references missing points" in w for w in warnings)


def test_validate_never_raises_on_garbage():
    """The validator's whole point is to be non-fatal."""
    b = LandmarkBundle()
    b.points["epiglottis_tip"] = LandmarkPoint(name="epiglottis_tip")
    # Should not raise even with image=None
    warnings = validate_landmarks(b, image=None)
    assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def test_voxel_to_physical_identity_affine(synth_cta):
    """With identity direction and zero origin, voxel index = physical mm
    (with axis swap z,y,x → x,y,z)."""
    px, py, pz = voxel_to_physical(synth_cta, (10, 20, 30))
    assert (px, py, pz) == (30.0, 20.0, 10.0)


def test_voxel_to_physical_nontrivial_origin_spacing():
    img = CTAImage(
        array=np.zeros((40, 40, 40), dtype=np.int16),
        spacing_xyz_mm=(0.5, 0.6, 0.7),
        origin_xyz_mm=(100.0, 200.0, 300.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/synth.nii.gz"),
        study_id="s", scan_id="c", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )
    px, py, pz = voxel_to_physical(img, (10, 20, 30))
    assert pytest.approx(px, abs=1e-6) == 100.0 + 0.5 * 30
    assert pytest.approx(py, abs=1e-6) == 200.0 + 0.6 * 20
    assert pytest.approx(pz, abs=1e-6) == 300.0 + 0.7 * 10


def test_physical_to_voxel_is_inverse_of_voxel_to_physical():
    img = CTAImage(
        array=np.zeros((40, 40, 40), dtype=np.int16),
        spacing_xyz_mm=(0.5, 0.6, 0.7),
        origin_xyz_mm=(100.0, 200.0, 300.0),
        direction_3x3=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        source_path=Path("/tmp/synth.nii.gz"),
        study_id="s", scan_id="c", orientation_code="LPS",
        is_contrast_enhanced=False, sidecar={},
    )
    phys = voxel_to_physical(img, (5, 6, 7))
    back = _physical_to_voxel(img, phys)
    assert back == (5, 6, 7)


def test_fill_physical_coords_populates_missing(synth_cta):
    b = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    b.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=(20, 44, 40),
    )
    b.z_levels["retropalatal_level"] = LandmarkZLevel(
        name="retropalatal_level", z_voxel=55,
    )
    fill_physical_coords(b, synth_cta)
    assert b.points["hyoid_centroid"].physical_mm == (40.0, 44.0, 20.0)
    assert b.z_levels["retropalatal_level"].z_physical_mm == 55.0


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def test_region_level_accessors_pull_from_bundle():
    b = _build_bundle_for_synth()
    levels = infer_region_levels_from_landmarks(b)
    assert levels["retropalatal_level"] == 55
    assert levels["retroglossal_level"] == 30
    assert levels["tongue_base_level"] == 35
    assert levels["hyoid_level"] == 20

    assert get_retropalatal_level(b) == 55
    assert get_retroglossal_level(b) == 30
    assert get_tongue_base_level(b) == 35
    assert get_hyoid_position(b) == (20, 44, 40)


def test_region_level_falls_back_to_point_when_zlevel_missing():
    """retroglossal_level should fall back to epiglottis_tip's z."""
    b = LandmarkBundle()
    b.points["epiglottis_tip"] = LandmarkPoint(
        name="epiglottis_tip", voxel_zyx=(25, 44, 40),
    )
    assert get_retroglossal_level(b) == 25


def test_get_mandibular_plane_requires_usable_points():
    b = LandmarkBundle()
    # No representation -> None
    b.planes["mandibular_plane"] = LandmarkPlane(name="mandibular_plane")
    assert get_mandibular_plane(b) is None

    # With three points whose physical_mm is populated -> returns the plane
    b2 = _build_bundle_for_synth()
    assert get_mandibular_plane(b2) is not None


# ---------------------------------------------------------------------------
# Heuristic estimator (estimate_from_airway)
# ---------------------------------------------------------------------------

def _synth_tube_mask(shape=(80, 80, 80)) -> np.ndarray:
    """Vertical tube z=8..76, radius 5 voxels at (y=44, x=40).

    Tall enough (68 slices ≥ 60 mm at 1 mm spacing) that the heuristic
    estimator's guard rails accept it.
    """
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    tube_xy = ((yy - 44) ** 2 + (xx - 40) ** 2) <= 25
    z_band = (zz >= 8) & (zz < 76)
    return np.broadcast_to(tube_xy & z_band, shape).copy()


def test_estimate_from_airway_populates_retroglossal_and_hard_palate(synth_cta):
    mask = _synth_tube_mask()
    airway = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                            confidence="medium", notes="")
    b = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    estimate_from_airway(b, synth_cta, airway)

    # Tube has uniform CSA so the min is somewhere in the lower 2/3 — should populate
    assert "retroglossal_level" in b.z_levels
    rg = b.z_levels["retroglossal_level"].z_voxel
    assert 8 <= rg < 76
    assert b.z_levels["retroglossal_level"].source == "heuristic_airway"
    # hard_palate_plane in upper 1/3 of the airway extent
    assert "hard_palate_plane" in b.z_levels
    hp = b.z_levels["hard_palate_plane"].z_voxel
    assert 8 <= hp <= 8 + (75 - 8) // 3 + 1


def test_estimate_from_airway_skips_short_airway(synth_cta):
    """A tiny airway should NOT trigger heuristic landmark population."""
    arr = np.zeros((80, 80, 80), dtype=bool)
    arr[20:25, 44, 40] = True  # 5-slice "airway" — way below 60 mm
    airway = AirwayMaskInfo(mask_zyx=arr, method="external_mask",
                            confidence="medium", notes="")
    b = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    estimate_from_airway(b, synth_cta, airway)
    assert "retroglossal_level" not in b.z_levels
    assert "hard_palate_plane" not in b.z_levels


def test_estimate_from_airway_no_overwrite_existing_levels(synth_cta):
    """Heuristic must not overwrite an external landmark when overwrite=False."""
    mask = _synth_tube_mask()
    airway = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                            confidence="medium", notes="")
    b = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    b.z_levels["retroglossal_level"] = LandmarkZLevel(
        name="retroglossal_level", z_voxel=42,
        source="external_json", confidence=0.95,
    )
    estimate_from_airway(b, synth_cta, airway, overwrite=False)
    assert b.z_levels["retroglossal_level"].z_voxel == 42
    assert b.z_levels["retroglossal_level"].source == "external_json"


def test_estimate_from_airway_absent_airway_is_noop(synth_cta):
    airway = AirwayMaskInfo(
        mask_zyx=np.zeros((80, 80, 80), dtype=bool),
        method="absent", confidence="none", notes="",
    )
    b = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    out = estimate_from_airway(b, synth_cta, airway)
    assert out.z_levels == {}


# ---------------------------------------------------------------------------
# Provider chain (build_landmark_bundle)
# ---------------------------------------------------------------------------

def test_provider_chain_explicit_wins_over_dental_and_heuristic(
    synth_cta, tmp_path,
):
    explicit = _build_bundle_for_synth()
    explicit.z_levels["retroglossal_level"] = LandmarkZLevel(
        name="retroglossal_level", z_voxel=99, source="external_json",
    )
    explicit_path = save_landmarks(explicit, tmp_path / "explicit.json")

    dental = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    dental.z_levels["retroglossal_level"] = LandmarkZLevel(
        name="retroglossal_level", z_voxel=11, source="dental_adapter",
    )
    dental_path = save_landmarks(dental, tmp_path / "dental.json")

    mask = _synth_tube_mask()
    airway = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                            confidence="medium", notes="")
    bundle = build_landmark_bundle(
        synth_cta,
        explicit_path=explicit_path,
        dental_landmarks_path=dental_path,
        airway=airway,
        allow_heuristic_fallback=True,
    )
    # Explicit value (99) must win over dental (11) and heuristic.
    # But note: 99 is out of bounds for shape 80, which is fine — the test is
    # purely about provenance order.
    assert bundle.z_levels["retroglossal_level"].z_voxel == 99


def test_provider_chain_dental_fills_gaps_left_by_explicit(synth_cta, tmp_path):
    explicit = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    explicit.points["hyoid_centroid"] = LandmarkPoint(
        name="hyoid_centroid", voxel_zyx=(20, 44, 40), source="external_json",
    )
    explicit_path = save_landmarks(explicit, tmp_path / "explicit.json")

    dental = LandmarkBundle(image_shape_zyx=synth_cta.shape_zyx)
    dental.points["menton"] = LandmarkPoint(
        name="menton", voxel_zyx=(70, 30, 40), source="dental_adapter",
    )
    dental.z_levels["retropalatal_level"] = LandmarkZLevel(
        name="retropalatal_level", z_voxel=55, source="dental_adapter",
    )
    dental_path = save_landmarks(dental, tmp_path / "dental.json")

    bundle = build_landmark_bundle(
        synth_cta,
        explicit_path=explicit_path,
        dental_landmarks_path=dental_path,
        airway=None,
        allow_heuristic_fallback=False,
    )
    assert bundle.points["hyoid_centroid"].source == "external_json"
    assert bundle.points["menton"].source == "dental_adapter"
    assert bundle.z_levels["retropalatal_level"].z_voxel == 55


def test_provider_chain_heuristic_only_when_no_landmarks_supplied(synth_cta):
    mask = _synth_tube_mask()
    airway = AirwayMaskInfo(mask_zyx=mask, method="external_mask",
                            confidence="medium", notes="")
    bundle = build_landmark_bundle(
        synth_cta, explicit_path=None, dental_landmarks_path=None,
        airway=airway, allow_heuristic_fallback=True,
    )
    assert "retroglossal_level" in bundle.z_levels
    assert bundle.z_levels["retroglossal_level"].source == "heuristic_airway"


def test_provider_chain_returns_empty_bundle_when_nothing_available(synth_cta):
    bundle = build_landmark_bundle(
        synth_cta, explicit_path=None, dental_landmarks_path=None,
        airway=None, allow_heuristic_fallback=True,
    )
    assert bundle.points == {}
    assert bundle.z_levels == {}
    assert bundle.image_shape_zyx == synth_cta.shape_zyx


def test_provider_chain_survives_corrupt_explicit_file(synth_cta, tmp_path):
    """A broken explicit JSON should fall through to dental/heuristic, not crash."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    bundle = build_landmark_bundle(
        synth_cta, explicit_path=bad,
        dental_landmarks_path=None, airway=None,
        allow_heuristic_fallback=False,
    )
    assert bundle.points == {}


# ---------------------------------------------------------------------------
# transform_landmarks_between_image_spaces
# ---------------------------------------------------------------------------

def test_transform_landmarks_between_isotropic_images(synth_cta):
    """If two images share the same physical frame, voxel indices round-trip."""
    img2 = CTAImage(
        array=np.zeros((40, 40, 40), dtype=np.int16),
        spacing_xyz_mm=(2.0, 2.0, 2.0),  # half-resolution
        origin_xyz_mm=(0.0, 0.0, 0.0),
        direction_3x3=synth_cta.direction_3x3,
        source_path=Path("/tmp/img2.nii.gz"),
        study_id=synth_cta.study_id, scan_id="scn_low",
        orientation_code=synth_cta.orientation_code,
        is_contrast_enhanced=False, sidecar={},
    )
    b = _build_bundle_for_synth()
    fill_physical_coords(b, synth_cta)
    out = transform_landmarks_between_image_spaces(b, synth_cta, img2)
    # hyoid was at voxel (20, 44, 40) in 1mm space → (10, 22, 20) in 2mm space
    assert out.points["hyoid_centroid"].voxel_zyx == (10, 22, 20)
