import numpy as np

from aorta_cta_radiomics.perivascular.composition import TissueLabel
from aorta_cta_radiomics.perivascular.contact import (
    DEFAULT_CONTACT_CLASSES,
    _surface_faces,
    calculate_surface_contacts,
)
from aorta_cta_radiomics.perivascular.coordinates import VesselCoordinateSystem
from aorta_cta_radiomics.perivascular.qc import perivascular_qc
from aorta_cta_radiomics.perivascular.radial_profiles import (
    aggregate_longitudinal_profiles,
    aggregate_radial_profiles,
    weighted_profile_gradient,
)
from aorta_cta_radiomics.perivascular.sectors import assign_angular_sectors


def test_sector_assignment_is_frame_based_reproducible_and_supports_8_16_32():
    shape = (21, 21, 21)
    roi = np.zeros(shape, dtype=bool)
    # Physical centreline x=10, y=10, along z.
    points_zyx = np.array([[10, 10, 15], [10, 15, 10], [10, 10, 5], [10, 5, 10]])
    roi[tuple(points_zyx.T)] = True
    system = VesselCoordinateSystem.from_polyline([[10, 10, 0], [10, 10, 20]], vessel_id="left_cca")

    first = assign_angular_sectors(roi, system, (1, 1, 1), number_of_sectors=8)
    second = assign_angular_sectors(roi, system, (1, 1, 1), number_of_sectors=8)
    assert np.array_equal(first.labels, second.labels)
    assert first.labels[10, 10, 15] == 1  # +normal (physical +x)
    assert first.labels[10, 15, 10] == 3  # +binormal (physical +y)
    assert first.labels[10, 10, 5] == 5
    assert first.labels[10, 5, 10] == 7
    assert first.orientation_reference.startswith("relative_")
    for count in (8, 16, 32):
        assigned = assign_angular_sectors(roi, system, (1, 1, 1), number_of_sectors=count)
        assert assigned.labels.max() <= count


def _contact_fixture():
    shape = (12, 15, 15)
    lumen = np.zeros(shape, dtype=bool)
    lumen[:, 5:10, 5:10] = True
    labels = np.zeros(shape, dtype=np.uint8)
    # Each z plane has 20 assessable lateral faces: 10 fat, 5 muscle, 5 soft tissue.
    labels[:, 5:10, 4] = int(TissueLabel.ADIPOSE)
    labels[:, 5:10, 10] = int(TissueLabel.ADIPOSE)
    labels[:, 4, 5:10] = int(TissueLabel.SKELETAL_MUSCLE)
    labels[:, 10, 5:10] = int(TissueLabel.OTHER_SOFT_TISSUE)
    return lumen, labels


def test_surface_contact_recovers_exact_50_25_25_and_reports_boundary_qc():
    lumen, labels = _contact_fixture()
    system = VesselCoordinateSystem.from_polyline([[7, 7, 0], [7, 7, 11]], vessel_id="right_cca")
    result = calculate_surface_contacts(
        lumen,
        labels,
        (1, 1, 1),
        distances_mm=(0.5,),
        coordinate_system=system,
    )
    contact = result.at(0.5)

    assert np.isclose(contact.classes["adipose"].contact_fraction, 0.50)
    assert np.isclose(contact.classes["skeletal_muscle"].contact_fraction, 0.25)
    assert np.isclose(contact.classes["other_soft_tissue"].contact_fraction, 0.25)
    assert contact.classes["vein"].contact_fraction == 0.0
    assert contact.boundary_missing_area_mm2 > 0  # cylinder deliberately exits z FOV
    assert "surface_truncated_by_image_boundary" in result.qc_flags
    assert len(contact.classes["adipose"].longitudinal_profile) > 1
    assert len(contact.classes["adipose"].circumferential_profile) == 8

    report = perivascular_qc(coordinate_system=system, contacts=result)
    assert report.status == "review"
    assert "surface_truncated_by_image_boundary" in report.codes


def test_vein_and_bone_contact_classes_are_separate_and_absent_class_is_not_false_positive():
    lumen, labels = _contact_fixture()
    labels[:, 4, 5:10] = int(TissueLabel.VEIN)
    labels[:, 10, 5:10] = int(TissueLabel.BONE)
    contact = calculate_surface_contacts(lumen, labels, (1, 1, 1), distances_mm=(0.5,)).at(0.5)

    assert np.isclose(contact.classes["vein"].contact_fraction, 0.25)
    assert np.isclose(contact.classes["bone"].contact_fraction, 0.25)
    assert contact.classes["skeletal_muscle"].contact_area_mm2 == 0.0
    assert contact.classes["other_soft_tissue"].contact_area_mm2 == 0.0


def test_bounded_contact_edt_matches_full_grid_reference_and_uses_local_crops(monkeypatch):
    from scipy import ndimage as ndi

    shape = (81, 83, 85)
    lumen = np.zeros(shape, dtype=bool)
    lumen[39:42, 40:43, 41:44] = True
    labels = np.zeros(shape, dtype=np.uint8)
    nearby = (
        (38, 41, 42),
        (42, 41, 42),
        (40, 39, 42),
        (40, 43, 42),
        (40, 41, 40),
        (40, 41, 44),
        (38, 40, 41),
        (42, 42, 43),
    )
    for coordinate, value in zip(nearby, DEFAULT_CONTACT_CLASSES.values()):
        labels[coordinate] = value
    # A remote class voxel must not pull the crop toward the full image extent.
    labels[3, 3, 3] = int(TissueLabel.ADIPOSE)

    original_edt = ndi.distance_transform_edt
    edt_shapes: list[tuple[int, ...]] = []

    def tracking_edt(array, *args, **kwargs):
        edt_shapes.append(tuple(array.shape))
        return original_edt(array, *args, **kwargs)

    monkeypatch.setattr(ndi, "distance_transform_edt", tracking_edt)
    distances = (0.5, 1.5, 2.0)
    result = calculate_surface_contacts(lumen, labels, (1, 1, 1), distances_mm=distances)

    faces = _surface_faces(lumen, (1, 1, 1))
    valid_indices = np.flatnonzero(faces.exterior_valid)
    outside = faces.exterior_zyx[valid_indices]
    for name, value in DEFAULT_CONTACT_CLASSES.items():
        full_distance = original_edt(~(labels == value), sampling=(1, 1, 1))
        for distance in distances:
            expected = np.zeros(len(faces.area_mm2), dtype=bool)
            expected[valid_indices] = full_distance[tuple(outside.T)] <= distance + 1.0e-9
            expected_area = float(faces.area_mm2[expected].sum())
            assert np.isclose(result.at(distance).classes[name].contact_area_mm2, expected_area)

    assert edt_shapes
    assert all(np.prod(call_shape) < 0.05 * np.prod(shape) for call_shape in edt_shapes)
    assert result.method.endswith("bounded_exterior_physical_edt")


def test_radial_and_longitudinal_profiles_preserve_local_measurements():
    shape = (6, 4, 4)
    image = np.full(shape, 20.0)
    labels = np.full(shape, int(TissueLabel.OTHER_SOFT_TISSUE), dtype=np.uint8)
    labels[:3] = int(TissueLabel.ADIPOSE)
    image[:3] = -100.0
    inner = np.zeros(shape, dtype=bool)
    inner[:, :, :2] = True
    outer = ~inner
    radial = aggregate_radial_profiles(
        image,
        labels,
        {"0_2mm": inner, "2_5mm": outer},
        (1, 1, 1),
    )
    assert len(radial) == 2
    assert all(np.isclose(record["adipose_fraction"], 0.5) for record in radial)

    path = np.broadcast_to(np.arange(6, dtype=float)[:, None, None], shape)
    longitudinal = aggregate_longitudinal_profiles(
        image,
        labels,
        np.ones(shape, dtype=bool),
        path,
        (1, 1, 1),
        bin_length_mm=1.0,
        vessel_length_mm=5.0,
    )
    assert longitudinal[0]["adipose_fraction"] == 1.0
    assert longitudinal[-1]["adipose_fraction"] == 0.0
    assert weighted_profile_gradient(longitudinal, "adipose_fraction") < 0
