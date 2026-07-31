import numpy as np

from aorta_cta_radiomics.perivascular.coordinates import (
    VesselCoordinateSystem,
    resample_polyline,
    round_trip_error_mm,
)


def test_straight_cylinder_coordinate_backbone_and_round_trip():
    original = np.array([[10.0, 20.0, 0.0], [10.0, 20.0, 12.3]])
    system = VesselCoordinateSystem.from_polyline(
        original,
        vessel_id="left_cca",
        interval_mm=1.0,
        branch_points_xyz=np.array([[10.0, 20.0, 6.0]]),
    )

    assert np.isclose(system.path_mm[-1], 12.3)
    assert np.allclose(system.points_xyz[0], original[0])
    assert np.allclose(system.points_xyz[-1], original[-1])
    assert np.allclose(system.tangents_xyz, [0.0, 0.0, 1.0])
    assert np.allclose(system.curvature_per_mm, 0.0)
    assert system.bifurcation_samples(1.0).any()

    path = np.array([1.2, 4.5, 9.7])
    radius = np.array([0.5, 2.0, 1.3])
    angle = np.array([0.0, np.pi / 2.0, 1.7 * np.pi])
    physical = system.vessel_to_physical(path, radius, angle)
    recovered = system.physical_to_vessel(physical)
    assert np.allclose(recovered.path_mm, path, atol=1e-8)
    assert np.allclose(recovered.radial_mm, radius, atol=1e-8)
    assert np.allclose(np.exp(1j * recovered.angle_rad), np.exp(1j * angle), atol=1e-8)
    assert np.max(round_trip_error_mm(system, physical)) < 1e-8


def test_parallel_transport_frames_are_stable_and_right_handed_on_curve():
    theta = np.linspace(0.0, np.pi / 2.0, 30)
    curved = np.column_stack((20.0 * np.cos(theta), 20.0 * np.sin(theta), 2.0 * theta))
    system = VesselCoordinateSystem.from_polyline(curved, interval_mm=0.75)

    assert np.allclose(np.linalg.norm(system.tangents_xyz, axis=1), 1.0, atol=1e-7)
    assert np.allclose(np.linalg.norm(system.normals_xyz, axis=1), 1.0, atol=1e-7)
    assert np.allclose(np.linalg.norm(system.binormals_xyz, axis=1), 1.0, atol=1e-7)
    assert np.max(np.abs(np.sum(system.tangents_xyz * system.normals_xyz, axis=1))) < 1e-7
    assert np.allclose(
        np.cross(system.tangents_xyz, system.normals_xyz), system.binormals_xyz, atol=1e-7
    )
    # A rotation-minimising frame has no arbitrary adjacent 180-degree flips.
    assert np.min(np.sum(system.normals_xyz[1:] * system.normals_xyz[:-1], axis=1)) > 0.9
    assert np.nanmedian(system.curvature_per_mm) > 0

    vessel_points = system.vessel_to_physical(
        system.path_mm[2:-2],
        np.full(len(system.path_mm) - 4, 0.25),
        np.linspace(0.0, 2.0 * np.pi, len(system.path_mm) - 4, endpoint=False),
    )
    assert np.percentile(round_trip_error_mm(system, vessel_points), 95) < 2e-3


def test_polyline_resampling_is_physical_and_rejects_invalid_interval():
    points, path = resample_polyline(np.array([[0, 0, 0], [0, 0, 5]], dtype=float), 2.0)
    assert np.allclose(path, [0.0, 2.0, 4.0, 5.0])
    assert np.allclose(points[:, 2], path)
    with np.testing.assert_raises(ValueError):
        resample_polyline(points, 0.0)
