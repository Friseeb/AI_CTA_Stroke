from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import nibabel as nib
import networkx as nx
import numpy as np
import pytest

from aorta_cta_radiomics.flowcat_adapter import (
    CoordinateReference,
    FlowcatGeometryError,
    FlowcatSchemaError,
    MissingFlowcatOutputError,
    MissingOptionalDependencyError,
    UnsafeArtifactError,
    artery_identity_from_flowcat,
    assert_geometry_compatible,
    build_output_manifest,
    build_typed_vessel_graph,
    discover_flowcat_files,
    flowcat_relative_to_image_physical,
    geometry_from_nifti,
    image_physical_to_flowcat_relative,
    load_binary_nifti,
    load_centerline_segments,
    load_networkx_graph,
    load_numpy_array,
    load_polydata,
    lpi_corner_ras_mm,
    lpi_corner_voxel,
    lps_to_ras,
    map_graph_labels_to_branches,
    ras_to_lps,
    validate_graph_schema,
    validate_flowcat_case,
    validate_orientation,
    validate_required_outputs,
    write_output_manifest,
)


MODE = "extracranial_vessels"


def _touch(path: Path, content: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _centerline_object_array() -> np.ndarray:
    array = np.empty((2, 2), dtype=object)
    array[0, 0] = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    array[0, 1] = np.asarray([2.0, 1.8, 1.6])
    array[1, 0] = np.asarray([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    array[1, 1] = np.asarray([1.6, 1.2])
    return array


def _predicted_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(0, 1, cell_id=0, vessel_type=3, vessel_type_name="RCCA")
    graph.add_edge(1, 2, cell_id=1, vessel_type=9, vessel_type_name="RICA")
    return graph


def test_discovery_supports_nested_and_documented_root_segmentation(tmp_path):
    nested = _touch(tmp_path / MODE / "segmentation.nii.gz")
    _touch(tmp_path / f"{MODE}_segmentation.nii.gz", b"legacy")
    branch = _touch(tmp_path / MODE / "branch_model.vtk")
    _touch(tmp_path / MODE / "segmentation.vtk")
    _touch(tmp_path / MODE / "segmentation.stl")
    _touch(tmp_path / MODE / "centerlines" / "centerlines_0.vtk")
    _touch(tmp_path / MODE / "landmarks" / "landmarks.json", b"{}")

    found = discover_flowcat_files(tmp_path)

    assert found["segmentation"] == nested
    assert found["branch_model"] == branch
    assert found["segmentation_vtk"].name == "segmentation.vtk"
    assert found["segmentation_stl"].name == "segmentation.stl"
    assert found["centerline_vtk[0]"].name == "centerlines_0.vtk"
    assert found["landmarks"].as_posix().endswith("landmarks/landmarks.json")


def test_discovery_falls_back_to_root_segmentation(tmp_path):
    root_segmentation = _touch(tmp_path / f"{MODE}_segmentation.nii.gz")
    assert discover_flowcat_files(tmp_path)["segmentation"] == root_segmentation


def test_missing_required_outputs_are_reported_together(tmp_path):
    found = discover_flowcat_files(tmp_path)
    with pytest.raises(MissingFlowcatOutputError, match="branch_model.*segmentation"):
        validate_required_outputs(found, required=("segmentation", "branch_model"))


def test_manifest_hashes_files_records_missing_and_writes_json(tmp_path):
    segmentation = _touch(tmp_path / MODE / "segmentation.nii.gz", b"mask")
    manifest = build_output_manifest(
        tmp_path,
        case_id="case-001",
        required=("segmentation", "branch_model"),
    )
    by_name = {artifact.name: artifact for artifact in manifest.artifacts}
    assert by_name["segmentation"].sha256 == (
        "48bf9b4f142ab45cac6f33cf99415bca6ca8e07608a031237aa76560fec2204b"
    )
    assert by_name["segmentation"].path == str(segmentation)
    assert by_name["branch_model"].validation_status == "missing"
    assert manifest.missing_required_outputs == ("branch_model",)

    output = write_output_manifest(manifest, tmp_path / "adapter" / "manifest.json")
    payload = json.loads(output.read_text())
    assert payload["case"]["case_id"] == "case-001"
    assert payload["artifacts"][0]["coordinate_reference"]


def test_numeric_npy_is_safe_by_default_but_object_npy_requires_trust(tmp_path):
    numeric_path = tmp_path / "numeric.npy"
    object_path = tmp_path / "object.npy"
    np.save(numeric_path, np.arange(4))
    np.save(object_path, _centerline_object_array())
    np.testing.assert_array_equal(load_numpy_array(numeric_path), np.arange(4))
    with pytest.raises(UnsafeArtifactError, match="trusted_source=True"):
        load_numpy_array(object_path)
    assert load_numpy_array(object_path, trusted_source=True).dtype == object


def test_centerline_loader_validates_schema_and_retains_relative_reference(tmp_path):
    path = tmp_path / "centerline_segments_array.npy"
    np.save(path, _centerline_object_array())
    branches = load_centerline_segments(path, trusted_source=True, processing_version="1991ca5")

    assert [branch.cell_id for branch in branches] == [0, 1]
    assert branches[0].coordinate_reference is CoordinateReference.FLOWCAT_LPI_CORNER_RELATIVE_MM
    assert branches[0].points[-1].path_distance_mm == pytest.approx(2.0)
    assert branches[0].points[1].tangent == pytest.approx((1.0, 0.0, 0.0))
    assert branches[0].points[0].local_radius_mm == 2.0
    assert branches[0].points[0].source_file == str(path.resolve())


def test_centerline_loader_rejects_schema_drift(tmp_path):
    bad = np.empty((1, 2), dtype=object)
    bad[0, 0] = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bad[0, 1] = np.asarray([1.0])
    path = tmp_path / "centerline_segments_array.npy"
    np.save(path, bad)
    with pytest.raises(FlowcatSchemaError, match="radii must have shape"):
        load_centerline_segments(path, trusted_source=True)


def test_graph_pickle_requires_explicit_trust_and_validates_predicted_schema(tmp_path):
    path = tmp_path / "segments_graph_pred.pickle"
    with path.open("wb") as stream:
        pickle.dump(_predicted_graph(), stream)
    with pytest.raises(UnsafeArtifactError, match="trusted_source=True"):
        load_networkx_graph(path)
    loaded = load_networkx_graph(path, trusted_source=True)
    assert loaded.number_of_edges() == 2


def test_graph_schema_rejects_class_code_mismatch_and_duplicate_cell_ids():
    mismatch = nx.Graph()
    mismatch.add_edge(0, 1, cell_id=0, vessel_type=3, vessel_type_name="LCCA")
    with pytest.raises(FlowcatSchemaError, match="class 3 is 'RCCA'"):
        validate_graph_schema(mismatch, graph_kind="segments_pred")

    duplicate = nx.Graph()
    duplicate.add_edge(0, 1, cell_id=0)
    duplicate.add_edge(1, 2, cell_id=0)
    with pytest.raises(FlowcatSchemaError, match="Duplicate centerline cell_id 0"):
        validate_graph_schema(duplicate, graph_kind="segments")


def test_local_graph_schema_requires_geometry_radius_and_14_class_label():
    graph = nx.Graph()
    graph.add_node(0, pos=np.asarray([0, 0, 0]), radius=1.0, cell_id=0, vessel_type=13, vessel_type_name="BA")
    graph.add_node(1, pos=np.asarray([1, 0, 0]), radius=1.0, cell_id=0, vessel_type=13, vessel_type_name="BA")
    graph.add_edge(0, 1)
    validate_graph_schema(graph, graph_kind="local")
    del graph.nodes[0]["radius"]
    with pytest.raises(FlowcatSchemaError, match="missing 'radius'"):
        validate_graph_schema(graph, graph_kind="local")


def test_graph_cell_id_mapping_uses_flowcat_14_class_convention(tmp_path):
    path = tmp_path / "centerline_segments_array.npy"
    np.save(path, _centerline_object_array())
    branches = load_centerline_segments(path, trusted_source=True)
    mapped = map_graph_labels_to_branches(_predicted_graph(), branches)
    assert mapped[0].artery == artery_identity_from_flowcat(3, code="RCCA")
    assert mapped[1].artery.flowcat_code == "RICA"
    assert all(point.artery == mapped[0].artery for point in mapped[0].points)


def test_graph_mapping_rejects_orphan_graph_cell_id(tmp_path):
    path = tmp_path / "centerline_segments_array.npy"
    array = _centerline_object_array()[:1]
    np.save(path, array)
    graph = _predicted_graph()
    with pytest.raises(FlowcatSchemaError, match="absent centerline cell_id values: 1"):
        map_graph_labels_to_branches(graph, load_centerline_segments(path, trusted_source=True))


def test_typed_vessel_graph_localizes_degree_three_bifurcation(tmp_path):
    array = np.empty((3, 2), dtype=object)
    array[0] = (np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float), np.asarray([2.0, 1.8]))
    array[1] = (np.asarray([[1, 0, 0], [2, 1, 0]], dtype=float), np.asarray([1.8, 1.2]))
    array[2] = (np.asarray([[1, 0, 0], [2, -1, 0]], dtype=float), np.asarray([1.8, 1.1]))
    path = tmp_path / "centerline_segments_array.npy"
    np.save(path, array)
    branches = load_centerline_segments(path, trusted_source=True)
    graph = nx.Graph()
    graph.add_node(1, pos=np.asarray([1.0, 0.0, 0.0]))
    graph.add_edge(0, 1, cell_id=0, vessel_type=3, vessel_type_name="RCCA")
    graph.add_edge(1, 2, cell_id=1, vessel_type=9, vessel_type_name="RICA")
    graph.add_edge(1, 3, cell_id=2, vessel_type=11, vessel_type_name="RECA")

    typed = build_typed_vessel_graph(graph, branches)

    assert len(typed.branches) == 3
    assert len(typed.bifurcations) == 1
    assert typed.bifurcations[0].coordinates_mm == (1.0, 0.0, 0.0)
    assert typed.bifurcations[0].connected_branch_ids == ("cell-0", "cell-1", "cell-2")


def test_binary_nifti_preserves_affine_spacing_and_orientation(tmp_path):
    data = np.zeros((4, 5, 6), dtype=np.uint8)
    data[1:3, 2, 3] = 1
    affine = np.asarray(
        [[0.7, 0.0, 0.0, 10.0], [0.0, 1.2, 0.0, 20.0], [0.0, 0.0, 2.5, -30.0], [0, 0, 0, 1]]
    )
    path = tmp_path / "segmentation.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    loaded = load_binary_nifti(path)
    assert loaded.data.dtype == bool
    assert loaded.data.sum() == 2
    assert loaded.geometry.shape_ijk == (4, 5, 6)
    assert loaded.geometry.spacing_mm == pytest.approx((0.7, 1.2, 2.5))
    assert loaded.geometry.orientation == ("R", "A", "S")
    np.testing.assert_allclose(np.asarray(loaded.geometry.affine_ras_mm), affine)


def test_binary_nifti_rejects_nonbinary_and_reference_geometry_mismatch(tmp_path):
    affine = np.eye(4)
    path = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray([[[0, 2]]], dtype=np.uint8), affine), path)
    with pytest.raises(FlowcatSchemaError, match="must be binary"):
        load_binary_nifti(path)

    good_path = tmp_path / "good.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine), good_path)
    nib.save(nib.Nifti1Image(np.zeros((3, 2, 2), dtype=np.uint8), affine), reference_path)
    reference = geometry_from_nifti(reference_path)
    with pytest.raises(FlowcatGeometryError, match="shape"):
        load_binary_nifti(good_path, reference_geometry=reference)


def test_lpi_relative_conversion_handles_lps_voxel_orientation(tmp_path):
    affine = np.asarray(
        [[-1.0, 0.0, 0.0, 50.0], [0.0, -2.0, 0.0, 80.0], [0.0, 0.0, 3.0, -30.0], [0, 0, 0, 1]]
    )
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.uint8), affine)
    geometry = geometry_from_nifti(image)
    assert geometry.orientation == ("L", "P", "S")
    assert lpi_corner_voxel(geometry) == (3, 4, 0)
    np.testing.assert_allclose(lpi_corner_ras_mm(geometry), [47.0, 72.0, -30.0])

    relative = np.asarray([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    physical = flowcat_relative_to_image_physical(relative, geometry)
    np.testing.assert_allclose(physical, [[47.0, 72.0, -30.0], [49.0, 76.0, -24.0]])
    np.testing.assert_allclose(image_physical_to_flowcat_relative(physical, geometry), relative)


def test_ras_lps_conversion_and_orientation_validation():
    points = np.asarray([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    expected = np.asarray([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    np.testing.assert_allclose(ras_to_lps(points), expected)
    np.testing.assert_allclose(lps_to_ras(expected), points)
    assert validate_orientation(("S", "R", "A")) == ("S", "R", "A")
    with pytest.raises(FlowcatGeometryError, match="one LR, AP, and IS"):
        validate_orientation(("R", "L", "S"))


def test_geometry_compatibility_checks_affine_not_only_shape(tmp_path):
    image_a = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    image_b = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), shifted)
    with pytest.raises(FlowcatGeometryError, match="affine matrices differ"):
        assert_geometry_compatible(geometry_from_nifti(image_a), geometry_from_nifti(image_b))


def test_full_validation_accepts_external_cta_without_copying_it(tmp_path):
    case_directory = tmp_path / "flowcat-case"
    output_directory = case_directory / MODE
    output_directory.mkdir(parents=True)
    cta_path = tmp_path / "original-cta.nii.gz"
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), affine), cta_path)
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[0:3, 0:2, 0] = 1
    nib.save(
        nib.Nifti1Image(segmentation, affine),
        output_directory / "segmentation.nii.gz",
    )
    _touch(output_directory / "branch_model.vtk", b"not-loaded-in-this-test")
    np.save(output_directory / "centerline_segments_array.npy", _centerline_object_array())
    with (output_directory / "segments_graph_pred.pickle").open("wb") as stream:
        pickle.dump(_predicted_graph(), stream)
    local_graph = nx.Graph()
    local_graph.add_node(
        0,
        pos=np.asarray([0.0, 0.0, 0.0]),
        radius=2.0,
        cell_id=0,
        vessel_type=3,
        vessel_type_name="RCCA",
    )
    local_graph.add_node(
        1,
        pos=np.asarray([1.0, 0.0, 0.0]),
        radius=1.8,
        cell_id=0,
        vessel_type=3,
        vessel_type_name="RCCA",
    )
    local_graph.add_edge(0, 1)
    with (output_directory / "local_graph.pickle").open("wb") as stream:
        pickle.dump(local_graph, stream)

    validated = validate_flowcat_case(
        case_directory,
        case_id="anon-case",
        cta_path=cta_path,
        trusted_source=True,
    )

    assert validated.cta_geometry is not None
    assert validated.manifest.case.cta_path == str(cta_path.resolve())
    assert validated.branches[0].artery is not None
    assert validated.branches[0].artery.flowcat_code == "RCCA"
    assert not (case_directory / "cta.nii.gz").exists()


def test_vtk_loader_has_clear_optional_dependency_error(tmp_path):
    if importlib.util.find_spec("vtk") is not None:
        pytest.skip("VTK is installed in this environment")
    path = _touch(tmp_path / "branch_model.vtk", b"# vtk DataFile Version 3.0\n")
    with pytest.raises(MissingOptionalDependencyError, match="VTK is required"):
        load_polydata(path)
