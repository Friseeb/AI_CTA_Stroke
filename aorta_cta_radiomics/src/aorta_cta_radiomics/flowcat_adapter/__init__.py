"""Typed, safe adapter for outputs from FLOWCAT-CV/arterial.

The adapter consumes already-generated FLOWCAT artifacts; it does not create a
second arterial segmentation or centerline pipeline. Pickle-backed NetworkX
graphs and object NumPy arrays are never opened without ``trusted_source=True``.
"""

from .centerlines import branches_to_image_physical, load_centerline_segments, load_numpy_array
from .coordinate_systems import (
    assert_geometry_compatible,
    flowcat_relative_to_image_physical,
    image_physical_to_flowcat_relative,
    lpi_corner_ras_mm,
    lpi_corner_voxel,
    lps_to_ras,
    ras_to_lps,
    validate_orientation,
    validate_points_within_image,
    voxel_to_ras_physical,
)
from .discovery import (
    DEFAULT_REQUIRED_OUTPUTS,
    build_output_manifest,
    create_and_write_output_manifest,
    discover_flowcat_files,
    sha256_file,
    validate_required_outputs,
    write_output_manifest,
)
from .exceptions import (
    FlowcatAdapterError,
    FlowcatGeometryError,
    FlowcatSchemaError,
    MissingFlowcatOutputError,
    MissingOptionalDependencyError,
    UnsafeArtifactError,
)
from .graphs import build_typed_vessel_graph, load_networkx_graph, validate_graph_schema
from .labels import (
    FLOWCAT_ARTERY_CLASSES,
    FLOWCAT_CODE_TO_CLASS_ID,
    artery_identity_from_flowcat,
    graph_cell_id_identities,
    map_graph_labels_to_branches,
)
from .landmarks import load_landmarks_ras_mm
from .provenance import collect_dependency_versions, collect_processing_provenance
from .schemas import (
    ArteryIdentity,
    Bifurcation,
    CaseMetadata,
    CenterlineBranch,
    CenterlinePoint,
    CoordinateReference,
    FlowcatArtifact,
    FlowcatOutputManifest,
    ImageGeometry,
    LocalGeometricFeature,
    ProcessingProvenance,
    QCFlag,
    QCSeverity,
    VesselGraph,
)
from .segmentation import BinaryNifti, geometry_from_nifti, load_binary_nifti
from .surfaces import load_polydata, load_vtk_polydata
from .validation import ValidatedFlowcatCase, validate_flowcat_case

__all__ = [
    "ArteryIdentity",
    "Bifurcation",
    "BinaryNifti",
    "CaseMetadata",
    "CenterlineBranch",
    "CenterlinePoint",
    "CoordinateReference",
    "DEFAULT_REQUIRED_OUTPUTS",
    "FLOWCAT_ARTERY_CLASSES",
    "FLOWCAT_CODE_TO_CLASS_ID",
    "FlowcatAdapterError",
    "FlowcatArtifact",
    "FlowcatGeometryError",
    "FlowcatOutputManifest",
    "FlowcatSchemaError",
    "ImageGeometry",
    "LocalGeometricFeature",
    "MissingFlowcatOutputError",
    "MissingOptionalDependencyError",
    "ProcessingProvenance",
    "QCFlag",
    "QCSeverity",
    "UnsafeArtifactError",
    "ValidatedFlowcatCase",
    "VesselGraph",
    "artery_identity_from_flowcat",
    "assert_geometry_compatible",
    "branches_to_image_physical",
    "build_output_manifest",
    "build_typed_vessel_graph",
    "collect_dependency_versions",
    "collect_processing_provenance",
    "create_and_write_output_manifest",
    "discover_flowcat_files",
    "flowcat_relative_to_image_physical",
    "geometry_from_nifti",
    "graph_cell_id_identities",
    "image_physical_to_flowcat_relative",
    "load_binary_nifti",
    "load_centerline_segments",
    "load_landmarks_ras_mm",
    "load_networkx_graph",
    "load_numpy_array",
    "load_polydata",
    "load_vtk_polydata",
    "lpi_corner_ras_mm",
    "lpi_corner_voxel",
    "lps_to_ras",
    "map_graph_labels_to_branches",
    "ras_to_lps",
    "sha256_file",
    "validate_flowcat_case",
    "validate_graph_schema",
    "validate_orientation",
    "validate_points_within_image",
    "validate_required_outputs",
    "voxel_to_ras_physical",
    "write_output_manifest",
]
