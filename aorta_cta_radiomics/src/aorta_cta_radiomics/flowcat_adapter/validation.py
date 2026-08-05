"""End-to-end contract validation for already-generated FLOWCAT artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .centerlines import load_centerline_segments
from .discovery import (
    DEFAULT_REQUIRED_OUTPUTS,
    build_output_manifest,
    discover_flowcat_files,
    validate_required_outputs,
)
from .graphs import load_networkx_graph
from .labels import map_graph_labels_to_branches
from .schemas import (
    CenterlineBranch,
    FlowcatOutputManifest,
    ImageGeometry,
    ProcessingProvenance,
)
from .segmentation import BinaryNifti, geometry_from_nifti, load_binary_nifti
from .surfaces import load_polydata


@dataclass(frozen=True, slots=True)
class ValidatedFlowcatCase:
    manifest: FlowcatOutputManifest
    cta_geometry: ImageGeometry | None
    segmentation: BinaryNifti
    branches: tuple[CenterlineBranch, ...]
    segments_graph_pred: Any
    local_graph: Any | None = None
    branch_model: Any | None = None


def validate_flowcat_case(
    case_directory: str | Path,
    *,
    case_id: str | None = None,
    cta_path: str | Path | None = None,
    provenance: ProcessingProvenance | None = None,
    mode: str = "extracranial_vessels",
    required: Sequence[str] = DEFAULT_REQUIRED_OUTPUTS,
    trusted_source: bool = False,
    convert_centerlines_to_image_physical: bool = True,
    load_vtk: bool = False,
) -> ValidatedFlowcatCase:
    """Validate the core FLOWCAT contracts and map graph labels to branches.

    Full validation necessarily opens FLOWCAT's pickle-backed graph and object
    NPY outputs. It therefore retains the safe default and requires callers to
    affirm ``trusted_source=True`` for a controlled local case directory.
    ``load_vtk`` remains separately opt-in because VTK is an optional package.
    ``cta_path`` may point to the original CTA outside the FLOWCAT output tree;
    its geometry is then used to validate the segmentation and reconstruct
    centreline coordinates without copying or linking patient data.
    """

    files = discover_flowcat_files(case_directory, mode=mode)
    validate_required_outputs(files, required)
    manifest = build_output_manifest(
        case_directory,
        case_id=case_id,
        mode=mode,
        required=required,
        provenance=provenance,
    )
    resolved_cta = (
        Path(cta_path).expanduser().resolve()
        if cta_path is not None
        else files.get("cta")
    )
    if resolved_cta is not None and not resolved_cta.is_file():
        raise FileNotFoundError(f"CTA NIfTI not found: {resolved_cta}")
    cta_geometry = geometry_from_nifti(resolved_cta) if resolved_cta is not None else None
    if resolved_cta is not None:
        manifest = replace(
            manifest,
            case=replace(manifest.case, cta_path=str(resolved_cta)),
        )
    segmentation = load_binary_nifti(files["segmentation"], reference_geometry=cta_geometry)
    branches = load_centerline_segments(
        files["centerline_segments_array"],
        trusted_source=trusted_source,
        geometry=cta_geometry,
        convert_to_image_physical=convert_centerlines_to_image_physical and cta_geometry is not None,
    )
    segments_graph = load_networkx_graph(
        files["segments_graph_pred"],
        trusted_source=trusted_source,
        graph_kind="segments_pred",
    )
    mapped_branches = map_graph_labels_to_branches(segments_graph, branches)
    local_graph = None
    if "local_graph" in files:
        local_graph = load_networkx_graph(
            files["local_graph"],
            trusted_source=trusted_source,
            graph_kind="local",
        )
    branch_model = load_polydata(files["branch_model"]) if load_vtk else None
    return ValidatedFlowcatCase(
        manifest=manifest,
        cta_geometry=cta_geometry,
        segmentation=segmentation,
        branches=mapped_branches,
        segments_graph_pred=segments_graph,
        local_graph=local_graph,
        branch_model=branch_model,
    )
