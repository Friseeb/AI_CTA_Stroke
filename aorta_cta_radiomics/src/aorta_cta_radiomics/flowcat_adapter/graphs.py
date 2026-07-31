"""Trusted NetworkX loading and FLOWCAT graph-schema validation."""

from __future__ import annotations

import pickle
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .exceptions import FlowcatSchemaError, MissingOptionalDependencyError, UnsafeArtifactError
from .labels import FLOWCAT_ARTERY_CLASSES, graph_cell_id_identities
from .labels import map_graph_labels_to_branches
from .schemas import Bifurcation, CenterlineBranch, CoordinateReference, VesselGraph


GraphKind = Literal["segments", "segments_pred", "local", "auto"]


def load_networkx_graph(
    path: str | Path,
    *,
    trusted_source: bool = False,
    graph_kind: GraphKind = "auto",
) -> Any:
    """Load and validate a trusted local FLOWCAT NetworkX pickle.

    Python pickle is executable, not a data-only interchange format. This
    function refuses to open a file unless ``trusted_source=True`` is supplied
    explicitly. Only use pickles emitted by a controlled local FLOWCAT run.
    """

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"FLOWCAT graph pickle not found: {file_path}")
    if trusted_source is not True:
        raise UnsafeArtifactError(
            f"Refusing to unpickle {file_path}. FLOWCAT graph pickles must originate "
            "from a trusted local pipeline; pass trusted_source=True explicitly."
        )
    if file_path.suffix.lower() not in {".pickle", ".pkl"}:
        raise FlowcatSchemaError(f"Expected a .pickle or .pkl FLOWCAT graph, got {file_path.name!r}")
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installations
        raise MissingOptionalDependencyError(
            "NetworkX is required to load FLOWCAT graph pickles; install `networkx`."
        ) from exc

    try:
        with file_path.open("rb") as stream:
            graph = pickle.load(stream)
    except Exception as exc:
        raise FlowcatSchemaError(f"Could not load trusted FLOWCAT graph {file_path}: {exc}") from exc
    if not isinstance(graph, (nx.Graph, nx.DiGraph)) or isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise FlowcatSchemaError(
            f"FLOWCAT graph must be a simple NetworkX Graph or DiGraph, got {type(graph).__name__}"
        )

    resolved_kind = _infer_graph_kind(file_path) if graph_kind == "auto" else graph_kind
    validate_graph_schema(graph, graph_kind=resolved_kind)
    return graph


def validate_graph_schema(graph: Any, *, graph_kind: GraphKind) -> None:
    """Validate only fields consumed by this adapter, with schema-drift errors."""

    if graph_kind == "auto":
        raise ValueError("validate_graph_schema requires an explicit graph_kind")
    if graph.number_of_nodes() == 0:
        raise FlowcatSchemaError(f"FLOWCAT {graph_kind} graph has no nodes")
    if graph.number_of_edges() == 0:
        raise FlowcatSchemaError(f"FLOWCAT {graph_kind} graph has no edges")

    if graph_kind in {"segments", "segments_pred"}:
        seen_cell_ids: set[int] = set()
        for source, target, data in graph.edges(data=True):
            if not isinstance(data, Mapping):
                raise FlowcatSchemaError(f"Edge {(source, target)!r} attributes are not a mapping")
            _require_integer(data, "cell_id", context=f"edge {(source, target)!r}", minimum=0)
            cell_id = int(data["cell_id"])
            if cell_id in seen_cell_ids:
                raise FlowcatSchemaError(f"Duplicate centerline cell_id {cell_id} in segment graph edges")
            seen_cell_ids.add(cell_id)
            coordinate_array = None
            radius_array = None
            if "coordinate_array" in data:
                coordinate_array = _require_points(
                    data["coordinate_array"],
                    context=f"edge {(source, target)!r} coordinate_array",
                )
            if "radius_array" in data:
                radius_array = _require_radii(
                    data["radius_array"],
                    context=f"edge {(source, target)!r} radius_array",
                )
            if coordinate_array is not None and radius_array is not None:
                if coordinate_array.shape[0] != radius_array.shape[0]:
                    raise FlowcatSchemaError(
                        f"Edge {(source, target)!r} coordinate/radius arrays have different lengths"
                    )
        if graph_kind == "segments_pred":
            # Also verifies code/class agreement and optional confidence ranges.
            graph_cell_id_identities(graph)
        return

    if graph_kind == "local":
        for node, data in graph.nodes(data=True):
            if not isinstance(data, Mapping):
                raise FlowcatSchemaError(f"Node {node!r} attributes are not a mapping")
            context = f"local-graph node {node!r}"
            _require_point(data.get("pos"), context=f"{context} pos")
            _require_nonnegative_real(data, "radius", context=context)
            _require_integer(data, "cell_id", context=context, minimum=0)
            vessel_type = _require_integer(data, "vessel_type", context=context, minimum=0, maximum=13)
            if "vessel_type_name" not in data:
                raise FlowcatSchemaError(f"{context} is missing 'vessel_type_name'")
            expected_code = FLOWCAT_ARTERY_CLASSES[vessel_type][0]
            if str(data["vessel_type_name"]) != expected_code:
                raise FlowcatSchemaError(
                    f"{context} class/code mismatch: {vessel_type} is {expected_code!r}, "
                    f"got {data['vessel_type_name']!r}"
                )
        return
    raise ValueError(f"Unknown graph_kind: {graph_kind!r}")


def build_typed_vessel_graph(
    graph: Any,
    branches: tuple[CenterlineBranch, ...],
    *,
    require_all_branches: bool = True,
) -> VesselGraph:
    """Convert a predicted FLOWCAT segment graph into dependency-light types."""

    validate_graph_schema(graph, graph_kind="segments_pred")
    mapped = map_graph_labels_to_branches(
        graph,
        branches,
        require_all_branches=require_all_branches,
    )
    references = {branch.coordinate_reference for branch in mapped}
    if len(references) != 1:
        raise FlowcatSchemaError("Centerline branches use mixed coordinate references")
    reference = next(iter(references))
    if reference is not CoordinateReference.FLOWCAT_LPI_CORNER_RELATIVE_MM:
        raise FlowcatSchemaError(
            "Build the typed graph before converting FLOWCAT-relative branches to image physical coordinates"
        )
    by_cell_id = {branch.cell_id: branch for branch in mapped}
    bifurcations: list[Bifurcation] = []
    for node, degree in graph.degree:
        if degree < 3:
            continue
        incident_cell_ids = tuple(
            sorted(int(data["cell_id"]) for _, _, data in graph.edges(node, data=True))
        )
        missing = set(incident_cell_ids).difference(by_cell_id)
        if missing:
            raise FlowcatSchemaError(
                f"Bifurcation node {node!r} references absent centerline cells {sorted(missing)!r}"
            )
        node_data = graph.nodes[node]
        if "pos" in node_data:
            position = _require_point(node_data["pos"], context=f"bifurcation node {node!r} pos")
        else:
            position = _shared_endpoint(tuple(by_cell_id[cell_id] for cell_id in incident_cell_ids))
        bifurcations.append(
            Bifurcation(
                bifurcation_id=f"node-{node}",
                coordinates_mm=tuple(float(value) for value in position),
                connected_branch_ids=tuple(f"cell-{cell_id}" for cell_id in incident_cell_ids),
                coordinate_reference=reference,
                source_file=mapped[0].source_file if mapped else None,
            )
        )
    return VesselGraph(
        branches=mapped,
        bifurcations=tuple(bifurcations),
        directed=bool(graph.is_directed()),
        source_file=mapped[0].source_file if mapped else None,
    )


def _infer_graph_kind(path: Path) -> Literal["segments", "segments_pred", "local"]:
    name = path.name.lower()
    if "segments_graph_pred" in name:
        return "segments_pred"
    if "segments_graph" in name:
        return "segments"
    if "local_graph" in name or name == "graph.pickle":
        return "local"
    raise FlowcatSchemaError(
        f"Cannot infer FLOWCAT graph schema from {path.name!r}; pass graph_kind explicitly"
    )


def _require_integer(
    data: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in data:
        raise FlowcatSchemaError(f"{context} is missing {key!r}")
    value = data[key]
    if not isinstance(value, Integral):
        raise FlowcatSchemaError(f"{context} {key!r} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise FlowcatSchemaError(f"{context} {key!r} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise FlowcatSchemaError(f"{context} {key!r} must be <= {maximum}, got {value}")
    return value


def _require_nonnegative_real(data: Mapping[str, Any], key: str, *, context: str) -> float:
    if key not in data:
        raise FlowcatSchemaError(f"{context} is missing {key!r}")
    value = data[key]
    if not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) < 0:
        raise FlowcatSchemaError(f"{context} {key!r} must be a finite non-negative number, got {value!r}")
    return float(value)


def _require_point(value: Any, *, context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FlowcatSchemaError(f"{context} is not a numeric coordinate") from exc
    if array.shape != (3,) or not np.isfinite(array).all():
        raise FlowcatSchemaError(f"{context} must be a finite shape-(3,) coordinate, got {array!r}")
    return array


def _require_points(value: Any, *, context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FlowcatSchemaError(f"{context} is not a numeric coordinate array") from exc
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise FlowcatSchemaError(f"{context} must have finite shape (n, 3), got {array.shape!r}")
    return array


def _require_radii(value: Any, *, context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FlowcatSchemaError(f"{context} is not a numeric radius array") from exc
    if array.ndim != 1 or not np.isfinite(array).all() or bool((array < 0).any()):
        raise FlowcatSchemaError(f"{context} must be a finite non-negative 1D array")
    return array


def _shared_endpoint(branches: tuple[CenterlineBranch, ...]) -> np.ndarray:
    endpoints = np.asarray(
        [
            point.coordinates_mm
            for branch in branches
            for point in (branch.points[0], branch.points[-1])
        ],
        dtype=float,
    )
    distances = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=2)
    counts = (distances <= 1e-3).sum(axis=1)
    best = int(np.argmax(counts))
    if counts[best] < len(branches):
        raise FlowcatSchemaError("Could not locate a shared centerline endpoint for a bifurcation")
    close = distances[best] <= 1e-3
    return endpoints[close].mean(axis=0)
