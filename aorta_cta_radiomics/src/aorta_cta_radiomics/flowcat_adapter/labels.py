"""FLOWCAT's 14-class extracranial artery convention and graph mappings."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral, Real
from typing import Any, Iterable, Mapping

from .exceptions import FlowcatSchemaError
from .schemas import ArteryIdentity, CenterlineBranch


# This is FLOWCAT's model class order, not the separate 0=background volumetric
# convention proposed for derived labelled masks.
FLOWCAT_ARTERY_CLASSES: dict[int, tuple[str, str, str | None]] = {
    0: ("other", "other artery", None),
    1: ("AA", "aorta / aortic arch", None),
    2: ("BT", "brachiocephalic trunk", None),
    3: ("RCCA", "common carotid artery", "right"),
    4: ("LCCA", "common carotid artery", "left"),
    5: ("RSA", "subclavian artery", "right"),
    6: ("LSA", "subclavian artery", "left"),
    7: ("RVA", "vertebral artery", "right"),
    8: ("LVA", "vertebral artery", "left"),
    9: ("RICA", "internal carotid artery", "right"),
    10: ("LICA", "internal carotid artery", "left"),
    11: ("RECA", "external carotid artery", "right"),
    12: ("LECA", "external carotid artery", "left"),
    13: ("BA", "basilar artery", None),
}

FLOWCAT_CODE_TO_CLASS_ID = {value[0]: key for key, value in FLOWCAT_ARTERY_CLASSES.items()}


def artery_identity_from_flowcat(
    class_id: int,
    *,
    code: str | None = None,
    confidence: float | None = None,
) -> ArteryIdentity:
    """Create a checked identity from a FLOWCAT model class."""

    if not isinstance(class_id, Integral) or int(class_id) not in FLOWCAT_ARTERY_CLASSES:
        raise FlowcatSchemaError(f"Unsupported FLOWCAT vessel_type {class_id!r}; expected integer 0..13")
    class_id = int(class_id)
    expected_code, name, laterality = FLOWCAT_ARTERY_CLASSES[class_id]
    if code is not None and str(code) != expected_code:
        raise FlowcatSchemaError(
            f"FLOWCAT vessel label mismatch: class {class_id} is {expected_code!r}, got {code!r}"
        )
    if confidence is not None:
        if not isinstance(confidence, Real) or not 0.0 <= float(confidence) <= 1.0:
            raise FlowcatSchemaError(f"Invalid vessel-label confidence: {confidence!r}")
        confidence = float(confidence)
    return ArteryIdentity(class_id, expected_code, name, laterality, confidence)


def graph_cell_id_identities(graph: Any) -> dict[int, ArteryIdentity]:
    """Extract FLOWCAT edge labels keyed by centerline-array ``cell_id``."""

    labels: dict[int, ArteryIdentity] = {}
    try:
        edges = graph.edges(data=True)
    except (AttributeError, TypeError) as exc:
        raise FlowcatSchemaError("Expected a NetworkX-like graph with edge attributes") from exc

    for source, target, data in edges:
        if not isinstance(data, Mapping):
            raise FlowcatSchemaError(f"Edge {(source, target)!r} attributes are not a mapping")
        missing = {key for key in ("cell_id", "vessel_type", "vessel_type_name") if key not in data}
        if missing:
            raise FlowcatSchemaError(f"Edge {(source, target)!r} is missing {sorted(missing)!r}")
        cell_id = data["cell_id"]
        if not isinstance(cell_id, Integral) or int(cell_id) < 0:
            raise FlowcatSchemaError(f"Edge {(source, target)!r} has invalid cell_id {cell_id!r}")
        cell_id = int(cell_id)
        confidence_keys = ("vessel_type_confidence", "label_confidence", "confidence")
        confidence = next((data[key] for key in confidence_keys if key in data), None)
        identity = artery_identity_from_flowcat(
            data["vessel_type"], code=str(data["vessel_type_name"]), confidence=confidence
        )
        if cell_id in labels and labels[cell_id] != identity:
            raise FlowcatSchemaError(f"Conflicting vessel labels for centerline cell_id {cell_id}")
        labels[cell_id] = identity
    return labels


def map_graph_labels_to_branches(
    graph: Any,
    branches: Iterable[CenterlineBranch],
    *,
    require_all_branches: bool = True,
) -> tuple[CenterlineBranch, ...]:
    """Attach graph edge labels to centerline branches through stable ``cell_id`` values."""

    labels = graph_cell_id_identities(graph)
    mapped: list[CenterlineBranch] = []
    branch_cell_ids: set[int] = set()
    for branch in branches:
        branch_cell_ids.add(branch.cell_id)
        identity = labels.get(branch.cell_id)
        if identity is None:
            if require_all_branches:
                raise FlowcatSchemaError(f"No graph label for centerline branch cell_id {branch.cell_id}")
            mapped.append(branch)
            continue
        points = tuple(
            replace(
                point,
                artery=identity,
                vessel_label_confidence=identity.confidence,
            )
            for point in branch.points
        )
        mapped.append(replace(branch, points=points, artery=identity))

    orphan_labels = set(labels).difference(branch_cell_ids)
    if orphan_labels:
        raise FlowcatSchemaError(
            "Graph labels reference absent centerline cell_id values: "
            + ", ".join(str(value) for value in sorted(orphan_labels))
        )
    return tuple(mapped)
