"""Optional VTK/VTP/STL readers for FLOWCAT geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import FlowcatSchemaError, MissingOptionalDependencyError


def load_polydata(path: str | Path) -> Any:
    """Read FLOWCAT polydata with VTK when that optional dependency exists."""

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"FLOWCAT geometry not found: {file_path}")
    try:
        import vtk
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "VTK is required to load FLOWCAT .vtk/.vtp/.stl geometry. "
            "Install the optional mesh dependency (`pip install vtk`) or load it in 3D Slicer."
        ) from exc

    suffix = file_path.suffix.lower()
    if suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    else:
        raise FlowcatSchemaError(
            f"Unsupported FLOWCAT geometry format {suffix!r}; expected .vtk, .vtp, or .stl"
        )
    reader.SetFileName(str(file_path))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise FlowcatSchemaError(f"FLOWCAT geometry contains no readable points: {file_path}")
    return output


load_vtk_polydata = load_polydata
