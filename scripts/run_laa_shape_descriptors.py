#!/usr/bin/env python3
"""
Compute LAA surface + shape descriptors + optional PCA using the STACOM2025 scripts.

This is a lightweight wrapper that operates on a single segmentation NIfTI.
It does NOT require the full ImageCAS dataset.

Outputs:
  - surface VTK file
  - JSON with shape descriptors
  - combined CSV (single-row)
  - optional PCA plot (only meaningful with multiple JSONs)

Example (single scan):
  python scripts/run_laa_shape_descriptors.py \
    --input /path/to/sub-547_defaced_laa8.nii.gz \
    --output-dir /path/to/outputs/laa_shape_547 \
    --label-id 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from sklearn.decomposition import PCA


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute LAA surface + shape descriptors + PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input LAA segmentation (.nii/.nii.gz)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--label-id", type=int, default=1, help="Label value to extract (use 1 for binary masks)")
    p.add_argument("--run-pca", action="store_true", help="Run PCA across all JSONs in output-dir")
    p.add_argument("--check-env", action="store_true", help="Check environment and exit")
    return p.parse_args()


def _check_env() -> None:
    import sys

    details = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "SimpleITK": sitk.__version__,
        "vtk": vtk.vtkVersion.GetVTKVersion(),
    }
    try:
        import sklearn

        details["scikit-learn"] = sklearn.__version__
    except Exception as exc:  # noqa: BLE001
        details["scikit-learn_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(k.endswith("_error") for k in details):
        raise SystemExit(2)


def _scan_id(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return input_path.stem


def _sitk_to_vtk(img: sitk.Image) -> vtk.vtkImageData:
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    arr = sitk.GetArrayFromImage(img)
    vtk_image = vtk.vtkImageData()

    if len(size) == 2:
        size.append(1)
    if len(origin) == 2:
        origin.append(0.0)
    if len(spacing) == 2:
        spacing.append(spacing[0])
    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    vtk_image.SetDirectionMatrix(direction)

    depth_array = numpy_to_vtk(arr.ravel(), deep=True)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)
    vtk_image.Modified()
    return vtk_image


def _extract_surface(seg_path: Path, label_id: int, surf_out: Path) -> vtk.vtkPolyData | None:
    img = sitk.ReadImage(str(seg_path))
    vtk_img = _sitk_to_vtk(img)

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, label_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print(f"No isosurface found for label {label_id}")
        return None

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputConnection(mc.GetOutputPort())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    surface = conn.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(surf_out))
    writer.SetInputData(surface)
    writer.Write()
    return surface


def _compute_descriptors(surface: vtk.vtkPolyData) -> dict:
    result: dict[str, float] = {}
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()

    volume = mass.GetVolume()
    surface_area = mass.GetSurfaceArea()
    nsi = mass.GetNormalizedShapeIndex()
    result["volume"] = volume
    result["surface_area"] = surface_area
    result["normalized_shape_index"] = nsi
    result["surface_to_volume_ratio"] = surface_area / volume if volume != 0 else float("inf")

    n_samples = surface.GetNumberOfPoints()
    data_matrix = np.zeros((n_samples, 3))
    for i in range(n_samples):
        p = surface.GetPoint(i)
        data_matrix[i, 0] = p[0]
        data_matrix[i, 1] = p[1]
        data_matrix[i, 2] = p[2]

    shape_pca = PCA()
    shape_pca.fit(data_matrix)
    eigenvalues = shape_pca.explained_variance_
    major_axis_length = 4 * np.sqrt(eigenvalues[0])
    minor_axis_length = 4 * np.sqrt(eigenvalues[1])
    least_axis_length = 4 * np.sqrt(eigenvalues[2])
    result["major_axis_length"] = major_axis_length
    result["minor_axis_length"] = minor_axis_length
    result["least_axis_length"] = least_axis_length
    result["elongation"] = minor_axis_length / major_axis_length if major_axis_length != 0 else 0.0
    result["flatness"] = least_axis_length / major_axis_length if major_axis_length != 0 else 0.0
    return result


def _write_combined_csv(descriptor_dir: Path, out_csv: Path) -> None:
    json_files = sorted(descriptor_dir.glob("*.json"))
    if not json_files:
        return
    with open(json_files[0], "r") as f:
        first = json.load(f)
    fieldnames = ["filename"] + list(first.keys())
    with open(out_csv, "w", newline="") as f:
        f.write(",".join(fieldnames) + "\n")
        for jf in json_files:
            data = json.loads(jf.read_text())
            row = [jf.stem.replace("_shape_descriptors", "")]
            row.extend(str(data.get(k, "")) for k in fieldnames[1:])
            f.write(",".join(row) + "\n")


def _run_pca_from_csv(csv_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    data = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if len(data.shape) == 0:
        print("PCA skipped: only one sample in CSV.")
        return
    data_matrix = np.array([list(row)[1:] for row in data], dtype=float)
    data_matrix = data_matrix - np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    std[std == 0] = 1.0
    data_matrix = data_matrix / std

    pca = PCA()
    components = pca.fit_transform(data_matrix)
    plt.figure()
    plt.plot(pca.explained_variance_ratio_ * 100)
    plt.xlabel("Principal component")
    plt.ylabel("Percent explained variance")
    plt.tight_layout()
    plt.savefig(str(out_png))

    if components.shape[0] >= 2:
        plt.figure()
        plt.plot(components[:, 0], components[:, 1], ".")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(str(out_png.with_name(out_png.stem + "_pc1_pc2.png")))


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_id = _scan_id(input_path)
    surface_dir = output_dir / "surfaces"
    descriptor_dir = output_dir / "descriptors"
    surface_dir.mkdir(parents=True, exist_ok=True)
    descriptor_dir.mkdir(parents=True, exist_ok=True)

    surf_out = surface_dir / f"{scan_id}_laa_surface.vtk"
    surface = _extract_surface(input_path, args.label_id, surf_out)
    if surface is None:
        return 2

    desc = _compute_descriptors(surface)
    desc_out = descriptor_dir / f"{scan_id}_shape_descriptors.json"
    desc_out.write_text(json.dumps(desc, indent=2))
    combined_csv = descriptor_dir / "combined_shape_descriptors.csv"
    _write_combined_csv(descriptor_dir, combined_csv)

    if args.run_pca:
        _run_pca_from_csv(combined_csv, descriptor_dir / "pca_explained_variance.png")

    print(f"Saved surface: {surf_out}")
    print(f"Saved descriptors: {desc_out}")
    print(f"Saved combined CSV: {combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
