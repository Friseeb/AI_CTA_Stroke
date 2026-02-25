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
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from sklearn.decomposition import PCA


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute LAA shape descriptors or run batch segmentation->mesh conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=None, help="Input LAA segmentation (.nii/.nii.gz) for single-case mode")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--label-id", type=int, default=1, help="Label value to extract (use 1 for binary masks)")
    p.add_argument("--run-pca", action="store_true", help="Run PCA across all JSONs in output-dir")
    p.add_argument(
        "--batch-mask-root",
        default=None,
        help="If set, run batch mesh conversion from <case>/<case>_<suffix>.nii.gz masks under this root",
    )
    p.add_argument(
        "--mask-suffix",
        action="append",
        default=[],
        help="Mask suffix(es) for batch mode, e.g. laa_nudf (repeatable)",
    )
    p.add_argument("--mesh-format", choices=["vtk", "stl", "ply"], default="vtk", help="Batch mesh format")
    p.add_argument("--subject", action="append", default=[], help="Optional subject IDs for batch mode")
    p.add_argument(
        "--isotropic-mm",
        type=float,
        default=None,
        help="Optional isotropic spacing (mm) before surface extraction; improves cross-scan comparability.",
    )
    p.add_argument(
        "--no-largest-component",
        action="store_true",
        help="Do not restrict masks/surfaces to the largest connected component.",
    )
    p.add_argument(
        "--smooth-iters",
        type=int,
        default=0,
        help="Laplacian smoothing iterations for mesh post-processing (0 disables smoothing).",
    )
    p.add_argument(
        "--smooth-relaxation",
        type=float,
        default=0.1,
        help="Laplacian smoothing relaxation factor.",
    )
    p.add_argument(
        "--decimate-reduction",
        type=float,
        default=0.0,
        help="Target fraction of polygons to remove in decimation (0-0.95).",
    )
    p.add_argument(
        "--provenance-json",
        default=None,
        help="Optional provenance JSON path (default: <output-dir>/mesh_conversion_provenance.json in batch mode).",
    )
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bar in batch mode if available")
    p.add_argument("--mesh-only", action="store_true", help="Single-case mode: export surface mesh only")
    p.add_argument(
        "--batch-summary-csv",
        default=None,
        help="Batch mode summary CSV path (default: <output-dir>/mesh_conversion_summary.csv)",
    )
    p.add_argument("--check-env", action="store_true", help="Check environment and exit")
    return p.parse_args()


def _check_env() -> None:
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
    try:
        import tqdm

        details["tqdm"] = tqdm.__version__
    except Exception as exc:  # noqa: BLE001
        details["tqdm_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(k.endswith("_error") for k in details):
        raise SystemExit(2)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _runtime_details() -> dict:
    details = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "SimpleITK": sitk.__version__,
        "vtk": vtk.vtkVersion.GetVTKVersion(),
    }
    try:
        import sklearn

        details["scikit-learn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import tqdm

        details["tqdm"] = tqdm.__version__
    except Exception:  # noqa: BLE001
        pass
    return details


def _resample_mask_isotropic(mask: sitk.Image, spacing_mm: float) -> sitk.Image:
    old_spacing = np.array(mask.GetSpacing(), dtype=np.float64)
    old_size = np.array(mask.GetSize(), dtype=np.int64)
    new_spacing = np.array([spacing_mm, spacing_mm, spacing_mm], dtype=np.float64)
    new_size = np.maximum(1, np.round(old_size * (old_spacing / new_spacing)).astype(np.int64))

    rs = sitk.ResampleImageFilter()
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetOutputSpacing(tuple(float(x) for x in new_spacing))
    rs.SetSize([int(x) for x in new_size])
    rs.SetOutputOrigin(mask.GetOrigin())
    rs.SetOutputDirection(mask.GetDirection())
    rs.SetDefaultPixelValue(0)
    out = rs.Execute(mask)
    return sitk.Cast(out > 0, sitk.sitkUInt8)


def _largest_component_sitk(mask: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    if not labels:
        out = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(mask)
        return out
    largest = max(labels, key=lambda lab: stats.GetNumberOfPixels(lab))
    out = sitk.Cast(cc == largest, sitk.sitkUInt8)
    out.CopyInformation(mask)
    return out


def _prepare_mask(
    seg_path: Path,
    label_id: int,
    isotropic_mm: float | None,
    largest_component: bool,
) -> sitk.Image:
    img = sitk.ReadImage(str(seg_path))
    if label_id > 0:
        mask = sitk.Cast(img == int(label_id), sitk.sitkUInt8)
    else:
        mask = sitk.Cast(img > 0, sitk.sitkUInt8)

    if isotropic_mm is not None and isotropic_mm > 0:
        mask = _resample_mask_isotropic(mask, float(isotropic_mm))

    if largest_component:
        mask = _largest_component_sitk(mask)
    return mask


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


def _write_surface(surface: vtk.vtkPolyData, surf_out: Path, mesh_format: str) -> None:
    surf_out.parent.mkdir(parents=True, exist_ok=True)
    if mesh_format == "vtk":
        writer = vtk.vtkPolyDataWriter()
    elif mesh_format == "stl":
        writer = vtk.vtkSTLWriter()
    elif mesh_format == "ply":
        writer = vtk.vtkPLYWriter()
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_format}")
    writer.SetFileName(str(surf_out))
    writer.SetInputData(surface)
    writer.Write()


def _postprocess_surface(
    surface: vtk.vtkPolyData,
    smooth_iters: int,
    smooth_relaxation: float,
    decimate_reduction: float,
) -> vtk.vtkPolyData:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(surface)
    tri.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()

    current_port = clean.GetOutputPort()
    if smooth_iters > 0:
        sm = vtk.vtkSmoothPolyDataFilter()
        sm.SetInputConnection(current_port)
        sm.SetNumberOfIterations(int(smooth_iters))
        sm.SetRelaxationFactor(float(smooth_relaxation))
        sm.FeatureEdgeSmoothingOff()
        sm.BoundarySmoothingOff()
        sm.Update()
        current_port = sm.GetOutputPort()

    reduction = float(decimate_reduction)
    if reduction > 0:
        reduction = min(0.95, max(0.0, reduction))
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputConnection(current_port)
        dec.SetTargetReduction(reduction)
        dec.Update()
        current_port = dec.GetOutputPort()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(current_port)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out


def _mesh_qc(surface: vtk.vtkPolyData) -> dict[str, float | int | bool]:
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()

    feature_boundary = vtk.vtkFeatureEdges()
    feature_boundary.SetInputData(surface)
    feature_boundary.BoundaryEdgesOn()
    feature_boundary.FeatureEdgesOff()
    feature_boundary.ManifoldEdgesOff()
    feature_boundary.NonManifoldEdgesOff()
    feature_boundary.Update()
    boundary_edges = int(feature_boundary.GetOutput().GetNumberOfCells())

    feature_nonmanifold = vtk.vtkFeatureEdges()
    feature_nonmanifold.SetInputData(surface)
    feature_nonmanifold.BoundaryEdgesOff()
    feature_nonmanifold.FeatureEdgesOff()
    feature_nonmanifold.ManifoldEdgesOff()
    feature_nonmanifold.NonManifoldEdgesOn()
    feature_nonmanifold.Update()
    non_manifold_edges = int(feature_nonmanifold.GetOutput().GetNumberOfCells())

    edge_extractor = vtk.vtkExtractEdges()
    edge_extractor.SetInputData(surface)
    edge_extractor.Update()
    n_edges = int(edge_extractor.GetOutput().GetNumberOfLines())

    n_points = int(surface.GetNumberOfPoints())
    n_polys = int(surface.GetNumberOfPolys())
    euler_proxy = n_points - n_edges + n_polys

    return {
        "points": n_points,
        "polys": n_polys,
        "surface_area_mm2": float(mass.GetSurfaceArea()),
        "volume_mm3": float(mass.GetVolume()),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "is_watertight": bool(boundary_edges == 0 and non_manifold_edges == 0),
        "euler_characteristic_proxy": int(euler_proxy),
    }


def _extract_surface(
    seg_path: Path,
    label_id: int,
    isotropic_mm: float | None,
    largest_component: bool,
    smooth_iters: int,
    smooth_relaxation: float,
    decimate_reduction: float,
) -> tuple[vtk.vtkPolyData | None, dict[str, float | int | bool]]:
    mask = _prepare_mask(
        seg_path=seg_path,
        label_id=label_id,
        isotropic_mm=isotropic_mm,
        largest_component=largest_component,
    )
    vtk_img = _sitk_to_vtk(mask)

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, 1)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print(f"No isosurface found for {seg_path.name}")
        return None, {}

    surface = vtk.vtkPolyData()
    surface.DeepCopy(mc.GetOutput())

    if largest_component:
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputData(surface)
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        largest = vtk.vtkPolyData()
        largest.DeepCopy(conn.GetOutput())
        surface = largest

    surface = _postprocess_surface(
        surface=surface,
        smooth_iters=smooth_iters,
        smooth_relaxation=smooth_relaxation,
        decimate_reduction=decimate_reduction,
    )
    qc = _mesh_qc(surface)
    return surface, qc


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


def _subject_id_from_case(case_id: str) -> str:
    token = case_id.split("_")[0].replace("sub-", "")
    if token.isdigit():
        return str(int(token))
    return token


def _run_batch_mesh_mode(args: argparse.Namespace) -> int:
    mask_root = Path(args.batch_mask_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.batch_summary_csv) if args.batch_summary_csv else output_dir / "mesh_conversion_summary.csv"

    if not mask_root.exists():
        raise FileNotFoundError(f"Batch mask root not found: {mask_root}")

    suffixes = args.mask_suffix[:] if args.mask_suffix else ["laa_nudf", "left_atrium_highres", "aorta_highres"]
    wanted_subjects = {str(int(s)) for s in args.subject if str(s).isdigit()}
    largest_component = not args.no_largest_component
    decimate_reduction = min(0.95, max(0.0, float(args.decimate_reduction)))
    provenance_json = Path(args.provenance_json) if args.provenance_json else output_dir / "mesh_conversion_provenance.json"

    work_items: list[tuple[str, str, str, Path, Path]] = []
    for case_dir in sorted(mask_root.glob("sub-*_acq-CTA_ct")):
        case_id = case_dir.name
        sid = _subject_id_from_case(case_id)
        if wanted_subjects and sid.isdigit() and sid not in wanted_subjects:
            continue

        for suffix in suffixes:
            mask_path = case_dir / f"{case_id}_{suffix}.nii.gz"
            mesh_path = output_dir / case_id / f"{case_id}_{suffix}.{args.mesh_format}"
            work_items.append((case_id, sid, suffix, mask_path, mesh_path))

    work_iter = work_items
    if args.progress:
        try:
            from tqdm.auto import tqdm

            work_iter = tqdm(work_items, total=len(work_items), unit="roi", desc="Segmentation->Mesh")
        except Exception:
            print("tqdm not available; running without progress bar.")

    rows: list[dict] = []
    for case_id, sid, suffix, mask_path, mesh_path in work_iter:
            row = {
                "case_id": case_id,
                "subject_id": sid,
                "region": suffix,
                "mask_path": str(mask_path),
                "mesh_path": str(mesh_path),
                "status": "pending",
                "points": "",
                "polys": "",
                "surface_area_mm2": "",
                "volume_mm3": "",
                "boundary_edges": "",
                "non_manifold_edges": "",
                "is_watertight": "",
                "euler_characteristic_proxy": "",
                "input_sha256": "",
                "mesh_sha256": "",
                "isotropic_mm": args.isotropic_mm if args.isotropic_mm is not None else "",
                "largest_component": largest_component,
                "smooth_iters": int(args.smooth_iters),
                "smooth_relaxation": float(args.smooth_relaxation),
                "decimate_reduction": decimate_reduction,
                "error": "",
            }
            if not mask_path.exists():
                row["status"] = "skip_missing_mask"
                rows.append(row)
                continue
            try:
                row["input_sha256"] = _sha256_file(mask_path)
                surface, qc = _extract_surface(
                    seg_path=mask_path,
                    label_id=args.label_id,
                    isotropic_mm=args.isotropic_mm,
                    largest_component=largest_component,
                    smooth_iters=args.smooth_iters,
                    smooth_relaxation=args.smooth_relaxation,
                    decimate_reduction=decimate_reduction,
                )
                if surface is None:
                    row["status"] = "skip_empty_surface"
                    rows.append(row)
                    continue
                _write_surface(surface, mesh_path, args.mesh_format)
                row["status"] = "success"
                row.update(qc)
                row["mesh_sha256"] = _sha256_file(mesh_path)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failure"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "subject_id",
                "region",
                "mask_path",
                "mesh_path",
                "status",
                "points",
                "polys",
                "surface_area_mm2",
                "volume_mm3",
                "boundary_edges",
                "non_manifold_edges",
                "is_watertight",
                "euler_characteristic_proxy",
                "input_sha256",
                "mesh_sha256",
                "isotropic_mm",
                "largest_component",
                "smooth_iters",
                "smooth_relaxation",
                "decimate_reduction",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "runtime": _runtime_details(),
        "parameters": {
            "batch_mask_root": str(mask_root),
            "output_dir": str(output_dir),
            "mask_suffix": suffixes,
            "label_id": int(args.label_id),
            "mesh_format": args.mesh_format,
            "isotropic_mm": args.isotropic_mm,
            "largest_component": largest_component,
            "smooth_iters": int(args.smooth_iters),
            "smooth_relaxation": float(args.smooth_relaxation),
            "decimate_reduction": decimate_reduction,
            "subject_filter": sorted(wanted_subjects),
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "row_count": len(rows),
            "success_count": sum(1 for r in rows if r["status"] == "success"),
        },
    }
    provenance_json.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    ok = sum(1 for r in rows if r["status"] == "success")
    print(f"Saved mesh summary: {summary_csv}")
    print(f"Saved provenance: {provenance_json}")
    print(f"Rows: {len(rows)} | success: {ok} | non-success: {len(rows) - ok}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    if args.batch_mask_root:
        return _run_batch_mesh_mode(args)

    if not args.input:
        raise ValueError("Single-case mode requires --input (or set --batch-mask-root for batch mode).")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_id = _scan_id(input_path)
    surface_dir = output_dir / "surfaces"
    descriptor_dir = output_dir / "descriptors"
    surface_dir.mkdir(parents=True, exist_ok=True)
    descriptor_dir.mkdir(parents=True, exist_ok=True)

    surf_out = surface_dir / f"{scan_id}_laa_surface.{args.mesh_format}"
    largest_component = not args.no_largest_component
    decimate_reduction = min(0.95, max(0.0, float(args.decimate_reduction)))
    surface, qc = _extract_surface(
        seg_path=input_path,
        label_id=args.label_id,
        isotropic_mm=args.isotropic_mm,
        largest_component=largest_component,
        smooth_iters=args.smooth_iters,
        smooth_relaxation=args.smooth_relaxation,
        decimate_reduction=decimate_reduction,
    )
    if surface is None:
        return 2
    _write_surface(surface, surf_out, args.mesh_format)

    single_provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "runtime": _runtime_details(),
        "parameters": {
            "input": str(input_path),
            "label_id": int(args.label_id),
            "mesh_format": args.mesh_format,
            "isotropic_mm": args.isotropic_mm,
            "largest_component": largest_component,
            "smooth_iters": int(args.smooth_iters),
            "smooth_relaxation": float(args.smooth_relaxation),
            "decimate_reduction": decimate_reduction,
        },
        "checksums": {
            "input_sha256": _sha256_file(input_path),
            "mesh_sha256": _sha256_file(surf_out),
        },
        "mesh_qc": qc,
    }
    single_prov_path = Path(args.provenance_json) if args.provenance_json else output_dir / "single_mesh_provenance.json"
    single_prov_path.write_text(json.dumps(single_provenance, indent=2), encoding="utf-8")

    if args.mesh_only:
        print(f"Saved surface: {surf_out}")
        print(f"Saved provenance: {single_prov_path}")
        return 0

    desc = _compute_descriptors(surface)
    desc_out = descriptor_dir / f"{scan_id}_shape_descriptors.json"
    desc_out.write_text(json.dumps(desc, indent=2))
    combined_csv = descriptor_dir / "combined_shape_descriptors.csv"
    _write_combined_csv(descriptor_dir, combined_csv)

    if args.run_pca:
        _run_pca_from_csv(combined_csv, descriptor_dir / "pca_explained_variance.png")

    print(f"Saved surface: {surf_out}")
    print(f"Saved provenance: {single_prov_path}")
    print(f"Saved descriptors: {desc_out}")
    print(f"Saved combined CSV: {combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
