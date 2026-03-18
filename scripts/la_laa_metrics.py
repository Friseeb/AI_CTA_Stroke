#!/usr/bin/env python3
"""Minimum viable LA/LAA relational shape metrics from mesh pairs.

Example:
  python scripts/la_laa_metrics.py \
    --la /path/to/la.ply \
    --laa /path/to/laa.ply \
    --out /path/to/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh
from vtk.util.numpy_support import vtk_to_numpy
import vtk


DEFAULT_PARAMS: dict[str, float] = {
    "near_contact_mm": 2.0,
    "closest_quantile": 0.02,
    "max_gap_fail_mm": 12.0,
    "min_ostium_points": 200,
    "proximal_length_mm": 10.0,
    "ostium_abs_cap_mm": 5.0,
    "closest_chunk_size": 5000.0,
}

EPS = 1e-12


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh file into a single trimesh.Trimesh."""
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    if mesh_path.suffix.lower() in {".vtk", ".vtp"}:
        mesh = _load_vtk_polydata_as_trimesh(mesh_path)
        mesh.process(validate=True)
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise ValueError(f"Loaded mesh is empty: {mesh_path}")
        return mesh

    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    else:
        raise TypeError(f"Unsupported trimesh object type: {type(loaded).__name__}")

    # Keep this deterministic and robust while avoiding topology-altering ops.
    mesh.process(validate=True)
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")
    return mesh


def _load_vtk_polydata(path: Path) -> vtk.vtkPolyData:
    suffix = path.suffix.lower()
    if suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported VTK extension: {suffix}")

    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"Failed to read VTK polydata: {path}")
    return poly


def _boundary_edges_vtk(poly: vtk.vtkPolyData) -> int:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    return int(fe.GetOutput().GetNumberOfCells())


def _cap_inferior_boundary_loops(poly: vtk.vtkPolyData, inferior_band_mm: float = 15.0) -> tuple[vtk.vtkPolyData, int]:
    """Cap only boundary loops close to inferior mesh extent (low-Z band)."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()

    boundary = fe.GetOutput()
    if boundary is None or boundary.GetNumberOfCells() == 0:
        return poly, 0

    stripper = vtk.vtkStripper()
    stripper.SetInputData(boundary)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    loops_pd = stripper.GetOutput()

    loops_pts = loops_pd.GetPoints()
    loops_lines = loops_pd.GetLines()
    if loops_pts is None or loops_lines is None or loops_lines.GetNumberOfCells() == 0:
        return poly, 0

    bounds = poly.GetBounds()
    z_min = float(bounds[4])
    z_thr = z_min + float(inferior_band_mm)

    conn = vtk_to_numpy(loops_lines.GetData())
    cap_points = vtk.vtkPoints()
    cap_polys = vtk.vtkCellArray()

    capped_loops = 0
    i = 0
    n = len(conn)
    while i < n:
        m = int(conn[i])
        if m < 3:
            i += 1 + max(m, 0)
            continue
        ids = conn[i + 1 : i + 1 + m].astype(np.int64)
        i += 1 + m

        pts = np.asarray([loops_pts.GetPoint(int(pid)) for pid in ids], dtype=np.float64)
        if pts.shape[0] >= 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
            pts = pts[:-1]
        if pts.shape[0] < 3:
            continue

        if float(np.mean(pts[:, 2])) > z_thr:
            continue

        start = cap_points.GetNumberOfPoints()
        for p in pts:
            cap_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        poly_cell = vtk.vtkPolygon()
        poly_cell.GetPointIds().SetNumberOfIds(int(pts.shape[0]))
        for j in range(int(pts.shape[0])):
            poly_cell.GetPointIds().SetId(j, start + j)
        cap_polys.InsertNextCell(poly_cell)
        capped_loops += 1

    if capped_loops == 0:
        return poly, 0

    caps_pd = vtk.vtkPolyData()
    caps_pd.SetPoints(cap_points)
    caps_pd.SetPolys(cap_polys)

    tri_caps = vtk.vtkTriangleFilter()
    tri_caps.SetInputData(caps_pd)
    tri_caps.Update()

    append = vtk.vtkAppendPolyData()
    append.AddInputData(poly)
    append.AddInputConnection(tri_caps.GetOutputPort())
    append.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(append.GetOutputPort())
    clean.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(tri.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    return normals.GetOutput(), capped_loops


def _vtk_faces_to_triangles(poly: vtk.vtkPolyData) -> np.ndarray:
    polys = poly.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        return np.empty((0, 3), dtype=np.int64)

    conn = vtk_to_numpy(polys.GetData())
    faces: list[list[int]] = []
    i = 0
    n = len(conn)
    while i < n:
        m = int(conn[i])
        if m < 3:
            i += 1 + max(m, 0)
            continue
        idx = conn[i + 1 : i + 1 + m].astype(np.int64)
        # Fan triangulation for polygonal faces.
        for j in range(1, m - 1):
            faces.append([int(idx[0]), int(idx[j]), int(idx[j + 1])])
        i += 1 + m

    if not faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _load_vtk_polydata_as_trimesh(path: Path) -> trimesh.Trimesh:
    poly = _load_vtk_polydata(path)
    points_vtk = poly.GetPoints()
    if points_vtk is None or points_vtk.GetNumberOfPoints() == 0:
        raise ValueError(f"VTK mesh has no points: {path}")

    vertices = vtk_to_numpy(points_vtk.GetData()).astype(np.float64)
    faces = _vtk_faces_to_triangles(poly)
    if faces.shape[0] == 0:
        raise ValueError(f"VTK mesh has no polygon faces: {path}")

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_mesh_vtk_hole_capped(
    path: str,
    hole_size: float = 50.0,
    repair_mode: str = "fill_holes",
    inferior_band_mm: float = 15.0,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Load .vtk/.vtp and cap boundary holes with VTK FillHolesFilter.

    Returns:
      mesh: repaired trimesh object
      qc: dict with before/after boundary edges + watertight flags
    """
    mesh_path = Path(path)
    if mesh_path.suffix.lower() not in {".vtk", ".vtp"}:
        raise ValueError(f"Hole-capped VTK loader requires .vtk/.vtp, got: {mesh_path.suffix}")

    poly = _load_vtk_polydata(mesh_path)

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    poly_clean = clean.GetOutput()

    bnd_before = _boundary_edges_vtk(poly_clean)

    capped_loops = 0
    if repair_mode == "fill_holes":
        fh = vtk.vtkFillHolesFilter()
        fh.SetInputData(poly_clean)
        fh.SetHoleSize(float(hole_size))
        fh.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(fh.GetOutputPort())
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.AutoOrientNormalsOn()
        normals.Update()
        poly_after = normals.GetOutput()
    elif repair_mode == "inferior_cap":
        poly_after, capped_loops = _cap_inferior_boundary_loops(poly_clean, inferior_band_mm=float(inferior_band_mm))
    else:
        raise ValueError(f"Unknown repair_mode: {repair_mode}")

    bnd_after = _boundary_edges_vtk(poly_after)

    mesh = _polydata_to_trimesh(poly_after)
    mesh.process(validate=True)

    qc = {
        "la_boundary_edges_before": bnd_before,
        "la_boundary_edges_after": bnd_after,
        "la_watertight_before": bool(bnd_before == 0),
        "la_watertight_after": bool(mesh.is_watertight),
        "la_hole_fill_size": float(hole_size),
        "la_repair_mode": repair_mode,
        "la_inferior_band_mm": float(inferior_band_mm),
        "la_capped_loops_n": int(capped_loops),
    }
    return mesh, qc


def _polydata_to_trimesh(poly: vtk.vtkPolyData) -> trimesh.Trimesh:
    points_vtk = poly.GetPoints()
    if points_vtk is None or points_vtk.GetNumberOfPoints() == 0:
        raise ValueError("VTK mesh has no points.")
    vertices = vtk_to_numpy(points_vtk.GetData()).astype(np.float64)
    faces = _vtk_faces_to_triangles(poly)
    if faces.shape[0] == 0:
        raise ValueError("VTK mesh has no polygon faces.")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def normalize(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return np.full(3, np.nan, dtype=float)
    return v / n


def angle_deg(u: np.ndarray, v: np.ndarray, eps: float = EPS) -> float:
    uu = normalize(np.asarray(u, dtype=float), eps=eps)
    vv = normalize(np.asarray(v, dtype=float), eps=eps)
    if np.isnan(uu).any() or np.isnan(vv).any():
        return float("nan")
    c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def pca_eig(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues/eigenvectors of covariance matrix (ascending eigenvalues)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("PCA requires points with shape (N, 3), N >= 3.")
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def safe_mesh_volume(mesh: trimesh.Trimesh) -> float:
    if not bool(mesh.is_watertight):
        return float("nan")
    vol = float(mesh.volume)
    return vol if np.isfinite(vol) else float("nan")


def _empty_result() -> dict[str, Any]:
    return {
        "qc_far_apart": False,
        "qc_ostium_too_few_points": False,
        "qc_tip_wrong_side": False,
        "qc_exception": False,
        "error": "",
        "min_distance_mm": float("nan"),
        "ostium_points_n": float("nan"),
        "ostium_dist_median_mm": float("nan"),
        "ostium_center_x": float("nan"),
        "ostium_center_y": float("nan"),
        "ostium_center_z": float("nan"),
        "ostium_normal_x": float("nan"),
        "ostium_normal_y": float("nan"),
        "ostium_normal_z": float("nan"),
        "ostium_planarity": float("nan"),
        "laa_axis_x": float("nan"),
        "laa_axis_y": float("nan"),
        "laa_axis_z": float("nan"),
        "laa_axis_length_mm": float("nan"),
        "laa_prox_dir_x": float("nan"),
        "laa_prox_dir_y": float("nan"),
        "laa_prox_dir_z": float("nan"),
        "bend_ostiumNormal_vs_proxLAA_deg": float("nan"),
        "bend_LAaxis_vs_LAAaxis_deg": float("nan"),
        "laa_surface_area_mm2": float("nan"),
        "laa_volume_mm3": float("nan"),
        "la_surface_area_mm2": float("nan"),
        "la_volume_mm3": float("nan"),
    }


def _assign_xyz(out: dict[str, Any], prefix: str, v: np.ndarray) -> None:
    out[f"{prefix}_x"] = float(v[0]) if np.isfinite(v[0]) else float("nan")
    out[f"{prefix}_y"] = float(v[1]) if np.isfinite(v[1]) else float("nan")
    out[f"{prefix}_z"] = float(v[2]) if np.isfinite(v[2]) else float("nan")


def _merge_params(params: dict | None) -> dict[str, float]:
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    return cfg


def _closest_interface_points(
    vertices_laa: np.ndarray,
    distances: np.ndarray,
    cfg: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    near_contact_mm = float(cfg["near_contact_mm"])
    closest_quantile = float(cfg["closest_quantile"])
    min_ostium_points = int(cfg["min_ostium_points"])
    ostium_abs_cap_mm = float(cfg["ostium_abs_cap_mm"])

    if float(np.min(distances)) <= near_contact_mm:
        idx = np.where(distances <= near_contact_mm)[0]
    else:
        n = int(vertices_laa.shape[0])
        k = max(int(n * closest_quantile), min_ostium_points)
        k = min(k, n)
        idx_sorted = np.argsort(distances)[:k]
        idx_cap = idx_sorted[distances[idx_sorted] <= ostium_abs_cap_mm]
        idx = idx_cap if idx_cap.size >= min_ostium_points else idx_sorted

    return vertices_laa[idx], idx


def _closest_distances_chunked(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    chunk_size: int,
    min_chunk_size: int = 250,
) -> np.ndarray:
    """Compute closest-point distances in adaptive chunks to reduce peak memory."""
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n == 0:
        return np.empty((0,), dtype=float)

    chunk = max(int(chunk_size), int(min_chunk_size))
    d = np.empty((n,), dtype=float)
    i = 0
    while i < n:
        j = min(i + chunk, n)
        try:
            _, dd, _ = trimesh.proximity.closest_point(mesh, pts[i:j])
            d[i:j] = np.asarray(dd, dtype=float)
            i = j
        except Exception:  # noqa: BLE001
            # Reduce chunk adaptively on memory-heavy cases; fail only at floor chunk.
            if chunk <= int(min_chunk_size):
                raise
            chunk = max(int(min_chunk_size), chunk // 2)
    return d


def compute_metrics(
    mesh_laa: trimesh.Trimesh,
    mesh_la: trimesh.Trimesh,
    params: dict | None = None,
) -> pd.DataFrame:
    cfg = _merge_params(params)
    out = _empty_result()

    try:
        v_laa = np.asarray(mesh_laa.vertices, dtype=float)
        v_la = np.asarray(mesh_la.vertices, dtype=float)
        c_laa = v_laa.mean(axis=0)
        c_la = v_la.mean(axis=0)

        out["laa_surface_area_mm2"] = float(mesh_laa.area)
        out["laa_volume_mm3"] = safe_mesh_volume(mesh_laa)
        out["la_surface_area_mm2"] = float(mesh_la.area)
        out["la_volume_mm3"] = safe_mesh_volume(mesh_la)

        # A) LAA->LA closest distances (chunked to avoid high peak memory).
        d = _closest_distances_chunked(
            mesh=mesh_la,
            points=v_laa,
            chunk_size=int(cfg["closest_chunk_size"]),
        )
        min_dist = float(np.nanmin(d))
        out["min_distance_mm"] = min_dist

        if min_dist > float(cfg["max_gap_fail_mm"]):
            out["qc_far_apart"] = True
            return pd.DataFrame([out])

        p, idx = _closest_interface_points(v_laa, d, cfg)
        out["ostium_points_n"] = int(p.shape[0])
        if p.shape[0] > 0:
            out["ostium_dist_median_mm"] = float(np.median(d[idx]))

        if p.shape[0] < int(cfg["min_ostium_points"]):
            out["qc_ostium_too_few_points"] = True
            return pd.DataFrame([out])

        # B) Ostium plane via PCA.
        c = p.mean(axis=0)
        eigvals, eigvecs = pca_eig(p)
        n = normalize(eigvecs[:, 0])
        if np.isnan(n).any():
            out["qc_ostium_too_few_points"] = True
            return pd.DataFrame([out])

        if float(np.dot(n, (c_laa - c_la))) < 0:
            n = -n

        lam_sum = float(np.sum(eigvals))
        planarity = float(eigvals[0] / lam_sum) if lam_sum > EPS else float("nan")
        _assign_xyz(out, "ostium_center", c)
        _assign_xyz(out, "ostium_normal", n)
        out["ostium_planarity"] = planarity

        # C) Tip and main LAA axis.
        t = (v_laa - c) @ n
        tip_idx = int(np.argmax(t))
        t_tip = float(t[tip_idx])
        tip = v_laa[tip_idx]

        if t_tip <= 0:
            out["qc_tip_wrong_side"] = True
            return pd.DataFrame([out])

        axis_vec = tip - c
        axis_len = float(np.linalg.norm(axis_vec))
        a = normalize(axis_vec)
        _assign_xyz(out, "laa_axis", a)
        out["laa_axis_length_mm"] = axis_len

        # D) Proximal direction.
        proximal_mask = (t > 0) & (t < float(cfg["proximal_length_mm"]))
        proximal_pts = v_laa[proximal_mask]
        if proximal_pts.shape[0] < 50:
            p1 = a.copy()
        else:
            _, prox_vecs = pca_eig(proximal_pts)
            p1 = normalize(prox_vecs[:, -1])
            if np.isnan(p1).any():
                p1 = a.copy()
            elif float(np.dot(p1, a)) < 0:
                p1 = -p1
        _assign_xyz(out, "laa_prox_dir", p1)

        # E) LA reference axis.
        _, la_vecs = pca_eig(v_la)
        la1 = normalize(la_vecs[:, -1])
        if not np.isnan(la1).any() and float(np.dot(la1, (c_laa - c_la))) < 0:
            la1 = -la1

        # F) Angles.
        out["bend_ostiumNormal_vs_proxLAA_deg"] = angle_deg(n, p1)
        out["bend_LAaxis_vs_LAAaxis_deg"] = angle_deg(la1, a)

        return pd.DataFrame([out])
    except Exception as exc:  # noqa: BLE001
        out["qc_exception"] = True
        out["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame([out])


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute minimum viable LA/LAA relational mesh metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--la", required=True, help="Path to LA mesh")
    p.add_argument("--laa", required=True, help="Path to LAA mesh")
    p.add_argument("--out", required=True, help="Output CSV (one-row)")

    p.add_argument("--near-contact-mm", type=float, default=DEFAULT_PARAMS["near_contact_mm"])
    p.add_argument("--closest-quantile", type=float, default=DEFAULT_PARAMS["closest_quantile"])
    p.add_argument("--max-gap-fail-mm", type=float, default=DEFAULT_PARAMS["max_gap_fail_mm"])
    p.add_argument("--min-ostium-points", type=int, default=int(DEFAULT_PARAMS["min_ostium_points"]))
    p.add_argument("--proximal-length-mm", type=float, default=DEFAULT_PARAMS["proximal_length_mm"])
    p.add_argument("--ostium-abs-cap-mm", type=float, default=DEFAULT_PARAMS["ostium_abs_cap_mm"])
    p.add_argument(
        "--repair-la-vtk-holes",
        action="store_true",
        help="If LA mesh is .vtk/.vtp, apply VTK hole capping before metric computation.",
    )
    p.add_argument(
        "--la-hole-size",
        type=float,
        default=50.0,
        help="VTK FillHolesFilter size used with --repair-la-vtk-holes.",
    )
    p.add_argument(
        "--la-repair-mode",
        choices=["fill_holes", "inferior_cap"],
        default="fill_holes",
        help="LA repair strategy when --repair-la-vtk-holes is enabled.",
    )
    p.add_argument(
        "--la-inferior-band-mm",
        type=float,
        default=15.0,
        help="Only for --la-repair-mode inferior_cap: cap loops with centroid Z <= (Zmin + band).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_cli()

    params = {
        "near_contact_mm": float(args.near_contact_mm),
        "closest_quantile": float(args.closest_quantile),
        "max_gap_fail_mm": float(args.max_gap_fail_mm),
        "min_ostium_points": int(args.min_ostium_points),
        "proximal_length_mm": float(args.proximal_length_mm),
        "ostium_abs_cap_mm": float(args.ostium_abs_cap_mm),
    }

    la_path = Path(args.la)
    repair_qc: dict[str, Any] = {}
    if args.repair_la_vtk_holes and la_path.suffix.lower() in {".vtk", ".vtp"}:
        mesh_la, repair_qc = load_mesh_vtk_hole_capped(
            str(la_path),
            hole_size=float(args.la_hole_size),
            repair_mode=str(args.la_repair_mode),
            inferior_band_mm=float(args.la_inferior_band_mm),
        )
    else:
        mesh_la = load_mesh(str(la_path))
    mesh_laa = load_mesh(args.laa)
    df = compute_metrics(mesh_laa=mesh_laa, mesh_la=mesh_la, params=params)
    for k, v in repair_qc.items():
        df[k] = v

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved metrics: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
