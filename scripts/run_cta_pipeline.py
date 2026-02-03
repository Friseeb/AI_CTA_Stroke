#!/usr/bin/env python3
"""
Generic CTA centerline extraction runner (single file or batch directory).

Usage examples:
- Single file: python -u scripts/run_cta_pipeline.py --input /path/to/cta.nii.gz --output outputs/run1
- Batch folder: python -u scripts/run_cta_pipeline.py --input /path/to/folder --output outputs/run_batch [--recursive]
- With defacing: python -u scripts/run_cta_pipeline.py --input /path/to/cta.nii.gz --output outputs/run1 --deface

Pipeline steps:
1. Defacing (optional) - Remove facial features using TotalSegmentator
2. Vessel mask creation - Segment vessels from CTA
3. Centerline extraction - Extract vessel centerlines (MPS/CUDA accelerated)
4. Radius/bifurcation/graph - Build vessel graph with radii and bifurcations

Notes:
- For batch mode, each input file gets its own subfolder under --output named after the CTA filename.
- Streams logs to stdout (line-buffered) for real-time monitoring.
- Converts outputs optionally to EVC and ArterialGNet formats.
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, label

# Ensure repo root on import path
REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Line-buffered stdout for real-time logs
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from python.analysis.centerline_antiga_2008 import CenterlineExtractionPipeline
from python.analysis.converters import (
    centerline_to_evc_graph,
    centerline_to_arterial_gnet_graph,
)

# Import defacing module
try:
    from scripts.deface_cta import deface_volume, run_totalsegmentator
    DEFACE_AVAILABLE = True
except ImportError:
    DEFACE_AVAILABLE = False


def export_centerline_mask(
    centerline_graph,
    reference_img: nib.Nifti1Image,
    output_path: Path,
    step: float = 0.5,
    dilation_iters: int = 0,
):
    """Rasterize centerline graph into a binary mask aligned to reference_img."""

    mask = np.zeros(reference_img.shape, dtype=np.uint8)
    bounds = np.array(reference_img.shape) - 1

    def _stamp_points(points: np.ndarray):
        if points.size == 0:
            return
        pts = np.round(points).astype(int)
        pts = np.clip(pts, 0, bounds)
        for pt in pts:
            mask[tuple(pt)] = 1

    segments = {}
    for _, node_data in centerline_graph.nodes(data=True):
        seg_id = node_data.get('segment_id')
        if seg_id is None:
            continue
        segments.setdefault(seg_id, []).append(
            (node_data.get('segment_index', 0), np.array(node_data.get('position', [0, 0, 0]), dtype=float))
        )

    for seg_nodes in segments.values():
        seg_nodes.sort(key=lambda x: x[0])
        pts = np.stack([p for _, p in seg_nodes]) if seg_nodes else np.empty((0, 3))
        _stamp_points(pts)

        for a, b in zip(pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(b - a))
            n_steps = max(2, int(np.ceil(seg_len / max(step, 1e-3))))
            interp = np.linspace(a, b, n_steps)
            _stamp_points(interp)

    if dilation_iters > 0:
        mask = binary_dilation(mask, iterations=dilation_iters).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, reference_img.affine, reference_img.header), str(output_path))
    return mask.sum()


def _load_union_mask(mask_paths, reference_img: nib.Nifti1Image) -> np.ndarray:
    if not mask_paths:
        return None
    union = np.zeros(reference_img.shape, dtype=bool)
    for path in mask_paths:
        mask_img = nib.load(str(path))
        mask_data = mask_img.get_fdata() > 0
        if mask_data.shape != reference_img.shape:
            raise ValueError(f"Mask shape mismatch: {path} ({mask_data.shape}) vs {reference_img.shape}")
        union |= mask_data
    return union


def _filter_centerline_graph_by_mask(centerline_graph, mask_union: np.ndarray):
    if mask_union is None:
        return centerline_graph, 0
    shape = mask_union.shape
    to_remove = []
    for node_id, node_data in centerline_graph.nodes(data=True):
        pos = node_data.get('position')
        if pos is None:
            continue
        idx = np.rint(np.array(pos, dtype=float)).astype(int)
        if np.any(idx < 0) or np.any(idx >= shape):
            continue
        if mask_union[tuple(idx)]:
            to_remove.append(node_id)
    if not to_remove:
        return centerline_graph, 0
    filtered = centerline_graph.copy()
    filtered.remove_nodes_from(to_remove)
    return filtered, len(to_remove)


def _graph_to_data(centerline_graph):
    node_data = []
    for node_id, node_attrs in centerline_graph.nodes(data=True):
        node_data.append({
            'node_id': node_id,
            'position': node_attrs.get('position'),
            'radius': float(node_attrs.get('radius', 0.0)),
            'segment_id': node_attrs.get('segment_id'),
            'segment_index': node_attrs.get('segment_index'),
        })
    edge_data = []
    for u, v, edge_attrs in centerline_graph.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'distance': float(edge_attrs.get('distance', 0.0)),
            'type': edge_attrs.get('type', 'centerline'),
        })
    return node_data, edge_data


def is_binary_mask(data: np.ndarray) -> bool:
    """Heuristic to detect if the input is already a binary mask."""
    unique_vals = np.unique(data)
    if unique_vals.size <= 3 and unique_vals.min() >= 0 and unique_vals.max() <= 1:
        return True
    return False


def _totalseg_label_path(totalseg_dir: Path, label: str) -> Path | None:
    if (totalseg_dir / f"{label}.nii.gz").exists():
        return totalseg_dir / f"{label}.nii.gz"
    if (totalseg_dir / label).exists():
        return totalseg_dir / label
    return None


def _load_totalseg_label_mask(
    totalseg_dir: Path,
    label: str,
    reference_shape: tuple | None = None,
) -> tuple[np.ndarray | None, any, any]:
    path = _totalseg_label_path(totalseg_dir, label)
    if path is None:
        return None, None, None
    img = nib.load(str(path))
    data = img.get_fdata() > 0
    if reference_shape is not None and data.shape != reference_shape:
        print(f"  ⚠ Shape mismatch for {label}: {data.shape} vs {reference_shape}")
        return None, None, None
    return data.astype(np.uint8), img.affine, img.header


def _export_totalseg_label_mask(
    totalseg_dir: Path,
    label: str,
    output_path: Path,
    reference_shape: tuple | None = None,
) -> bool:
    mask, affine, header = _load_totalseg_label_mask(totalseg_dir, label, reference_shape)
    if mask is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, header), str(output_path))
    print(f"✓ Saved {label} mask: {output_path}")
    return True


def _has_any_totalseg_labels(totalseg_dir: Path, labels: list[str]) -> bool:
    return any(_totalseg_label_path(totalseg_dir, label) for label in labels)


def _missing_totalseg_labels(totalseg_dir: Path, labels: list[str]) -> list[str]:
    missing = []
    for label in labels:
        if _totalseg_label_path(totalseg_dir, label) is None:
            missing.append(label)
    return missing


# Head/neck specific vessel structures from TotalSegmentator
# Split by task because label availability differs between tasks.
HEAD_NECK_TOTAL_LABELS = [
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "subclavian_artery_right",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
]

HEAD_NECK_HEADNECK_LABELS = [
    "internal_carotid_artery_left",
    "internal_carotid_artery_right",
    "internal_jugular_vein_left",
    "internal_jugular_vein_right",
]

HEAD_NECK_VESSEL_STRUCTURES = HEAD_NECK_TOTAL_LABELS + HEAD_NECK_HEADNECK_LABELS

# Structures to EXCLUDE (chest/cardiac vessels)
EXCLUDE_VESSEL_STRUCTURES = [
    "aorta",
    "heart",
    "pulmonary_vein",
    "superior_vena_cava",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
]

# Cardiac labels for export (not used for centerline extraction)
DEFAULT_LAA_LABEL = "atrial_appendage_left"
DEFAULT_LA_LABEL = "heart_atrium_left"


def load_totalseg_vessel_mask(
    totalseg_dir: Path,
    structures: list[str],
    reference_shape: tuple,
) -> tuple[np.ndarray | None, any, any]:
    """Load and combine vessel masks from TotalSegmentator output."""
    combined = None
    affine = None
    header = None
    found = []
    missing = []

    for name in structures:
        # Try with .nii.gz extension
        path = totalseg_dir / f"{name}.nii.gz"
        if not path.exists():
            path = totalseg_dir / name
        if not path.exists():
            missing.append(name)
            continue

        img = nib.load(str(path))
        data = img.get_fdata() > 0

        if data.shape != reference_shape:
            print(f"  ⚠ Shape mismatch for {name}: {data.shape} vs {reference_shape}")
            continue

        found.append(name)
        if combined is None:
            combined = data.astype(np.uint8)
            affine = img.affine
            header = img.header
        else:
            combined |= data.astype(np.uint8)

    if found:
        print(f"  ✓ Found: {', '.join(found)}")
    if missing:
        print(f"  ⚠ Missing: {', '.join(missing)}")

    return combined, affine, header


def create_headneck_vessel_mask(
    cta_path: Path,
    totalseg_dir: Path,
    output_path: Path | None = None,
    include_hu_vessels: bool = True,
    hu_threshold: int = 200,
    hu_max: int = 600,
    min_component_size: int = 100,
    exclude_cardiac: bool = True,
    roi_dilation: int = 50,
) -> tuple[np.ndarray, any, any]:
    """
    Create vessel mask focused on head/neck vessels using TotalSegmentator.

    This approach uses TotalSegmentator's anatomical vessel labels to specifically
    target head/neck vessels (carotids, vertebrals, basilar) while excluding
    cardiac and chest structures.

    Parameters
    ----------
    cta_path : Path
        Path to CTA NIfTI file
    totalseg_dir : Path
        TotalSegmentator output directory
    output_path : Path, optional
        Output path for vessel mask
    include_hu_vessels : bool
        Also include HU-thresholded vessels within the anatomical ROI
    hu_threshold : int
        Lower HU threshold for contrast-enhanced vessels
    hu_max : int
        Upper HU threshold (to exclude bone)
    min_component_size : int
        Minimum connected component size
    exclude_cardiac : bool
        Exclude cardiac/chest vessels from the mask
    roi_dilation : int
        Dilation iterations for anatomical ROI (larger = more permissive)

    Returns
    -------
    tuple
        (mask, affine, header)
    """
    print(f"Loading CTA: {cta_path}")
    cta = nib.load(str(cta_path))
    data = cta.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {cta.header.get_zooms()}")

    # Check if input is already a binary mask
    if is_binary_mask(data):
        print("  Input is already a binary mask")
        mask = (data > 0).astype(np.uint8)
        if output_path:
            nib.save(nib.Nifti1Image(mask, cta.affine, cta.header), str(output_path))
        return mask, cta.affine, cta.header

    print("\n  Loading TotalSegmentator head/neck vessel labels...")
    vessel_mask, _, _ = load_totalseg_vessel_mask(
        totalseg_dir, HEAD_NECK_VESSEL_STRUCTURES, data.shape
    )

    if vessel_mask is None:
        print("  ⚠ No TotalSegmentator vessel masks found, using HU threshold only")
        vessel_mask = np.zeros(data.shape, dtype=np.uint8)

    # Add HU-thresholded vessels
    if include_hu_vessels:
        print(f"\n  Adding HU-thresholded vessels ({hu_threshold}-{hu_max} HU)...")
        hu_mask = (data > hu_threshold) & (data < hu_max)

        # If we have anatomical vessel masks, use them as ROI but be more permissive
        if vessel_mask.sum() > 0 and roi_dilation > 0:
            from scipy.ndimage import binary_dilation
            # Use a larger dilation to capture intracranial extensions
            roi = binary_dilation(vessel_mask > 0, iterations=roi_dilation)
            hu_mask_roi = hu_mask & roi
            print(f"    Anatomical ROI ({roi_dilation} dilation): {roi.sum():,} voxels")
            print(f"    HU vessels in ROI: {hu_mask_roi.sum():,} voxels")
            vessel_mask = (vessel_mask | hu_mask_roi.astype(np.uint8))
        else:
            # No anatomical guidance, use all HU-thresholded vessels
            print("    Using all HU-thresholded vessels (no anatomical ROI)")
            vessel_mask = (vessel_mask | hu_mask.astype(np.uint8))

    # Exclude cardiac/chest vessels
    if exclude_cardiac:
        print("\n  Excluding cardiac/chest structures...")
        exclude_mask, _, _ = load_totalseg_vessel_mask(
            totalseg_dir, EXCLUDE_VESSEL_STRUCTURES, data.shape
        )
        if exclude_mask is not None:
            # Minimal dilation - just remove the cardiac structures, not surrounding tissue
            exclude_mask = binary_dilation(exclude_mask > 0, iterations=2)
            before = vessel_mask.sum()
            vessel_mask = vessel_mask & ~exclude_mask.astype(np.uint8)
            print(f"    Removed {before - vessel_mask.sum():,} voxels")

    # Filter small components
    print(f"\n  Filtering small components (< {min_component_size} voxels)...")
    labeled, num_features = label(vessel_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        large_components = np.where(component_sizes > min_component_size)[0]
        large_components = large_components[large_components > 0]
        vessel_mask = np.isin(labeled, large_components).astype(np.uint8)
        print(f"    Kept {len(large_components)} components, {vessel_mask.sum():,} voxels")

    # Morphological cleaning - lighter touch
    print("  Morphological cleaning (light)...")
    # Skip erosion to preserve thin vessels
    # Just use the mask as-is

    print(f"\n  Final vessel mask: {vessel_mask.sum():,} voxels")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(vessel_mask, cta.affine, cta.header), str(output_path))
        print(f"✓ Saved vessel mask: {output_path}")

    return vessel_mask, cta.affine, cta.header


def create_vessel_mask_from_cta(
    cta_path: Path,
    threshold_hu: int = 150,
    max_hu: int | None = None,
    bone_hu: int = 900,
    strip_boundary_bone: bool = True,
    boundary_margin_mm: float = 6.0,
    bone_mask_path: Path | None = None,
    vessel_mask_path: Path | None = None,
    min_component_size: int = 500,
    apply_bone_mask_early: bool = False,
    output_path: Path | None = None,
):
    print(f"Loading CTA: {cta_path}")
    cta = nib.load(str(cta_path))
    data = cta.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {cta.header.get_zooms()}")
    print(f"  Intensity range: {data.min():.1f} to {data.max():.1f} HU")

    # Check if input is already a binary mask
    if is_binary_mask(data):
        print("\n  Input detected as binary mask; using directly (no HU thresholding)")
        mask = (data > 0).astype(np.uint8)
        if output_path:
            nib.save(nib.Nifti1Image(mask, cta.affine, cta.header), str(output_path))
            print(f"✓ Saved vessel mask: {output_path}")
        return mask, cta.affine, cta.header

    upper_desc = f" and < {max_hu}" if max_hu is not None else ""
    print(f"\nCreating vessel mask (HU > {threshold_hu}{upper_desc}, removing boundary bone >= {bone_hu})...")
    bandpass = data > threshold_hu
    if max_hu is not None:
        bandpass &= data < max_hu

    # Optional explicit bone mask removal (early application - may exclude vertebral arteries in vertebral foramina)
    # For head/neck CTA with vertebral arteries: set apply_bone_mask_early=False to preserve VA segments
    bone_mask_data = None
    if apply_bone_mask_early and bone_mask_path:
        bone_mask_img = nib.load(str(bone_mask_path))
        bone_mask_data = bone_mask_img.get_fdata() > 0
        if bone_mask_data.shape != data.shape:
            raise ValueError("Bone mask shape does not match CTA shape")
        bandpass &= ~bone_mask_data
        print(f"  Applied bone mask early (WARNING: may exclude vertebral artery in foramina)")
    elif bone_mask_path:
        print("  Bone mask provided but not applied early; using HU-only boundary stripping to preserve VA")

    # Optional explicit vessel mask union (e.g., from TotalSegmentator vascular labels)
    if vessel_mask_path:
        vessel_mask_img = nib.load(str(vessel_mask_path))
        vessel_mask_data = vessel_mask_img.get_fdata() > 0
        if vessel_mask_data.shape != data.shape:
            raise ValueError("Vessel mask shape does not match CTA shape")
        bandpass = bandpass | vessel_mask_data

    if strip_boundary_bone:
        # Hard bone exclusion only for boundary-connected bone so we keep internal arterial calcifications.
        bone_mask = (data >= bone_hu).astype(np.uint8)
        print(f"  Stripping boundary bone using HU >= {bone_hu}")
        labeled_bone, _ = label(bone_mask)

        # Identify bone components touching volume boundary (skull, vertebrae, sternum, ribs, sternum)
        boundary_ids = set()
        faces = [
            labeled_bone[0, :, :], labeled_bone[-1, :, :],
            labeled_bone[:, 0, :], labeled_bone[:, -1, :],
            labeled_bone[:, :, 0], labeled_bone[:, :, -1],
        ]
        for face in faces:
            boundary_ids.update(np.unique(face))
        boundary_ids.discard(0)

        # Also remove high-HU voxels within a physical margin from the volume boundary (captures ribs/sternum shell)
        if boundary_margin_mm > 0:
            vx, vy, vz = cta.header.get_zooms()
            margin_vx = max(1, int(round(boundary_margin_mm / min(vx, vy, vz))))
            boundary_shell = np.zeros_like(bone_mask, dtype=bool)
            boundary_shell[:margin_vx, :, :] = True
            boundary_shell[-margin_vx:, :, :] = True
            boundary_shell[:, :margin_vx, :] = True
            boundary_shell[:, -margin_vx:, :] = True
            boundary_shell[:, :, :margin_vx] = True
            boundary_shell[:, :, -margin_vx:] = True
            boundary_shell &= bone_mask.astype(bool)
            shell_ids = np.unique(labeled_bone[boundary_shell])
            shell_ids = set(shell_ids.tolist())
            shell_ids.discard(0)
            boundary_ids.update(shell_ids)

        if boundary_ids:
            boundary_bone = np.isin(labeled_bone, list(boundary_ids))
            bandpass &= ~boundary_bone

    mask = bandpass.astype(np.uint8)
    print(f"  Initial vessel voxels: {np.sum(mask):,}")

    print("  Filtering small components...")
    labeled, _ = label(mask)
    component_sizes = np.bincount(labeled.ravel())
    large_components = np.where(component_sizes > min_component_size)[0]
    large_components = large_components[large_components > 0]
    mask_filtered = np.isin(labeled, large_components).astype(np.uint8)
    print(f"  After filtering: {np.sum(mask_filtered):,} voxels ({len(large_components)} components)")

    print("  Morphological cleaning...")
    mask_clean = binary_erosion(mask_filtered, iterations=1)
    mask_clean = binary_dilation(mask_clean, iterations=1).astype(np.uint8)
    print(f"  Final vessel voxels: {np.sum(mask_clean):,}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img = nib.Nifti1Image(mask_clean, cta.affine, cta.header)
        nib.save(mask_img, str(output_path))
        print(f"✓ Saved vessel mask: {output_path}")

    return mask_clean, cta.affine, cta.header


def extract_morphology_features(centerline_graph):
    features = {}
    segments = {}
    for _, node_data in centerline_graph.nodes(data=True):
        seg_id = node_data.get('segment_id')
        if seg_id is None:
            # Skip nodes without segment_id to avoid KeyErrors
            continue
        if seg_id not in segments:
            segments[seg_id] = {'positions': [], 'radii': []}
        segments[seg_id]['positions'].append(node_data['position'])
        segments[seg_id]['radii'].append(node_data['radius'])

    if segments:
        all_radii = np.concatenate([s['radii'] for s in segments.values()])
        features['total_nodes'] = centerline_graph.number_of_nodes()
        features['total_edges'] = centerline_graph.number_of_edges()
        features['num_segments'] = len(segments)
        features['mean_radius_mm'] = float(np.mean(all_radii))
        features['std_radius_mm'] = float(np.std(all_radii))
        features['min_radius_mm'] = float(np.min(all_radii))
        features['max_radius_mm'] = float(np.max(all_radii))
        features['median_radius_mm'] = float(np.median(all_radii))

        all_diameters = all_radii * 2
        features['mean_diameter_mm'] = float(np.mean(all_diameters))
        features['median_diameter_mm'] = float(np.median(all_diameters))

        segment_lengths = [len(s['positions']) for s in segments.values()]
        features['mean_segment_length_nodes'] = float(np.mean(segment_lengths))
        features['median_segment_length_nodes'] = float(np.median(segment_lengths))
        features['max_segment_length_nodes'] = int(np.max(segment_lengths))

        total_length_mm = 0
        for seg_data in segments.values():
            positions = np.array(seg_data['positions'])
            if len(positions) > 1:
                segment_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                total_length_mm += segment_length
        features['total_centerline_length_mm'] = float(total_length_mm)

        degrees = dict(centerline_graph.degree())
        features['num_bifurcations'] = sum(1 for d in degrees.values() if d > 2)

        tortuosities = []
        for seg_data in segments.values():
            positions = np.array(seg_data['positions'])
            if len(positions) > 2:
                path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                straight_dist = np.linalg.norm(positions[-1] - positions[0])
                if straight_dist > 1.0:
                    tortuosities.append(path_length / straight_dist)
        if tortuosities:
            features['mean_tortuosity'] = float(np.mean(tortuosities))
            features['median_tortuosity'] = float(np.median(tortuosities))
            features['max_tortuosity'] = float(np.max(tortuosities))

    return features


def strip_nii_suffix(path: Path) -> str:
    """Return filename stem without .nii or .nii.gz."""
    name = path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return path.stem


def collect_inputs(input_path: Path, recursive: bool = False) -> list[Path]:
    """Resolve input path to a list of CTA files (supports single file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        patterns = ['**/*.nii.gz', '**/*.nii'] if recursive else ['*.nii.gz', '*.nii']
        files = []
        for pattern in patterns:
            files.extend(sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern)))
        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        return unique_files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run_single_cta(cta_path: Path, output_dir: Path, args) -> int:
    """Run the full CTA → centerline → conversions pipeline for one file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track the working CTA path (may change after defacing)
    working_cta_path = cta_path
    totalseg_dir = Path(args.totalseg_dir) if args.totalseg_dir else output_dir / 'totalseg'
    heartchambers_dir = (
        Path(args.heartchambers_dir)
        if args.heartchambers_dir
        else totalseg_dir / "heartchambers_highres"
    )

    totalseg_total_labels = []
    totalseg_headneck_labels = []
    if args.deface:
        totalseg_total_labels.append("face")
    if args.head_neck_mode and (args.run_totalseg or args.deface):
        totalseg_total_labels.extend(HEAD_NECK_TOTAL_LABELS)
        totalseg_headneck_labels.extend(HEAD_NECK_HEADNECK_LABELS)
        if args.exclude_cardiac:
            totalseg_total_labels.extend(EXCLUDE_VESSEL_STRUCTURES)
    if args.export_laa:
        totalseg_total_labels.append(args.laa_label)

    heartchambers_labels = []
    if args.export_la:
        heartchambers_labels.append(args.la_label)

    totalseg_total_labels = sorted(set(totalseg_total_labels))
    totalseg_headneck_labels = sorted(set(totalseg_headneck_labels))

    if totalseg_total_labels:
        missing_labels = _missing_totalseg_labels(totalseg_dir, totalseg_total_labels)
        if missing_labels:
            if args.run_totalseg or args.deface:
                print("\n" + "=" * 50)
                print("PREP: TOTALSEGMENTATOR (TOTAL)")
                print("=" * 50)
                print(f"  Generating {len(missing_labels)} missing label(s)...")
                run_totalsegmentator(
                    input_path=cta_path,
                    output_dir=totalseg_dir,
                    fast=True,
                    roi_subset=totalseg_total_labels,
                    task="total",
                )
            else:
                print("  ⚠ TotalSegmentator labels missing; pass --run-totalseg or set --totalseg-dir")

    if totalseg_headneck_labels:
        missing_hn = _missing_totalseg_labels(totalseg_dir, totalseg_headneck_labels)
        if missing_hn:
            if args.run_totalseg or args.deface:
                print("\n" + "=" * 50)
                print("PREP: TOTALSEGMENTATOR (HEAD/NECK)")
                print("=" * 50)
                print(f"  Generating {len(missing_hn)} missing label(s)...")
                run_totalsegmentator(
                    input_path=cta_path,
                    output_dir=totalseg_dir,
                    fast=False,
                    roi_subset=totalseg_headneck_labels,
                    task="headneck_bones_vessels",
                )
            else:
                print("  ⚠ Head/neck labels missing; pass --run-totalseg or set --totalseg-dir")

    if heartchambers_labels:
        missing_hc = _missing_totalseg_labels(heartchambers_dir, heartchambers_labels)
        if missing_hc:
            if args.run_totalseg:
                print("\n" + "=" * 50)
                print("PREP: TOTALSEGMENTATOR (HEARTCHAMBERS)")
                print("=" * 50)
                print(f"  Generating {len(missing_hc)} missing label(s)...")
                run_totalsegmentator(
                    input_path=cta_path,
                    output_dir=heartchambers_dir,
                    fast=False,
                    roi_subset=heartchambers_labels,
                    task="heartchambers_highres",
                )
            else:
                print("  ⚠ Heartchambers labels missing; pass --run-totalseg or set --heartchambers-dir")

    # Step 1: Defacing (optional)
    if args.deface:
        if not DEFACE_AVAILABLE:
            raise RuntimeError("Defacing module not available. Check scripts/deface_cta.py exists.")

        print("\n" + "=" * 50)
        print("STEP 1: DEFACING")
        print("=" * 50)

        defaced_path = output_dir / 'defaced_cta.nii.gz'

        deface_result = deface_volume(
            input_path=cta_path,
            output_path=defaced_path,
            totalseg_dir=totalseg_dir,
            run_totalseg=False,  # Already ran above if needed
            dilation_mm=args.deface_dilation_mm,
            fill_value=args.deface_fill_value,
            save_mask=args.save_deface_mask,
        )
        print(f"  Defaced {deface_result['defaced_voxels']:,} voxels")
        working_cta_path = defaced_path
    else:
        print("\n(Skipping defacing - use --deface to enable)")

    print("\n" + "=" * 50)
    print("STEP 2: VESSEL MASK CREATION")
    print("=" * 50)

    mask_path = output_dir / 'vessel_mask.nii.gz'

    if args.head_neck_mode:
        print("  Mode: HEAD/NECK (TotalSegmentator-based)")

        if not _has_any_totalseg_labels(totalseg_dir, HEAD_NECK_VESSEL_STRUCTURES):
            print("  ⚠ TotalSegmentator vessel labels not found, falling back to HU threshold mode")
            args.head_neck_mode = False

        if args.head_neck_mode:
            _mask, _affine, _hdr = create_headneck_vessel_mask(
                cta_path=working_cta_path,
                totalseg_dir=totalseg_dir,
                output_path=mask_path,
                include_hu_vessels=True,
                hu_threshold=args.threshold,
                hu_max=args.max_hu if args.max_hu else 600,
                min_component_size=args.min_component_size,
                exclude_cardiac=args.exclude_cardiac,
            )

    if not args.head_neck_mode:
        print("  Mode: HU THRESHOLD (legacy)")
        print("  ⚠ This may include heart/aorta. Use --head-neck-mode for better results.")
        _mask, _affine, _hdr = create_vessel_mask_from_cta(
            cta_path=working_cta_path,
            threshold_hu=args.threshold,
            max_hu=args.max_hu,
            bone_hu=args.bone_hu,
            strip_boundary_bone=args.strip_boundary_bone,
            boundary_margin_mm=args.boundary_margin_mm,
            bone_mask_path=Path(args.bone_mask) if args.bone_mask else None,
            vessel_mask_path=Path(args.vessel_mask) if args.vessel_mask else None,
            apply_bone_mask_early=args.apply_bone_mask_early,
            min_component_size=args.min_component_size,
            output_path=mask_path,
        )

    if args.export_la or args.export_laa:
        print("\n" + "=" * 50)
        print("STEP 2B: CARDIAC MASK EXPORT")
        print("=" * 50)
        ref_shape = nib.load(str(working_cta_path)).shape
        cardiac_dir = output_dir / "cardiac_masks"
        if args.export_laa:
            ok = _export_totalseg_label_mask(
                totalseg_dir,
                args.laa_label,
                cardiac_dir / "left_atrial_appendage.nii.gz",
                reference_shape=ref_shape,
            )
            if not ok:
                print("  ⚠ LAA mask not found; check TotalSegmentator outputs")
        if args.export_la:
            ok = _export_totalseg_label_mask(
                heartchambers_dir,
                args.la_label,
                cardiac_dir / "left_atrium.nii.gz",
                reference_shape=ref_shape,
            )
            if not ok:
                print("  ⚠ LA mask not found; check heartchambers_highres outputs")

    topcow_seg_path = None
    if args.run_topcow:
        print("\n" + "=" * 50)
        print("STEP 2C: TOPCOW INTRACRANIAL SEGMENTATION")
        print("=" * 50)

        if not args.topcow_yolo_model or not args.topcow_nnunet_model_dir:
            raise RuntimeError(
                "TopCoW requested but --topcow-yolo-model or --topcow-nnunet-model-dir not provided."
            )

        topcow_out = Path(args.topcow_output) if args.topcow_output else (output_dir / "topcow")
        topcow_labels_json = (
            Path(args.topcow_labels_json)
            if args.topcow_labels_json
            else topcow_out / "topcow_labels.json"
        )

        try:
            from scripts.run_topcow_claim import run_inference
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to import TopCoW wrapper. Check dependencies.") from exc

        outputs = run_inference(
            input_path=working_cta_path,
            output_dir=topcow_out,
            yolo_model_path=Path(args.topcow_yolo_model),
            nnunet_model_dir=Path(args.topcow_nnunet_model_dir),
            device=args.topcow_device,
            work_dir=Path(args.topcow_work_dir) if args.topcow_work_dir else None,
            keep_temp=args.topcow_keep_temp,
            overwrite=args.topcow_overwrite,
            labels_json_path=topcow_labels_json,
            keep_label_13=args.topcow_keep_label_13,
        )
        if outputs:
            topcow_seg_path = outputs[0]
            print(f"TopCoW segmentation: {topcow_seg_path}")

    if args.build_multilabel:
        print("\n" + "=" * 50)
        print("STEP 2D: MULTI-LABEL MERGE (EXTRA/INTRACRANIAL)")
        print("=" * 50)

        try:
            from scripts.build_multilabel_vascular_map import build_multilabel_map
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to import multi-label builder.") from exc

        multilabel_out = (
            Path(args.multilabel_output)
            if args.multilabel_output
            else output_dir / "labels_multilabel.nii.gz"
        )
        multilabel_labels_json = (
            Path(args.multilabel_labels_json)
            if args.multilabel_labels_json
            else output_dir / "labels_multilabel.json"
        )

        topcow_path = Path(args.multilabel_topcow) if args.multilabel_topcow else None
        if topcow_path is None and topcow_seg_path is not None:
            topcow_path = topcow_seg_path

        build_multilabel_map(
            reference_path=working_cta_path,
            output_path=multilabel_out,
            totalseg_dir=totalseg_dir,
            headneck_dir=totalseg_dir,
            heartchambers_dir=heartchambers_dir if heartchambers_dir.exists() else None,
            label_config=Path(args.multilabel_label_config) if args.multilabel_label_config else None,
            labels_json_path=multilabel_labels_json,
            overwrite=args.multilabel_overwrite,
            topcow_path=topcow_path,
            topcow_mode=args.multilabel_topcow_mode,
            topcow_label=args.multilabel_topcow_label,
            topcow_offset=args.multilabel_topcow_offset,
            topcow_remap=Path(args.multilabel_topcow_remap) if args.multilabel_topcow_remap else None,
        )

    print("\n" + "=" * 50)
    print("STEP 3: CENTERLINE EXTRACTION (GPU ACCELERATED)")
    print("=" * 50)
    print(f"  Device: {args.device}, EDT backend: {args.edt_backend}")
    pipeline = CenterlineExtractionPipeline(
        nifti_path=str(mask_path),
        output_dir=str(output_dir / 'centerline'),
        log_level='WARNING',
        device=args.device,
    )

    crop_margin_vox = args.crop_margin_vox
    if any(
        val is not None for val in (args.crop_margin_vox_x, args.crop_margin_vox_y, args.crop_margin_vox_z)
    ):
        mx = args.crop_margin_vox_x if args.crop_margin_vox_x is not None else crop_margin_vox
        my = args.crop_margin_vox_y if args.crop_margin_vox_y is not None else crop_margin_vox
        mz = args.crop_margin_vox_z if args.crop_margin_vox_z is not None else crop_margin_vox
        crop_margin_vox = (mx, my, mz)

    results = pipeline.run(
        min_component_size=300,
        min_distance_value=args.min_distance_value,
        step_size=args.step_size,
        max_iterations=args.max_iterations,
        contact_distance_threshold=args.contact_distance_threshold,
        crop_margin_vox=crop_margin_vox,
        downsample_factor=args.downsample_factor,
        edt_backend=args.edt_backend,
        thick_component_max_radius=args.thick_component_max_radius,
        erosion_iterations=args.erosion_iterations,
        dilation_iterations=args.dilation_iterations,
        allow_cpu_edt=args.allow_cpu_edt,
        save_intermediates=args.save_intermediates,
    )

    centerline_graph = results['stage7']['graph']
    post_masks = []
    for mask_entry in args.post_centerline_mask or []:
        post_masks.extend([item.strip() for item in mask_entry.split(",") if item.strip()])
    if post_masks:
        vessel_mask_img = nib.load(str(mask_path))
        mask_union = _load_union_mask(post_masks, vessel_mask_img)
        centerline_graph, removed = _filter_centerline_graph_by_mask(centerline_graph, mask_union)
        if removed > 0:
            node_data, edge_data = _graph_to_data(centerline_graph)
            results['stage7'] = {
                'graph': centerline_graph,
                'node_data': node_data,
                'edge_data': edge_data,
            }
        print(f"  Post-centerline mask filter removed {removed} nodes")

    print("\n" + "=" * 50)
    print("STEP 4: RADIUS / BIFURCATION / GRAPH")
    print("=" * 50)
    summary = pipeline.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n  Extracting morphology features...")
    features = extract_morphology_features(results['stage7']['graph'])
    features_path = output_dir / 'morphology_features.json'
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"✓ Saved: {features_path}")

    if args.export_centerline_mask:
        print("\nRasterizing centerline mask...")
        vessel_mask_img = nib.load(str(mask_path))
        voxels_on = export_centerline_mask(
            results['stage7']['graph'],
            vessel_mask_img,
            output_dir / 'centerline_mask.nii.gz',
            step=args.centerline_step,
            dilation_iters=args.centerline_dilation,
        )
        print(f"  Centerline voxels: {voxels_on:,}")

    if args.convert_evc:
        print("\nConverting to EVC format...")
        evc_graph = centerline_to_evc_graph(
            results['stage7']['graph'],
            output_pickle_path=str(output_dir / 'evc_graph.pkl'),
            vessel_type_name='other',
        )
        print(f"  EVC graph: {evc_graph.number_of_nodes()} bifurcations, {evc_graph.number_of_edges()} vessel segments")

    if args.convert_arterial_gnet:
        print("\nConverting to ArterialGNet format...")
        gnet_result = centerline_to_arterial_gnet_graph(
            results['stage7']['graph'],
            output_pickle_path=str(output_dir / 'arterial_gnet_graph.pkl'),
            include_segment_graph=True,
        )
        print(f"  Dense graph: {gnet_result['dense_graph'].number_of_nodes():,} nodes")
        print(f"  Segment graph: {gnet_result['segment_graph'].number_of_nodes()} segments")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="CTA Centerline Extraction Runner")
    p.add_argument("--input", required=True, help="Path to CTA NIfTI (.nii/.nii.gz) or directory for batch processing")
    p.add_argument("--output", required=True, help="Output directory (single) or root directory (batch)")
    p.add_argument("--recursive", action="store_true", help="When input is a directory, search recursively for .nii/.nii.gz")

    # Step 1: Defacing options
    p.add_argument("--deface", action="store_true", help="Deface the CTA before processing (removes facial features)")
    p.add_argument("--deface-dilation-mm", type=float, default=0.0, help="Dilate face mask by this amount (mm)")
    p.add_argument("--deface-fill-value", type=float, default=-1024.0, help="Fill value for defaced region (-1024=air)")
    p.add_argument("--totalseg-dir", type=str, default=None, help="Pre-computed TotalSegmentator output directory (for defacing and vessel masks)")
    p.add_argument("--heartchambers-dir", type=str, default=None, help="Pre-computed TotalSegmentator heartchambers_highres output directory")
    p.add_argument("--save-deface-mask", action="store_true", help="Save the face mask used for defacing")

    # Step 2: Vessel mask options
    p.add_argument("--head-neck-mode", action="store_true", help="Use TotalSegmentator-based head/neck vessel targeting (RECOMMENDED - avoids heart/aorta)")
    p.add_argument("--run-totalseg", action="store_true", help="Run TotalSegmentator automatically if --totalseg-dir not provided")
    p.add_argument("--exclude-cardiac", action="store_true", default=True, help="Exclude cardiac/chest vessels in head-neck mode (default: True)")
    p.add_argument("--no-exclude-cardiac", dest="exclude_cardiac", action="store_false", help="Include cardiac/chest vessels")
    p.add_argument("--export-la", action="store_true", help="Export left atrium mask (TotalSegmentator heartchambers_highres)")
    p.add_argument("--export-laa", action="store_true", help="Export left atrial appendage mask (TotalSegmentator total)")
    p.add_argument("--la-label", type=str, default=DEFAULT_LA_LABEL, help="Label name for LA in heartchambers_highres output")
    p.add_argument("--laa-label", type=str, default=DEFAULT_LAA_LABEL, help="Label name for LAA in total output")
    p.add_argument("--threshold", type=int, default=150, help="Lower HU threshold for vessel mask")
    p.add_argument("--max-hu", type=int, default=700, help="Upper HU cutoff to suppress dense bone (None to disable)")
    p.add_argument("--bone-hu", type=int, default=900, help="Bone threshold; boundary-connected components >= this HU are excluded (arterial calcifications are preserved)")
    p.add_argument("--strip-boundary-bone", action="store_true", default=True, help="Enable boundary bone stripping (skull/vertebra/ribs/sternum)")
    p.add_argument("--no-strip-boundary-bone", dest="strip_boundary_bone", action="store_false", help="Disable boundary bone stripping")
    p.add_argument("--apply-bone-mask-early", action="store_true", default=False, help="Apply bone mask BEFORE centerline extraction (may exclude vertebral arteries in foramina). Default=False (defer to post-processing)")
    p.add_argument("--boundary-margin-mm", type=float, default=6.0, help="Physical margin from volume boundary to remove high-HU bone shell")
    p.add_argument(
        "--bone-mask",
        type=str,
        default=None,
        help=(
            "Path to external bone mask (NIfTI). When --apply-bone-mask-early is set, "
            "voxels >0 are removed from vessel mask; otherwise used to strip only "
            "boundary-connected bone."
        ),
    )
    p.add_argument("--vessel-mask", type=str, default=None, help="Path to external vessel mask (NIfTI). Voxels >0 are added to vessel mask (use TotalSegmentator vascular labels).")
    p.add_argument("--min-component-size", type=int, default=500, help="Minimum connected component size (voxels)")
    p.add_argument("--step-size", type=float, default=0.5, help="Centerline step size")
    p.add_argument("--min-distance-value", type=float, default=1.5, help="Minimum distance value for path expansion")
    p.add_argument("--max-iterations", type=int, default=8000, help="Max iterations for path extraction")
    p.add_argument("--contact-distance-threshold", type=float, default=1.5, help="Contact distance threshold")
    p.add_argument("--crop-margin-vox", type=int, default=8, help="Crop mask to bbox + margin (voxels)")
    p.add_argument("--crop-margin-vox-x", type=int, default=None, help="Crop margin (voxels) for X/LR axis")
    p.add_argument("--crop-margin-vox-y", type=int, default=None, help="Crop margin (voxels) for Y/AP axis")
    p.add_argument("--crop-margin-vox-z", type=int, default=None, help="Crop margin (voxels) for Z/IS axis")
    p.add_argument("--downsample-factor", type=int, default=1, help="Optional integer downsample factor before distance map (>=1)")
    p.add_argument("--thick-component-max-radius", type=float, default=0.0, help="Remove connected components whose max distance (radius) exceeds this value (mm). Set 0 to disable.")
    p.add_argument("--erosion-iterations", type=int, default=1, help="Erosion iterations in Stage1 cleaning")
    p.add_argument("--dilation-iterations", type=int, default=1, help="Dilation iterations in Stage1 cleaning")
    p.add_argument("--edt-backend", type=str, default="auto", choices=["auto", "metal", "cuda", "cpu"], help="Distance transform backend: auto (Metal on macOS if available, else GPU if cuda requested, else SciPy), metal (distance_transforms on macOS), cuda (distance_transforms/CuPy), cpu (SciPy)")
    p.add_argument("--no-cpu-edt", dest="allow_cpu_edt", action="store_false", help="Disable CPU EDT fallback (requires Metal/CUDA)")
    p.add_argument("--export-centerline-mask", action="store_true", help="Export binary centerline mask (NIfTI) alongside graph")
    p.add_argument("--centerline-step", type=float, default=0.5, help="Interpolation step (voxels) when rasterizing centerlines")
    p.add_argument("--centerline-dilation", type=int, default=0, help="Optional dilation iterations on the rasterized centerline mask")
    p.add_argument(
        "--post-centerline-mask",
        action="append",
        default=[],
        help="Remove centerline points inside these masks after extraction (repeatable, comma-separated).",
    )
    p.add_argument("--convert-evc", action="store_true", help="Convert centerlines to EVC format")
    p.add_argument("--convert-arterial-gnet", action="store_true", help="Convert centerlines to ArterialGNet format")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device: auto (prefer MPS then CUDA), cpu, cuda, mps")
    p.add_argument("--save-intermediates", action="store_true", help="Save intermediate outputs as .nii.gz at each step for debugging")

    # Step 2C: TopCoW intracranial segmentation
    p.add_argument("--run-topcow", action="store_true", help="Run TopCoW (CLAIM) intracranial artery segmentation")
    p.add_argument("--topcow-yolo-model", type=str, default=None, help="Path to yolo-cow-detection.pt")
    p.add_argument("--topcow-nnunet-model-dir", type=str, default=None, help="Path to topcow-claim-models folder")
    p.add_argument("--topcow-output", type=str, default=None, help="Output dir for TopCoW segmentations")
    p.add_argument("--topcow-device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="TopCoW inference device")
    p.add_argument("--topcow-work-dir", type=str, default=None, help="Temporary work dir for TopCoW inference")
    p.add_argument("--topcow-keep-temp", action="store_true", help="Keep TopCoW temporary folders")
    p.add_argument("--topcow-overwrite", action="store_true", help="Overwrite existing TopCoW outputs")
    p.add_argument("--topcow-keep-label-13", action="store_true", help="Keep TopCoW label 13 for 3rd-A2 (default converts 13->15)")
    p.add_argument("--topcow-labels-json", type=str, default=None, help="Write TopCoW label mapping JSON")

    # Step 2D: Multi-label merge (extra + intracranial)
    p.add_argument("--build-multilabel", action="store_true", help="Build combined multi-label vessel map")
    p.add_argument("--multilabel-output", type=str, default=None, help="Output NIfTI for multi-label map")
    p.add_argument("--multilabel-label-config", type=str, default=None, help="JSON config overriding default label list")
    p.add_argument("--multilabel-labels-json", type=str, default=None, help="Write label mapping JSON")
    p.add_argument("--multilabel-overwrite", action="store_true", help="Allow label overwrites when merging masks")
    p.add_argument("--multilabel-topcow", type=str, default=None, help="TopCoW label map to merge (if not running TopCoW)")
    p.add_argument("--multilabel-topcow-mode", choices=["auto", "binary", "label"], default="auto", help="TopCoW input type")
    p.add_argument("--multilabel-topcow-label", type=int, default=50, help="Label ID for binary TopCoW mask")
    p.add_argument("--multilabel-topcow-offset", type=int, default=100, help="Offset added to TopCoW labels")
    p.add_argument("--multilabel-topcow-remap", type=str, default=None, help="JSON mapping for TopCoW label remap")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        cta_paths = collect_inputs(input_path, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not cta_paths:
        print(f"ERROR: No .nii/.nii.gz files found in {input_path}")
        return 2

    batch_mode = len(cta_paths) > 1

    print("=" * 70)
    print("CENTERLINE EXTRACTION: CTA RUNNER")
    print("=" * 70)
    if batch_mode:
        print(f"Batch mode: {len(cta_paths)} files")

    failures = []
    for idx, cta_path in enumerate(cta_paths, start=1):
        case_name = strip_nii_suffix(cta_path)
        case_output = output_root / case_name if batch_mode else output_root
        print("\n" + "-" * 70)
        print(f"[{idx}/{len(cta_paths)}] Processing {cta_path}")
        print("-" * 70)
        try:
            run_single_cta(cta_path, case_output, args)
        except Exception as exc:  # noqa: BLE001
            failures.append((cta_path, exc))
            print(f"ERROR processing {cta_path}: {exc}")

    if failures:
        print("\nCompleted with errors:")
        for cta_path, exc in failures:
            print(f"  {cta_path}: {exc}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
