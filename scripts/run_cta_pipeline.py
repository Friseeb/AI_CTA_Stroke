#!/usr/bin/env python3
"""
Generic CTA centerline extraction runner.

Usage:
  python -u scripts/run_cta_pipeline.py --input /path/to/cta.nii.gz --output outputs/run1

Notes:
- No subject-specific naming; caller controls paths.
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
    output_path: Path | None = None,
):
    print(f"Loading CTA: {cta_path}")
    cta = nib.load(str(cta_path))
    data = cta.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {cta.header.get_zooms()}")
    print(f"  Intensity range: {data.min():.1f} to {data.max():.1f} HU")

    upper_desc = f" and < {max_hu}" if max_hu is not None else ""
    print(f"\nCreating vessel mask (HU > {threshold_hu}{upper_desc}, removing boundary bone >= {bone_hu})...")
    bandpass = data > threshold_hu
    if max_hu is not None:
        bandpass &= data < max_hu

    # Optional explicit bone mask removal (e.g., from external/monai/TotalSegmentator)
    if bone_mask_path:
        bone_mask_img = nib.load(str(bone_mask_path))
        bone_mask_data = bone_mask_img.get_fdata() > 0
        if bone_mask_data.shape != data.shape:
            raise ValueError("Bone mask shape does not match CTA shape")
        bandpass &= ~bone_mask_data

    # Optional explicit vessel mask union (e.g., from TotalSegmentator vascular labels)
    if vessel_mask_path:
        vessel_mask_img = nib.load(str(vessel_mask_path))
        vessel_mask_data = vessel_mask_img.get_fdata() > 0
        if vessel_mask_data.shape != data.shape:
            raise ValueError("Vessel mask shape does not match CTA shape")
        bandpass = bandpass | vessel_mask_data

    if strip_boundary_bone:
        # Hard bone exclusion only for boundary-connected bone so we keep internal arterial calcifications
        bone_mask = (data >= bone_hu).astype(np.uint8)
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


def parse_args():
    p = argparse.ArgumentParser(description="CTA Centerline Extraction Runner")
    p.add_argument("--input", required=True, help="Path to CTA NIfTI (.nii/.nii.gz)")
    p.add_argument("--output", required=True, help="Output directory for artifacts")
    p.add_argument("--threshold", type=int, default=150, help="Lower HU threshold for vessel mask")
    p.add_argument("--max-hu", type=int, default=700, help="Upper HU cutoff to suppress dense bone (None to disable)")
    p.add_argument("--bone-hu", type=int, default=900, help="Bone threshold; boundary-connected components >= this HU are excluded (arterial calcifications are preserved)")
    p.add_argument("--strip-boundary-bone", action="store_true", default=True, help="Enable boundary bone stripping (skull/vertebra/ribs/sternum)")
    p.add_argument("--no-strip-boundary-bone", dest="strip_boundary_bone", action="store_false", help="Disable boundary bone stripping")
    p.add_argument("--boundary-margin-mm", type=float, default=6.0, help="Physical margin from volume boundary to remove high-HU bone shell")
    p.add_argument("--bone-mask", type=str, default=None, help="Path to external bone mask (NIfTI). Voxels >0 are removed from vessel mask (use your MONAI/OAL segmenter output).")
    p.add_argument("--vessel-mask", type=str, default=None, help="Path to external vessel mask (NIfTI). Voxels >0 are added to vessel mask (use TotalSegmentator vascular labels).")
    p.add_argument("--min-component-size", type=int, default=500, help="Minimum connected component size (voxels)")
    p.add_argument("--step-size", type=float, default=0.5, help="Centerline step size")
    p.add_argument("--min-distance-value", type=float, default=1.5, help="Minimum distance value for path expansion")
    p.add_argument("--max-iterations", type=int, default=8000, help="Max iterations for path extraction")
    p.add_argument("--contact-distance-threshold", type=float, default=1.5, help="Contact distance threshold")
    p.add_argument("--convert-evc", action="store_true", help="Convert centerlines to EVC format")
    p.add_argument("--convert-arterial-gnet", action="store_true", help="Convert centerlines to ArterialGNet format")
    return p.parse_args()


def main():
    args = parse_args()

    cta_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cta_path.exists():
        print(f"ERROR: CTA file not found: {cta_path}")
        return 2

    print("=" * 70)
    print("CENTERLINE EXTRACTION: CTA RUNNER")
    print("=" * 70)

    mask_path = output_dir / 'vessel_mask.nii.gz'
    _mask, _affine, _hdr = create_vessel_mask_from_cta(
        cta_path=cta_path,
        threshold_hu=args.threshold,
        max_hu=args.max_hu,
        bone_hu=args.bone_hu,
        strip_boundary_bone=args.strip_boundary_bone,
        boundary_margin_mm=args.boundary_margin_mm,
        bone_mask_path=Path(args.bone_mask) if args.bone_mask else None,
        vessel_mask_path=Path(args.vessel_mask) if args.vessel_mask else None,
        min_component_size=args.min_component_size,
        output_path=mask_path,
    )

    print("\nRunning centerline extraction pipeline...")
    pipeline = CenterlineExtractionPipeline(
        nifti_path=str(mask_path),
        output_dir=str(output_dir / 'centerline'),
        log_level='WARNING',
    )

    results = pipeline.run(
        min_component_size=300,
        min_distance_value=args.min_distance_value,
        step_size=args.step_size,
        max_iterations=args.max_iterations,
        contact_distance_threshold=args.contact_distance_threshold,
    )

    print("\nPipeline Summary:")
    summary = pipeline.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nExtracting morphology features...")
    features = extract_morphology_features(results['stage7']['graph'])
    features_path = output_dir / 'morphology_features.json'
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"✓ Saved: {features_path}")

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


if __name__ == '__main__':
    sys.exit(main())