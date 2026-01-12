"""Reference segmentation method wrappers.

These functions implement the two literature methods documented in
`docs/literature/centerline_extraction_methods.md`:

1. VMTK-based extraction (recommended): Uses modular pipeline implementing
   Antiga et al. 2008 Eikonal-equation-based shortest path tracing.

2. Skeleton-based extraction: Simpler distance transform approach, useful for
   quick prototyping and testing.

The VMTK implementation is modularized in `centerline_antiga_2008/` with
separate stages for surface extraction, extremal point detection, path tracing,
radius computation, bifurcation detection, and graph construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from .centerline_antiga_2008 import CenterlineExtractionPipeline
from .qc_metrics import simple_centerline
from .segmentation_runner import SegmentationOutputs


def vmtk_eikonal_centerline(cta_img: nib.Nifti1Image) -> SegmentationOutputs:
    """VMTK-based centerline extraction via Eikonal equation (Antiga et al., 2008).

    Uses modular pipeline in `centerline_antiga_2008/`:
    1. Surface extraction (morphological cleaning)
    2. Extremal point detection (distance map maxima)
    3. Voronoi skeleton (implicit, via distance map)
    4. Eikonal path tracing (gradient descent on distance transform)
    5. Radius computation (trilinear interpolation)
    6. Bifurcation detection (tube containment)
    7. Graph construction (NetworkX export)

    Parameters
    ----------
    cta_img : nib.Nifti1Image
        CTA image volume

    Returns
    -------
    SegmentationOutputs
        mask (segmentation) and centerline (as binary skeleton)

    References
    ----------
    Antiga, L., Piccinelli, M., Botti, L., Ene-Iordache, B., Remuzzi, A., &
    Steinman, D. A. (2008). An image-based modeling framework for
    patient-specific computational hemodynamics. Medical & Biological
    Engineering & Computing, 43(3), 252-261.
    """
    # Binary segmentation
    data = cta_img.get_fdata()
    thresh = np.percentile(data, 98.5)
    binary_mask = (data >= thresh).astype(np.uint8)

    # Save binary mask temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        mask_img = nib.Nifti1Image(binary_mask, cta_img.affine, cta_img.header)
        nib.save(mask_img, tmp_path)

        # Run Antiga 2008 modular pipeline
        pipeline = CenterlineExtractionPipeline(
            nifti_path=tmp_path,
            log_level='WARNING',
        )
        results = pipeline.run(
            min_component_size=50,
            step_size=0.1,
            max_iterations=5000,
            contact_distance_threshold=1.0,
        )

        # Extract centerlines from Stage 4
        centerlines = results['stage4']['centerlines']
        centerline_mask = np.zeros_like(binary_mask)
        for centerline in centerlines.values():
            path = np.array(centerline['path']).astype(int)
            path = np.clip(path, 0, np.array(binary_mask.shape) - 1)
            centerline_mask[tuple(path.T)] = 1

        # Cleanup
        tmp_path.unlink()

    centerline_img = nib.Nifti1Image(centerline_mask, cta_img.affine, cta_img.header)

    metadata: Dict[str, str] = {
        'method': 'vmtk_antiga_2008_modular',
        'threshold_percentile': '98.5',
        'stages_completed': '7 (surface, extremal, voronoi, eikonal, radius, bifurcation, graph)',
        'reference': 'Antiga et al. 2008',
    }

    return SegmentationOutputs(mask=mask_img, centerline=centerline_img, metadata=metadata)


def skeleton_based_method(cta_img: nib.Nifti1Image) -> SegmentationOutputs:
    """Skeleton-based extraction (simpler distance transform approach).

    Uses morphological skeletonization on the vessel mask. Faster and
    lighter-weight than VMTK but less robust for complex vascular trees.

    Output: Binary skeleton approximating centerlines."""
    data = cta_img.get_fdata()
    smoothed = gaussian_filter(data, sigma=1.0)
    thresh = np.percentile(smoothed, 98)
    mask = (smoothed >= thresh).astype(np.uint8)

    mask_img = nib.Nifti1Image(mask, cta_img.affine, cta_img.header)
    centerline_img = simple_centerline(mask_img)
    metadata: Dict[str, str] = {"threshold": f"p98={thresh:.2f}"}
    return SegmentationOutputs(mask=mask_img, centerline=centerline_img, metadata=metadata)


def synthetic_cylinder(shape=(160, 160, 160), radius=8, length=140) -> nib.Nifti1Image:
    """Generate a simple cylindrical phantom for quick tests."""
    x0, y0 = np.array(shape[:2]) // 2
    mask = np.zeros(shape, dtype=np.uint8)
    z_start = max(0, (shape[2] - length) // 2)
    z_end = min(shape[2], z_start + length)
    for z in range(z_start, z_end):
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        circle = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
        mask[circle, z] = 1
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    return nib.Nifti1Image(mask, affine)
