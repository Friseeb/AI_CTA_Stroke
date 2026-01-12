"""Pipeline orchestrator: chains all stages of Antiga 2008 centerline extraction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .stage1_surface_extraction import extract_surface
from .stage2_extremal_points import detect_extremal_points
from .stage4_eikonal import extract_centerlines_via_eikonal
from .stage5_radius import compute_radii
from .stage6_bifurcations import detect_bifurcations
from .stage7_graph import build_centerline_graph, export_graph


class CenterlineExtractionPipeline:
    """Orchestrates multi-stage Antiga 2008 centerline extraction.

    Stages:
        1. Surface extraction (morphological cleaning)
        2. Extremal points detection (distance map maxima)
        3. Voronoi skeleton (implicit, distance map)
        4. Eikonal path tracing (gradient descent on distance map)
        5. Radius computation (trilinear interpolation)
        6. Bifurcation detection (tube containment)
        7. Graph construction (NetworkX export)
    """

    def __init__(
        self,
        nifti_path: str | Path,
        output_dir: str | Path | None = None,
        log_level: str = 'INFO',
    ):
        """Initialize pipeline.

        Parameters
        ----------
        nifti_path : str | Path
            Path to binary vessel mask (NIfTI)
        output_dir : str | Path, optional
            Directory for outputs; defaults to nifti_path parent / 'centerline_output'
        log_level : str
            Logging level
        """
        self.nifti_path = Path(nifti_path)

        if output_dir is None:
            output_dir = self.nifti_path.parent / 'centerline_output'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(self.output_dir / 'pipeline.log')
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        self.logger.info(f'Initialized pipeline for {self.nifti_path}')
        self.results = {}

    def load_image(self) -> np.ndarray:
        """Load NIfTI image and extract binary mask.

        Returns
        -------
        np.ndarray
            3D binary array of vessel segmentation
        """
        self.logger.info(f'Loading image from {self.nifti_path}')
        img = nib.load(self.nifti_path)
        mask = np.asarray(img.dataobj) > 0
        self.logger.info(f'Image shape: {mask.shape}, voxel count: {mask.sum()}')
        return mask

    def run(self, **stage_kwargs) -> dict:
        """Execute full pipeline.

        Parameters
        ----------
        **stage_kwargs : dict
            Keyword arguments for individual stages
            (e.g., min_component_size=50, contact_distance_threshold=1.0)

        Returns
        -------
        dict
            Pipeline results with keys: stage1, stage2, stage4, stage5, stage6, stage7
        """
        try:
            # Load image
            mask = self.load_image()
            self.results['mask'] = mask

            # Stage 1: Surface extraction
            self.logger.info('Running Stage 1: Surface extraction')
            stage1_result = extract_surface(
                mask,
                min_component_size=stage_kwargs.get('min_component_size', 50),
                erosion_iterations=stage_kwargs.get('erosion_iterations', 1),
                dilation_iterations=stage_kwargs.get('dilation_iterations', 1),
            )
            self.results['stage1'] = stage1_result
            self.logger.info(
                f'Stage 1 complete: {stage1_result["num_components"]} components'
            )

            # Stage 2: Extremal points
            self.logger.info('Running Stage 2: Extremal points detection')
            stage2_result = detect_extremal_points(
                stage1_result['cleaned_mask'],
                distance_map=stage1_result['distance_map'],
            )
            self.results['stage2'] = stage2_result
            self.logger.info(
                f'Stage 2 complete: {len(stage2_result["extremal_points"])} extremal points'
            )

            # Stage 4: Eikonal path tracing
            self.logger.info('Running Stage 4: Eikonal path tracing')
            stage4_result = extract_centerlines_via_eikonal(
                distance_map=stage1_result['distance_map'],
                extremal_points=stage2_result['extremal_points'],
                step_size=stage_kwargs.get('step_size', 0.1),
                max_iterations=stage_kwargs.get('max_iterations', 5000),
            )
            self.results['stage4'] = stage4_result
            self.logger.info(
                f'Stage 4 complete: {len(stage4_result["centerlines"])} centerlines'
            )

            # Stage 5: Radius computation
            self.logger.info('Running Stage 5: Radius computation')
            stage5_result = compute_radii(
                centerlines=stage4_result['centerlines'],
                distance_map=stage1_result['distance_map'],
            )
            self.results['stage5'] = stage5_result
            self.logger.info(f'Stage 5 complete: radii assigned to all centerlines')

            # Stage 6: Bifurcation detection
            self.logger.info('Running Stage 6: Bifurcation detection')
            stage6_result = detect_bifurcations(
                stage5_result['centerlines'],
                contact_distance_threshold=stage_kwargs.get(
                    'contact_distance_threshold', 1.0
                ),
            )
            self.results['stage6'] = stage6_result
            self.logger.info(
                f'Stage 6 complete: {stage6_result["num_bifurcations"]} bifurcations detected'
            )

            # Stage 7: Graph construction
            self.logger.info('Running Stage 7: Graph construction')
            stage7_result = build_centerline_graph(
                stage5_result['centerlines'],
                stage6_result,
            )
            self.results['stage7'] = stage7_result
            self.logger.info(
                f'Stage 7 complete: {len(stage7_result["node_data"])} nodes, '
                f'{len(stage7_result["edge_data"])} edges'
            )

            # Export graph
            self.logger.info('Exporting centerline graph')
            export_result = export_graph(
                stage7_result,
                self.output_dir,
                basename='centerline',
            )
            self.results['export'] = export_result
            self.logger.info(
                f'Graph exported to {export_result["pickle_path"]}'
            )

            self.logger.info('Pipeline completed successfully')
            return self.results

        except Exception as e:
            self.logger.error(f'Pipeline failed: {e}', exc_info=True)
            raise

    def summary(self) -> dict:
        """Generate summary statistics from pipeline results.

        Returns
        -------
        dict
            Summary with num_centerlines, total_path_length, etc.
        """
        summary = {
            'num_components': self.results.get('stage1', {}).get(
                'num_components', 0
            ),
            'num_extremal_points': len(
                self.results.get('stage2', {}).get('extremal_points', [])
            ),
            'num_centerlines': len(
                self.results.get('stage4', {}).get('centerlines', {})
            ),
            'num_bifurcations': self.results.get('stage6', {}).get(
                'num_bifurcations', 0
            ),
            'num_nodes': len(
                self.results.get('stage7', {}).get('node_data', [])
            ),
            'num_edges': len(
                self.results.get('stage7', {}).get('edge_data', [])
            ),
        }

        # Compute total path length
        total_length = 0.0
        for centerline in (
            self.results.get('stage4', {}).get('centerlines', {}).values()
        ):
            path = np.array(centerline['path'])
            if len(path) > 1:
                diffs = np.diff(path, axis=0)
                lengths = np.linalg.norm(diffs, axis=1)
                total_length += float(lengths.sum())
        summary['total_centerline_length_mm'] = total_length

        return summary
