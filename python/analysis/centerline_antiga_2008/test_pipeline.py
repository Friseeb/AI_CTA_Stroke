"""Test suite for Antiga 2008 centerline extraction pipeline."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports like `python.analysis.*`
REPO_ROOT = Path(__file__).parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.analysis.centerline_antiga_2008 import (
    CenterlineExtractionPipeline,
    build_centerline_graph,
    compute_radii,
    detect_bifurcations,
    detect_extremal_points,
    extract_centerlines_via_eikonal,
    extract_surface,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_vessel_mask(shape=(100, 100, 100), radius=5) -> np.ndarray:
    """Create a synthetic Y-shaped vessel for testing.

    Two branches merging into one main vessel.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, z0 = np.array(shape) // 2

    # Main branch (vertical)
    for z in range(10, 80):
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        circle = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
        mask[circle, z] = 1

    # Left branch (diagonal)
    for z in range(30, 70):
        offset = int((z - 30) * 0.3)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        circle = (xx - (x0 - offset)) ** 2 + (yy - (y0 - offset)) ** 2 <= (radius - 1) ** 2
        mask[circle, z] = 1

    # Right branch (diagonal)
    for z in range(30, 70):
        offset = int((z - 30) * 0.3)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        circle = (xx - (x0 + offset)) ** 2 + (yy - (y0 - offset)) ** 2 <= (radius - 1) ** 2
        mask[circle, z] = 1

    return mask


def test_stage1_surface_extraction():
    """Test Stage 1: surface extraction."""
    logger.info('Testing Stage 1: Surface Extraction')
    mask = create_synthetic_vessel_mask()

    result = extract_surface(mask, min_component_size=50)

    assert 'cleaned_mask' in result
    assert 'distance_map' in result
    assert 'num_components' in result

    assert result['cleaned_mask'].shape == mask.shape
    assert result['distance_map'].shape == mask.shape
    assert result['num_components'] >= 1

    logger.info(f"✓ Stage 1 passed. Components: {result['num_components']}")
    return result


def test_stage2_extremal_points(stage1_result):
    """Test Stage 2: extremal points detection."""
    logger.info('Testing Stage 2: Extremal Points Detection')

    result = detect_extremal_points(
        stage1_result['cleaned_mask'],
        distance_map=stage1_result['distance_map'],
    )

    assert 'extremal_points' in result
    assert len(result['extremal_points']) >= 1

    logger.info(f"✓ Stage 2 passed. Extremal points: {len(result['extremal_points'])}")
    return result


def test_stage4_eikonal(stage1_result, stage2_result):
    """Test Stage 4: Eikonal path tracing."""
    logger.info('Testing Stage 4: Eikonal Path Tracing')

    result = extract_centerlines_via_eikonal(
        distance_map=stage1_result['distance_map'],
        extremal_points=stage2_result['extremal_points'],
        step_size=0.2,
        max_iterations=1000,
    )

    assert 'centerlines' in result
    assert len(result['centerlines']) >= 1

    for seg_id, centerline in result['centerlines'].items():
        assert 'path' in centerline
        assert len(centerline['path']) >= 2
        logger.info(f"  Centerline {seg_id}: {len(centerline['path'])} points")

    logger.info(f"✓ Stage 4 passed. Centerlines: {len(result['centerlines'])}")
    return result


def test_stage5_radius(stage1_result, stage4_result):
    """Test Stage 5: radius computation."""
    logger.info('Testing Stage 5: Radius Computation')

    result = compute_radii(
        centerlines=stage4_result['centerlines'],
        distance_map=stage1_result['distance_map'],
    )

    assert 'centerlines' in result

    for seg_id, centerline in result['centerlines'].items():
        assert 'radii' in centerline
        assert len(centerline['radii']) == len(centerline['path'])
        logger.info(
            f"  Centerline {seg_id}: "
            f"mean_radius={np.mean(centerline['radii']):.2f}, "
            f"max_radius={np.max(centerline['radii']):.2f}"
        )

    logger.info(f"✓ Stage 5 passed. Radii assigned.")
    return result


def test_stage6_bifurcations(stage5_result):
    """Test Stage 6: bifurcation detection."""
    logger.info('Testing Stage 6: Bifurcation Detection')

    result = detect_bifurcations(stage5_result['centerlines'])

    assert 'bifurcations' in result
    assert 'num_bifurcations' in result

    logger.info(f"✓ Stage 6 passed. Bifurcations: {result['num_bifurcations']}")
    return result


def test_stage7_graph(stage5_result, stage6_result):
    """Test Stage 7: graph construction."""
    logger.info('Testing Stage 7: Graph Construction')

    result = build_centerline_graph(
        stage5_result['centerlines'],
        stage6_result,
    )

    assert 'graph' in result
    assert 'node_data' in result
    assert 'edge_data' in result

    graph = result['graph']
    assert len(graph.nodes) >= 1
    assert len(graph.edges) >= 1

    logger.info(
        f"✓ Stage 7 passed. Graph: "
        f"{len(graph.nodes)} nodes, {len(graph.edges)} edges"
    )
    return result


def test_full_pipeline():
    """Test full pipeline orchestration."""
    logger.info('Testing Full Pipeline Orchestration')

    # Create synthetic vessel
    mask = create_synthetic_vessel_mask()

    # Save to temporary NIfTI file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vessel_path = tmpdir / 'vessel.nii.gz'
        affine = np.diag([0.5, 0.5, 0.5, 1.0])
        img = nib.Nifti1Image(mask.astype(np.float32), affine)
        nib.save(img, vessel_path)

        # Run pipeline
        pipeline = CenterlineExtractionPipeline(
            nifti_path=vessel_path,
            output_dir=tmpdir / 'output',
            log_level='INFO',
        )

        results = pipeline.run(
            min_component_size=50,
            step_size=0.2,
            max_iterations=1000,
        )

        # Check results
        assert 'stage1' in results
        assert 'stage2' in results
        assert 'stage4' in results
        assert 'stage5' in results
        assert 'stage6' in results
        assert 'stage7' in results
        assert 'export' in results

        # Print summary
        summary = pipeline.summary()
        logger.info(f'Pipeline Summary: {summary}')

        # Verify files were exported
        export_result = results['export']
        assert Path(export_result['pickle_path']).exists()
        assert Path(export_result['json_nodes_path']).exists()
        assert Path(export_result['json_edges_path']).exists()

    logger.info('✓ Full pipeline test passed.')


def run_all_tests():
    """Run all tests."""
    logger.info('=' * 60)
    logger.info('ANTIGA 2008 CENTERLINE EXTRACTION TEST SUITE')
    logger.info('=' * 60)

    try:
        # Individual stages
        stage1_result = test_stage1_surface_extraction()
        stage2_result = test_stage2_extremal_points(stage1_result)
        stage4_result = test_stage4_eikonal(stage1_result, stage2_result)
        stage5_result = test_stage5_radius(stage1_result, stage4_result)
        stage6_result = test_stage6_bifurcations(stage5_result)
        test_stage7_graph(stage5_result, stage6_result)

        # Full pipeline
        test_full_pipeline()

        logger.info('=' * 60)
        logger.info('ALL TESTS PASSED ✓')
        logger.info('=' * 60)

    except AssertionError as e:
        logger.error(f'Test failed: {e}', exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    run_all_tests()
