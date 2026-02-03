"""Example script: end-to-end centerline extraction using Antiga 2008 pipeline."""
import logging
from pathlib import Path

from AI_CTA_Stroke.python.analysis.centerline_antiga_2008 import (
    CenterlineExtractionPipeline,
)


def main():
    """Run centerline extraction pipeline on example vessel mask."""
    # Setup
    input_nifti = Path('/path/to/vessel_mask.nii.gz')  # Update path
    output_dir = Path('/path/to/output')  # Update path

    # Initialize pipeline
    pipeline = CenterlineExtractionPipeline(
        nifti_path=input_nifti,
        output_dir=output_dir,
        log_level='INFO',
    )

    # Run all stages (1-7)
    results = pipeline.run(
        min_component_size=50,  # Stage 1: min voxels per component
        erosion_iterations=1,  # Stage 1: morphological cleaning
        dilation_iterations=1,
        step_size=0.1,  # Stage 4: gradient descent step
        max_iterations=5000,  # Stage 4: max gradient descent steps
        contact_distance_threshold=1.0,  # Stage 6: bifurcation detection threshold
        save_label_map=True,  # Optional: save labeled segmentation map
    )

    # Print summary
    summary = pipeline.summary()
    print('\n=== Centerline Extraction Summary ===')
    for key, value in summary.items():
        print(f'{key}: {value}')

    # Access results
    print(f"\nCenterline graph exported to: {results['export']['pickle_path']}")
    print(f"Node data: {results['export']['json_nodes_path']}")
    print(f"Edge data: {results['export']['json_edges_path']}")

    return pipeline


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    pipeline = main()
