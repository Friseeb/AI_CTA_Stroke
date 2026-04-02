#!/usr/bin/env python
"""
run_plots_batch.py — Run all 7 raincloud plot scripts in sequence.
Usage:
    python scripts/run_plots_batch.py --csv <path> --outdir <path>
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

PLOTS = [
    ('plot_01_roi_voxels.py',          'ROI voxel counts'),
    ('plot_02_before_after.py',         'Before/after resampling'),
    ('plot_03_interpolated_volumes.py', 'Interpolated volumes'),
    ('plot_04_hu_by_structure.py',      'HU by structure'),
    ('plot_05_hu_ratio.py',             'LA/Aorta HU ratio'),
    ('plot_06_laa_la_ratio.py',         'LAA/LA HU ratio'),
    ('plot_07_laa_aorta_ratio.py',      'LAA/Aorta HU ratio'),
]


def main():
    parser = argparse.ArgumentParser(description='Run all raincloud plot scripts')
    parser.add_argument('--csv',    required=True, help='Path to radiomics CSV')
    parser.add_argument('--outdir', required=True, help='Output directory for plots')
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    errors = []

    for script, label in tqdm(PLOTS, desc='Generating plots', unit='plot', dynamic_ncols=True):
        tqdm.write(f'  → {label}')
        result = subprocess.run(
            [sys.executable, str(scripts_dir / script),
             '--csv', args.csv,
             '--outdir', args.outdir],
            capture_output=False,
        )
        if result.returncode != 0:
            tqdm.write(f'  ERROR in {script} (exit code {result.returncode})')
            errors.append(script)

    print()
    if errors:
        print(f'Finished with errors in: {", ".join(errors)}')
        sys.exit(1)
    else:
        print(f'All {len(PLOTS)} plots saved to: {args.outdir}')


if __name__ == '__main__':
    main()
