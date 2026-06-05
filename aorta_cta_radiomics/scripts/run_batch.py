"""Run a manifest without installing the console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aorta_cta_radiomics.cli import _write_aggregated, run_pipeline_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aorta CTA radiomics cases from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    results = [
        run_pipeline_case(row.image_path, row.aorta_mask_path, row.case_id, args.outdir, args.config)
        for row in manifest.itertuples(index=False)
    ]
    _write_aggregated(results, Path(args.outdir) / "qc", Path(args.outdir) / "features")


if __name__ == "__main__":
    main()
