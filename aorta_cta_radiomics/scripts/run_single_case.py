"""Run one case without installing the console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aorta_cta_radiomics.cli import run_pipeline_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one aorta CTA radiomics case.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--aorta-mask", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_pipeline_case(args.image, args.aorta_mask, args.case_id, args.outdir, args.config)


if __name__ == "__main__":
    main()
