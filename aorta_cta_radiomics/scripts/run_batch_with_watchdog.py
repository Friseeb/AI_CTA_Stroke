#!/usr/bin/env python
"""Run the staged aorta batch with neuro-CTA metadata filtering and ntfy watchdog."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aorta_cta_radiomics.batch_watchdog import main


if __name__ == "__main__":
    main()
