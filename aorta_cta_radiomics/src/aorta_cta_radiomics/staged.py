"""Console entry point for the staged manifest runner."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Run ``scripts/run_manifest_staged.py`` from an editable checkout."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_manifest_staged.py"
    if not script.exists():
        raise FileNotFoundError(
            "The staged runner script is not available. Use an editable/source checkout "
            "or run scripts/run_manifest_staged.py directly."
        )
    runpy.run_path(str(script), run_name="__main__")

