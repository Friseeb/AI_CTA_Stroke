"""Centralised logging setup. Mirrors the dental subproject style so the
two pipelines feel uniform when run side by side. No PHI is logged: callers
must pass already-scrubbed identifiers (study_id, scan_id) — never DICOM
patient name / DOB / MRN.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"stroke_cta_osa.{name}")


def configure_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
