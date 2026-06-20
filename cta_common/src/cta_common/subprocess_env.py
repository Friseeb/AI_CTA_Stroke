"""Environment helpers for spawning an external Python as a subprocess.

Slicer (and host virtualenvs) export ``PYTHONHOME``/``PYTHONPATH``/``DYLD_*`` etc.
that corrupt the imports of an external conda/venv interpreter (e.g. one with
torch/monai/nnU-Net). Strip those, put the target interpreter's ``bin`` dir first
on ``PATH``, and apply any extra overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

# Host-Python variables that break an externally launched interpreter.
DEFAULT_STRIP_VARS: tuple[str, ...] = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "LD_PRELOAD",
    "QT_PLUGIN_PATH",
    "SSL_CERT_FILE",
)


def make_subprocess_env(
    python_exe: str | Path,
    *,
    strip_vars: Sequence[str] = DEFAULT_STRIP_VARS,
    base_env: Mapping[str, str] | None = None,
    **extra: object,
) -> dict[str, str]:
    """Build a clean environment dict for running ``python_exe`` as a subprocess.

    - removes ``strip_vars`` from a copy of ``base_env`` (defaults to ``os.environ``)
    - prepends the interpreter's ``bin`` directory to ``PATH``
    - applies ``extra`` key/values last (stringified), e.g. CUDA_VISIBLE_DEVICES=""
    """
    src = os.environ if base_env is None else base_env
    strip = set(strip_vars)
    env = {k: v for k, v in src.items() if k not in strip}
    bin_dir = str(Path(python_exe).resolve().parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    for key, value in extra.items():
        env[key] = str(value)
    return env


__all__ = ["DEFAULT_STRIP_VARS", "make_subprocess_env"]
