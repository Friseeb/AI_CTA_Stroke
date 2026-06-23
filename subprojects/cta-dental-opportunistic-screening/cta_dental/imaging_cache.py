"""Process-local cache for label-array reads, shared within one case.

``features`` and ``qc`` both load the same per-tooth/jaw label NIfTIs during a
single ``run``. Routing those reads through one memoized reader avoids loading
each file from disk multiple times across modules. Callers must copy (e.g.
``.astype(...)``) before mutating, so the cached array is never modified in place.

Call :func:`clear_cache` at the start of processing a case to keep the cache
per-case (no unbounded growth or cross-case staleness in a batch).
"""

from __future__ import annotations

import functools

import numpy as np
import SimpleITK as sitk


@functools.lru_cache(maxsize=None)
def read_label_array(path_str: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path_str))


def label_array(path) -> np.ndarray:
    """Cached read of a label NIfTI as a numpy array."""
    return read_label_array(str(path))


def clear_cache() -> None:
    read_label_array.cache_clear()
