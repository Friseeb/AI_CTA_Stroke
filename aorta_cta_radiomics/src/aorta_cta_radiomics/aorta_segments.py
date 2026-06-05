"""Pragmatic aorta segment placeholders for version 1."""

from __future__ import annotations

import numpy as np
import pandas as pd


SEGMENT_LABELS = {
    0: "background",
    1: "whole_aorta",
}


def whole_aorta_segment_mask(mask: np.ndarray) -> np.ndarray:
    """Return a label map with the full aorta as label 1.

    Anatomical subsegmentation is intentionally conservative in version 1.
    Centerline/landmark-based ascending, arch, descending, and abdominal labels
    should replace this when validated for the acquisition protocol.
    """
    labels = np.zeros_like(mask, dtype=np.uint8)
    labels[np.asarray(mask, dtype=bool)] = 1
    return labels


def segment_summary(
    labels: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
) -> pd.DataFrame:
    """Summarize voxel counts and volumes for segment labels."""
    voxel_volume = float(np.prod(spacing_xyz))
    rows = []
    for label, name in SEGMENT_LABELS.items():
        if label == 0:
            continue
        voxel_count = int((labels == label).sum())
        rows.append(
            {
                "case_id": case_id,
                "segment_label": int(label),
                "segment_name": name,
                "voxel_count": voxel_count,
                "volume_mm3": voxel_count * voxel_volume,
                "segmentation_method": "whole_aorta_placeholder_v1",
            }
        )
    return pd.DataFrame(rows)
