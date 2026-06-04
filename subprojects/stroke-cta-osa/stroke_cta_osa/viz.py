"""Lightweight QC review images. Matplotlib is treated as optional: if it is
not installed we skip the call silently rather than crashing the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .logging_utils import get_logger
from .types import AirwayMaskInfo, CTAImage

log = get_logger("viz")


def _try_mpl():
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def save_qc_images(
    image: CTAImage,
    airway: Optional[AirwayMaskInfo],
    fat_total_mask: Optional[np.ndarray],
    fat_parapharyngeal_mask: Optional[np.ndarray],
    fat_retropharyngeal_mask: Optional[np.ndarray],
    out_dir: Path,
    min_csa_z_index: Optional[int],
    window_center: float = 100.0,
    window_width: float = 700.0,
) -> list[Path]:
    plt = _try_mpl()
    if plt is None:
        log.info("matplotlib not available; skipping QC images.")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _axial(z: int, mask: Optional[np.ndarray], name: str, color: str) -> Optional[Path]:
        if z is None or z < 0 or z >= image.array.shape[0]:
            return None
        sl = image.array[z]
        fig, ax = plt.subplots(figsize=(4, 4))
        vmin = window_center - window_width / 2
        vmax = window_center + window_width / 2
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        if mask is not None and 0 <= z < mask.shape[0] and mask[z].any():
            ax.contour(mask[z], levels=[0.5], colors=color, linewidths=0.7)
        ax.set_axis_off()
        ax.set_title(f"{name} z={z}", fontsize=9)
        p = out_dir / f"qc_{name}_z{z:04d}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return p

    if airway is not None and airway.is_present and min_csa_z_index is not None:
        p = _axial(min_csa_z_index, airway.mask_zyx, "airway_min_csa", "cyan")
        if p: paths.append(p)
    if fat_parapharyngeal_mask is not None and min_csa_z_index is not None:
        p = _axial(min_csa_z_index, fat_parapharyngeal_mask, "fat_parapharyngeal", "yellow")
        if p: paths.append(p)
    if fat_retropharyngeal_mask is not None and min_csa_z_index is not None:
        p = _axial(min_csa_z_index, fat_retropharyngeal_mask, "fat_retropharyngeal", "magenta")
        if p: paths.append(p)
    if fat_total_mask is not None:
        # Middle of fat extent
        zs = np.where(fat_total_mask.any(axis=(1, 2)))[0]
        if zs.size:
            z = int(zs[len(zs) // 2])
            p = _axial(z, fat_total_mask, "fat_cervical_total", "orange")
            if p: paths.append(p)
    return paths
