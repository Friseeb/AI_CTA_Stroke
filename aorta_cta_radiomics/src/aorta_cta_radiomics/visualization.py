"""Non-interactive QC figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_overlay_quicklook(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    title: str,
    overlay: np.ndarray | None = None,
) -> Path:
    """Save a central-slice PNG with mask and optional overlay."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for quicklook figure generation.") from exc

    binary = np.asarray(mask, dtype=bool)
    z_indices = np.where(binary.any(axis=(1, 2)))[0]
    z = int(z_indices[len(z_indices) // 2]) if z_indices.size else image.shape[0] // 2
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(image[z], cmap="gray", vmin=-100, vmax=700)
    plt.contour(binary[z], levels=[0.5], colors="lime", linewidths=0.8)
    if overlay is not None and overlay.any():
        masked_overlay = np.ma.masked_where(~overlay[z].astype(bool), overlay[z])
        plt.imshow(masked_overlay, cmap="autumn", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out
