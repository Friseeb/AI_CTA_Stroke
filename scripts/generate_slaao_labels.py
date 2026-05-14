#!/usr/bin/env python3
"""Derive SLAAO multi-label JSON from a filling-defect map and optional LAA mask.

Each SLAAO feature is an independent yes/no binary label derived from
quantitative thresholds on the filling-defect map:

  dark_thrombus_component   — thrombus-like voxels exceed fraction threshold
  contrast_stagnation       — stagnation voxels exceed fraction threshold
  rim_pattern               — thrombus/stagnation voxels cluster near the mask boundary
  whole_LAA_involvement     — dark/stagnation region covers most of the LAA cavity
  regional_pooling          — stagnation concentrated in distal third of LAA
  distal_tip_involvement    — thrombus or stagnation in the distal 20% of LAA
  mixed_pattern             — mixed-label voxels exceed fraction threshold
  uncertain_artifact        — prior disagreement map indicates high uncertainty

All thresholds are configurable. The JSON output is compatible with the
SLAAO structured metadata format described in docs/protocols/laa_slaao_framework.md.

Example:
  python scripts/generate_slaao_labels.py \\
    --filling-defect-map <BIDS_ROOT>/derivatives/laa_slaao/<CASE_ID>/<CASE_ID>_filling_defect_map.nii.gz \\
    --laa-mask <BIDS_ROOT>/derivatives/laa_slaao/<CASE_ID>/<CASE_ID>_laa_corrected.nii.gz \\
    --output-dir <BIDS_ROOT>/derivatives/laa_slaao/<CASE_ID> \\
    --case-id <CASE_ID>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


# Must match label values written by generate_filling_defect_map.py
_LABEL_NORMAL = 1
_LABEL_STAGNATION = 2
_LABEL_THROMBUS = 3
_LABEL_MIXED = 4


@dataclass
class SLAAOLabels:
    dark_thrombus_component: bool = False
    contrast_stagnation: bool = False
    rim_pattern: bool = False
    whole_LAA_involvement: bool = False
    regional_pooling: bool = False
    distal_tip_involvement: bool = False
    mixed_pattern: bool = False
    uncertain_artifact: bool = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Derive SLAAO multi-label JSON from a filling-defect map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--filling-defect-map", required=True, help="Filling-defect map NIfTI (.nii.gz)")
    p.add_argument("--laa-mask", required=True, help="LAA cavity mask NIfTI (.nii.gz)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--case-id", required=True, help="Case identifier")
    p.add_argument(
        "--disagreement-map",
        default=None,
        help="Prior disagreement map NIfTI (from fuse_laa_priors.py). Used for uncertain_artifact.",
    )

    p.add_argument(
        "--thrombus-frac",
        type=float,
        default=0.05,
        help="Minimum fraction of LAA voxels with thrombus label to set dark_thrombus_component",
    )
    p.add_argument(
        "--stagnation-frac",
        type=float,
        default=0.10,
        help="Minimum fraction of LAA voxels with stagnation label to set contrast_stagnation",
    )
    p.add_argument(
        "--mixed-frac",
        type=float,
        default=0.05,
        help="Minimum fraction of LAA voxels with mixed label to set mixed_pattern",
    )
    p.add_argument(
        "--whole-laa-frac",
        type=float,
        default=0.60,
        help="Minimum fraction of LAA voxels that are dark/stagnation to set whole_LAA_involvement",
    )
    p.add_argument(
        "--rim-shell-voxels",
        type=int,
        default=3,
        help="Shell thickness (voxels) from mask boundary used to detect rim_pattern",
    )
    p.add_argument(
        "--rim-frac",
        type=float,
        default=0.30,
        help="Fraction of boundary-shell voxels that must be thrombus/stagnation to set rim_pattern",
    )
    p.add_argument(
        "--distal-frac",
        type=float,
        default=0.20,
        help="Fraction of LAA extent (along principal axis) considered distal for distal_tip_involvement",
    )
    p.add_argument(
        "--pooling-distal-frac",
        type=float,
        default=0.33,
        help="Fraction of LAA extent considered distal for regional_pooling",
    )
    p.add_argument(
        "--pooling-stag-frac",
        type=float,
        default=0.40,
        help="Fraction of distal-region voxels that must be stagnation to set regional_pooling",
    )
    p.add_argument(
        "--uncertain-disagree-frac",
        type=float,
        default=0.20,
        help=(
            "Fraction of LAA voxels with disagreement > 0 (from prior fusion) "
            "required to set uncertain_artifact"
        ),
    )
    p.add_argument("--check-env", action="store_true", help="Check required packages and exit")
    return p.parse_args()


def _check_env() -> None:
    import sys
    details: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("nibabel", "numpy", "scipy"):
        try:
            m = __import__(pkg)
            details[pkg] = getattr(m, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            details[f"{pkg}_error"] = str(exc)
    print(json.dumps(details, indent=2))
    if any(k.endswith("_error") for k in details):
        raise SystemExit(2)


def _boundary_shell(mask: np.ndarray, shell_voxels: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask, iterations=shell_voxels)
    return mask & ~eroded


def _principal_axis_extent(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Return axis index (0/1/2) and sorted indices along that axis for masked voxels."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 2, np.array([])
    ranges = coords.max(axis=0) - coords.min(axis=0)
    axis = int(np.argmax(ranges))
    return axis, coords[:, axis]


def _distal_mask(mask: np.ndarray, distal_frac: float) -> np.ndarray:
    axis, axis_vals = _principal_axis_extent(mask)
    if axis_vals.size == 0:
        return np.zeros_like(mask, dtype=bool)
    ax_min, ax_max = axis_vals.min(), axis_vals.max()
    cutoff = ax_min + (1.0 - distal_frac) * (ax_max - ax_min)
    # Build index array along the principal axis
    idx = np.arange(mask.shape[axis])
    slicer = [np.newaxis] * 3
    slicer[axis] = slice(None)
    idx_broadcast = idx[tuple(slicer)] * np.ones(mask.shape, dtype=np.int32)
    return mask & (idx_broadcast >= cutoff)


def _safe_frac(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def main() -> int:
    args = _parse_args()
    if args.check_env:
        _check_env()
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    defect_img = nib.load(args.filling_defect_map)
    mask_img = nib.load(args.laa_mask)

    defect = np.asarray(defect_img.get_fdata(), dtype=np.uint8)
    mask = (np.asarray(mask_img.get_fdata()) > 0)

    if defect.shape != mask.shape:
        raise RuntimeError(
            f"filling-defect-map shape {defect.shape} != laa-mask shape {mask.shape}. "
            "Pre-register inputs to the same space."
        )

    roi_n = int(mask.sum())
    if roi_n == 0:
        print("WARNING: LAA mask is empty; all SLAAO labels will be False.")
        labels = SLAAOLabels()
        evidence: dict[str, object] = {"roi_voxels": 0}
    else:
        thrombus_mask = (defect == _LABEL_THROMBUS) & mask
        stag_mask = (defect == _LABEL_STAGNATION) & mask
        mixed_mask = (defect == _LABEL_MIXED) & mask
        dark_mask = thrombus_mask | stag_mask  # used for whole-LAA and rim

        thrombus_frac = _safe_frac(int(thrombus_mask.sum()), roi_n)
        stag_frac = _safe_frac(int(stag_mask.sum()), roi_n)
        mixed_frac = _safe_frac(int(mixed_mask.sum()), roi_n)
        dark_frac = _safe_frac(int(dark_mask.sum()), roi_n)

        # Rim pattern: boundary shell enriched with thrombus/stagnation
        shell = _boundary_shell(mask, args.rim_shell_voxels)
        shell_n = int(shell.sum())
        shell_dark_frac = _safe_frac(int((dark_mask & shell).sum()), shell_n)

        # Distal involvement and regional pooling
        distal = _distal_mask(mask, args.distal_frac)
        distal_n = int(distal.sum())
        distal_dark_frac = _safe_frac(int(((thrombus_mask | stag_mask) & distal).sum()), distal_n)

        pooling_distal = _distal_mask(mask, args.pooling_distal_frac)
        pooling_distal_n = int(pooling_distal.sum())
        pooling_stag_frac = _safe_frac(int((stag_mask & pooling_distal).sum()), pooling_distal_n)

        # Uncertain artifact from prior disagreement map
        uncertain_frac = 0.0
        if args.disagreement_map and Path(args.disagreement_map).exists():
            disagree_img = nib.load(args.disagreement_map)
            disagree = np.asarray(disagree_img.get_fdata())
            uncertain_frac = _safe_frac(int(((disagree > 0) & mask).sum()), roi_n)

        labels = SLAAOLabels(
            dark_thrombus_component=thrombus_frac >= args.thrombus_frac,
            contrast_stagnation=stag_frac >= args.stagnation_frac,
            rim_pattern=shell_dark_frac >= args.rim_frac,
            whole_LAA_involvement=dark_frac >= args.whole_laa_frac,
            regional_pooling=pooling_stag_frac >= args.pooling_stag_frac,
            distal_tip_involvement=distal_dark_frac > 0,
            mixed_pattern=mixed_frac >= args.mixed_frac,
            uncertain_artifact=uncertain_frac >= args.uncertain_disagree_frac,
        )

        evidence = {
            "roi_voxels": roi_n,
            "thrombus_frac": round(thrombus_frac, 4),
            "stagnation_frac": round(stag_frac, 4),
            "mixed_frac": round(mixed_frac, 4),
            "dark_frac": round(dark_frac, 4),
            "rim_shell_dark_frac": round(shell_dark_frac, 4),
            "distal_dark_frac": round(distal_dark_frac, 4),
            "pooling_distal_stag_frac": round(pooling_stag_frac, 4),
            "uncertain_disagree_frac": round(uncertain_frac, 4),
        }

    output = {
        "case_id": args.case_id,
        "slaao_labels": asdict(labels),
        "thresholds": {
            "thrombus_frac": args.thrombus_frac,
            "stagnation_frac": args.stagnation_frac,
            "mixed_frac": args.mixed_frac,
            "whole_laa_frac": args.whole_laa_frac,
            "rim_shell_voxels": args.rim_shell_voxels,
            "rim_frac": args.rim_frac,
            "distal_frac": args.distal_frac,
            "pooling_distal_frac": args.pooling_distal_frac,
            "pooling_stag_frac": args.pooling_stag_frac,
            "uncertain_disagree_frac": args.uncertain_disagree_frac,
        },
        "evidence": evidence,
    }

    out_path = out_dir / f"{args.case_id}_SLAAO_labels.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"SLAAO labels for {args.case_id}:")
    for feat, val in asdict(labels).items():
        print(f"  {feat}: {val}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
