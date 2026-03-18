#!/usr/bin/env python3
"""
Fix a vertebral labelmap outside Slicer by resampling to CTA geometry.

This makes the labelmap load correctly in ITK-Snap / downstream pipelines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def _print_geom(img: sitk.Image, name: str) -> None:
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    print(f"{name}: size={size}, spacing={spacing}, origin={origin}, direction={direction}")


def fix_label(
    cta_path: Path,
    label_path: Path,
    output_path: Path,
    copy_only: bool = False,
) -> None:
    cta = sitk.ReadImage(str(cta_path))
    label = sitk.ReadImage(str(label_path))

    _print_geom(cta, "CTA")
    _print_geom(label, "Label (input)")

    if copy_only:
        if cta.GetSize() != label.GetSize():
            raise RuntimeError("copy-only requested but label size differs from CTA.")
        fixed = sitk.Cast(label, sitk.sitkUInt16)
        fixed.CopyInformation(cta)
    else:
        fixed = sitk.Resample(
            label,
            cta,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt16,
        )
        fixed.CopyInformation(cta)

    _print_geom(fixed, "Label (fixed)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(fixed, str(output_path), True)
    print(f"Saved fixed label: {output_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix labelmap geometry outside Slicer")
    ap.add_argument("--cta", default=None, help="CTA NIfTI (single case)")
    ap.add_argument("--label", default=None, help="Labelmap NIfTI (single case)")
    ap.add_argument("--output", default=None, help="Output fixed labelmap (single case)")
    ap.add_argument("--root", default=None, help="Root folder containing CTA + label files")
    ap.add_argument(
        "--pattern",
        default="sub-*_acq-CTA_ct.nii.gz",
        help="Glob for CTA files under --root",
    )
    ap.add_argument(
        "--copy-only",
        action="store_true",
        help="Only copy header geometry (no resample). Requires same size.",
    )
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
        if not root.exists():
            raise FileNotFoundError(f"Root not found: {root}")
        cta_files = sorted(root.glob(args.pattern))
        if not cta_files:
            raise RuntimeError(f"No CTA files found under {root} with pattern {args.pattern}")
        out_dir = root / "derivatives" / "vertebral_manual"
        out_dir.mkdir(parents=True, exist_ok=True)
        for cta_path in cta_files:
            base = cta_path.name.replace("_acq-CTA_ct.nii.gz", "")
            # Prefer exact expected label names, but allow hyphen variants and generic match
            candidates = [
                out_dir / f"{base}_acq-CTA_ct_vert.nii.gz",
                out_dir / f"{base}-acq-CTA-ct_vert.nii.gz",
            ]
            label_path = None
            for cand in candidates:
                if cand.exists():
                    label_path = cand
                    break
            if label_path is None:
                # Fallback: any label containing base + 'vert' (exclude already-clean)
                glob_matches = sorted(out_dir.glob(f"{base}*vert*.nii*"))
                glob_matches = [p for p in glob_matches if "clean" not in p.name]
                if glob_matches:
                    # choose most recent
                    label_path = max(glob_matches, key=lambda p: p.stat().st_mtime)

            if label_path is None or not label_path.exists():
                print(f"Skipping (missing label): {out_dir}/{base}*_vert*.nii.gz")
                continue

            output_path = out_dir / f"{base}_vert_clean.nii.gz"
            print(f"Fixing {base} -> {output_path.name}")
            fix_label(
                cta_path=cta_path,
                label_path=label_path,
                output_path=output_path,
                copy_only=args.copy_only,
            )
        return 0

    if not args.cta or not args.label or not args.output:
        raise RuntimeError("Provide --cta/--label/--output or use --root for batch mode.")

    fix_label(
        cta_path=Path(args.cta),
        label_path=Path(args.label),
        output_path=Path(args.output),
        copy_only=args.copy_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
