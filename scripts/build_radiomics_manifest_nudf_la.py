#!/usr/bin/env python3
"""
Build a radiomics manifest for NUDF LAA + LA highres outputs in DAYLIGHTBIDS.

For each case in derivatives/nudf_la/sub-*/:
  - expects {case_id}_laa_nudf.nii.gz
  - expects {case_id}_left_atrium_highres.nii.gz
  - finds heartchambers_highres.nii.gz under cardiac_ct_explorer/**
  - extracts aorta label (default 6) to {case_id}_aorta_highres.nii.gz
  - matches defaced CTA in derivatives/defaced/{case_id}_defaced.nii.gz

Outputs a CSV manifest:
  case_id, cta_path, aorta_mask, la_mask, laa_mask
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build radiomics manifest for NUDF+LA outputs")
    p.add_argument("--root", default="./data/daylightbids", help="DAYLIGHTBIDS root")
    p.add_argument("--nudf-dir", default=None, help="NUDF output dir (default: root/derivatives/nudf_la)")
    p.add_argument("--defaced-dir", default=None, help="Defaced dir (default: root/derivatives/defaced)")
    p.add_argument(
        "--out-manifest",
        default=None,
        help="Output manifest CSV (default: root/derivatives/nudf_la/radiomics_manifest.csv)",
    )
    p.add_argument("--aorta-label", type=int, default=6, help="Label id for aorta in heartchambers_highres")
    p.add_argument("--force-aorta", action="store_true", help="Recompute aorta mask even if it exists")
    p.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    p.add_argument("--dry-run", action="store_true", help="Print actions only")
    p.add_argument("--include-missing", action="store_true", help="Include rows even if some paths are missing")
    return p.parse_args()


def _find_heartchambers(case_dir: Path) -> Path | None:
    matches = list(case_dir.glob("**/heartchambers_highres.nii.gz"))
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _extract_aorta(heartchambers_path: Path, out_path: Path, label_id: int, dry_run: bool) -> bool:
    if dry_run:
        print(f"[dry-run] extract aorta label {label_id} -> {out_path}")
        return True
    img = nib.load(str(heartchambers_path))
    data = np.asarray(img.dataobj)
    aorta = (data == label_id).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(aorta, img.affine, img.header), str(out_path))
    return True


def _iter_cases(nudf_dir: Path) -> Iterable[Path]:
    return sorted([p for p in nudf_dir.glob("sub-*") if p.is_dir()])


def _maybe_print_missing(case_id: str, missing: list[str]) -> None:
    if missing:
        print(f"⚠ {case_id} missing: {', '.join(missing)}")


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    nudf_dir = Path(args.nudf_dir) if args.nudf_dir else root / "derivatives" / "nudf_la"
    defaced_dir = Path(args.defaced_dir) if args.defaced_dir else root / "derivatives" / "defaced"
    out_manifest = (
        Path(args.out_manifest)
        if args.out_manifest
        else root / "derivatives" / "nudf_la" / "radiomics_manifest.csv"
    )

    if not nudf_dir.exists():
        raise FileNotFoundError(f"NUDF dir not found: {nudf_dir}")
    if not defaced_dir.exists():
        raise FileNotFoundError(f"Defaced dir not found: {defaced_dir}")

    rows = []
    missing_counts = {"cta": 0, "laa": 0, "la": 0, "aorta": 0, "heartchambers": 0}

    for idx, case_dir in enumerate(_iter_cases(nudf_dir)):
        if args.limit is not None and idx >= args.limit:
            break

        case_id = case_dir.name
        defaced_path = defaced_dir / f"{case_id}_defaced.nii.gz"
        laa_path = case_dir / f"{case_id}_laa_nudf.nii.gz"
        la_path = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
        aorta_path = case_dir / f"{case_id}_aorta_highres.nii.gz"

        missing = []
        if not defaced_path.exists():
            missing.append("cta")
            missing_counts["cta"] += 1
        if not laa_path.exists():
            missing.append("laa")
            missing_counts["laa"] += 1
        if not la_path.exists():
            missing.append("la")
            missing_counts["la"] += 1

        heartchambers = _find_heartchambers(case_dir)
        if heartchambers is None:
            missing.append("heartchambers")
            missing_counts["heartchambers"] += 1
        else:
            if args.force_aorta or not aorta_path.exists():
                _extract_aorta(heartchambers, aorta_path, args.aorta_label, args.dry_run)
            if not aorta_path.exists() and not args.dry_run:
                missing.append("aorta")
                missing_counts["aorta"] += 1

        if missing and not args.include_missing:
            _maybe_print_missing(case_id, missing)
            continue

        rows.append(
            {
                "case_id": case_id,
                "cta_path": str(defaced_path),
                "aorta_mask": str(aorta_path),
                "la_mask": str(la_path),
                "laa_mask": str(laa_path),
            }
        )

    if args.dry_run:
        print(f"[dry-run] would write manifest with {len(rows)} rows -> {out_manifest}")
        return 0

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "cta_path", "aorta_mask", "la_mask", "laa_mask"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_manifest}")
    if any(missing_counts.values()):
        print("Missing counts:", ", ".join(f"{k}={v}" for k, v in missing_counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
