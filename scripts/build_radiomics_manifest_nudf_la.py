#!/usr/bin/env python3
"""
Build radiomics manifest for ALL CT types in SLAAOBIDS.

Two modes (--mode):

  nudf  (default)
    Scans two segmentation output directories produced by run_full_seg_batch.py:
      derivatives/nudf_la_eCTA/    -> eCTA patients
      derivatives/nudf_la_multict/ -> all other CT types
    Masks expected per case:
      {case_id}_laa_vista3d.nii.gz          <- VISTA3D label 108
      {case_id}_left_atrium_highres.nii.gz  <- TotalSegmentator heartchambers label 2
      {case_id}_aorta_highres_ts.nii.gz     <- TotalSegmentator heartchambers label 6
    Output: <root>/derivatives/radiomics_manifest_all.csv

  totalseg
    Scans the same nudf_la_eCTA / nudf_la_multict directories as nudf mode.
    Masks expected per case (same filenames as nudf):
      {case_id}_laa_vista3d.nii.gz          <- LAA
      {case_id}_left_atrium_highres.nii.gz  <- LA
      {case_id}_aorta_highres_ts.nii.gz     <- Aorta
    Output: <root>/derivatives/radiomics_manifest_ts_laa.csv

CTA image path (both modes):
  eCTA      -> <root>/derivatives/defaced/{case_id}.nii.gz   (already defaced)
  non-eCTA  -> <root>/<sub_id>/<case_id>.nii.gz

Cases where any mask or CTA file is missing are skipped.
Run with --dry-run to see a preview without writing the CSV.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_CT_TYPE_DISPLAY: Dict[str, str] = {
    "ecta":      "eCTA",
    "ctthorax":  "CT_thorax",
    "ctheart":   "CT_heart",
    "ctbody":    "CT_totalbody",
    "ctabdomen": "CT_abdomen",
}

_FIELDNAMES = ["sub_id", "case_id", "ct_type", "cta_path", "laa_mask", "la_mask", "aorta_mask"]

# TotalSegmentator 'total' task label for atrial_appendage_left
_TS_LAA_LABEL = 61


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build radiomics manifest for all CT types in SLAAOBIDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default="C:/Users/spost/Desktop/CT_image/SLAAOBIDS",
        help="SLAAOBIDS root directory",
    )
    p.add_argument(
        "--mode",
        choices=["nudf", "totalseg"],
        default="nudf",
        help=(
            "nudf: scan nudf_la_eCTA / nudf_la_multict → radiomics_manifest_all.csv. "
            "totalseg: same dirs, same masks → radiomics_manifest_ts_laa.csv."
        ),
    )
    p.add_argument(
        "--out-manifest",
        default=None,
        help=(
            "Output CSV path. Defaults: "
            "nudf -> <root>/derivatives/radiomics_manifest_all.csv ; "
            "totalseg -> <root>/derivatives/radiomics_manifest_ts_laa.csv"
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Limit total cases scanned")
    p.add_argument("--dry-run", action="store_true", help="Print summary without writing CSV")
    p.add_argument(
        "--qc-csv",
        default=None,
        help=(
            "Path to a QC dashboard CSV export (seg tab). When provided, the manifest is "
            "filtered to green cases only: status='fixed' or empty/unreviewed are included; "
            "status='pending' cases are excluded and logged; status='not_fixable' cases are "
            "excluded silently. Latest timestamp per case_id is used."
        ),
    )
    return p.parse_args()


def _detect_acq_type(case_id: str) -> str:
    """Extract acquisition type from case_id like sub-224_acq-ctheart_ph1_ct."""
    m = re.search(r"_acq-(\w+?)_", case_id)
    return m.group(1) if m else "unknown"


def _find_heartchambers(case_dir: Path) -> Path | None:
    """Find heartchambers_highres.nii.gz anywhere under case_dir."""
    matches = sorted(case_dir.rglob("heartchambers_highres.nii.gz"), key=lambda p: len(p.parts))
    return matches[0] if matches else None


def _extract_label(src_path: Path, out_path: Path, label: int, name: str = "") -> bool:
    """Extract a single integer label from a NIfTI segmentation and save as binary mask."""
    try:
        import nibabel as nib
        import numpy as np
        img  = nib.load(str(src_path))
        data = np.asarray(img.dataobj)
        mask = (data == label).astype(np.uint8)
        if mask.sum() == 0:
            print(f"    [warn] label {label} ({name}) empty in {src_path.name}")
            return False
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] could not extract label {label} ({name}) from {src_path}: {exc}")
        return False


def _scan_cardiac_ct_explorer_dir(
    root: Path,
    limit: int | None,
) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
    """
    Scan cardiac_ct_explorer_* directories (previous-version pipeline).

    For each case:
      - Discovers case_id from the TotalSegmentator/<case_id>/ subdir name.
      - Extracts atrial_appendage_left (label 61) from total.nii.gz.
      - Extracts heart_atrium_left (label 2) and aorta (label 6) from heartchambers_highres.nii.gz.
      - Saves extracted masks alongside the cardiac_ct_explorer dir if not already present.
    """
    rows: List[dict] = []
    included_ct: Dict[str, int] = defaultdict(int)
    skipped_ct:  Dict[str, int] = defaultdict(int)

    derivs = root / "derivatives"
    if not derivs.exists():
        return rows, dict(included_ct), dict(skipped_ct)

    cce_dirs = sorted(
        [p for p in derivs.iterdir() if p.is_dir() and p.name.startswith("cardiac_ct_explorer_")]
    )
    if limit is not None:
        cce_dirs = cce_dirs[:limit]

    for cce_dir in cce_dirs:
        ts_root = cce_dir / "TotalSegmentator"
        if not ts_root.exists():
            continue
        ts_subdirs = [p for p in ts_root.iterdir() if p.is_dir()]
        if not ts_subdirs:
            continue
        ts_case_dir = ts_subdirs[0]
        case_id = ts_case_dir.name

        total_nii        = ts_case_dir / "total.nii.gz"
        heartchambers_nii = ts_case_dir / "heartchambers_highres.nii.gz"

        laa_mask   = cce_dir / f"{case_id}_atrial_appendage_left.nii.gz"
        la_mask    = cce_dir / f"{case_id}_left_atrium_highres.nii.gz"
        aorta_mask = cce_dir / f"{case_id}_aorta_highres_ts.nii.gz"

        # Extract masks if not already on disk
        if not laa_mask.exists() and total_nii.exists():
            ok = _extract_label(total_nii, laa_mask, label=_TS_LAA_LABEL, name="atrial_appendage_left")
            if ok:
                print(f"  ✓ extracted laa_mask (label {_TS_LAA_LABEL}) for {case_id}")

        if not la_mask.exists() and heartchambers_nii.exists():
            ok = _extract_label(heartchambers_nii, la_mask, label=2, name="heart_atrium_left")
            if ok:
                print(f"  ✓ extracted la_mask (label 2) for {case_id}")

        if not aorta_mask.exists() and heartchambers_nii.exists():
            ok = _extract_label(heartchambers_nii, aorta_mask, label=6, name="aorta")
            if ok:
                print(f"  ✓ extracted aorta_mask (label 6) for {case_id}")

        # Determine image path and CT type
        sub_id = case_id.split("_acq-")[0]
        if "_defaced" in case_id or "ecta" in case_id.lower():
            cta_path = root / "derivatives" / "defaced" / f"{case_id}.nii.gz"
            ct_type  = "eCTA"
        else:
            cta_path = root / sub_id / f"{case_id}.nii.gz"
            acq_type = _detect_acq_type(case_id)
            ct_type  = _CT_TYPE_DISPLAY.get(acq_type, acq_type)

        missing = []
        if not laa_mask.exists():
            missing.append("laa_mask")
        if not la_mask.exists():
            missing.append("la_mask")
        if not aorta_mask.exists():
            missing.append("aorta_mask")
        if not cta_path.exists():
            missing.append("cta")

        if missing:
            print(f"  ⚠ skip {case_id}  missing: {', '.join(missing)}")
            skipped_ct[ct_type] += 1
            continue

        rows.append({
            "sub_id":     sub_id,
            "case_id":    case_id,
            "ct_type":    ct_type,
            "cta_path":   str(cta_path),
            "laa_mask":   str(laa_mask),
            "la_mask":    str(la_mask),
            "aorta_mask": str(aorta_mask),
        })
        included_ct[ct_type] += 1

    return rows, dict(included_ct), dict(skipped_ct)


def _cta_path_ecta(root: Path, case_id: str) -> Path:
    return root / "derivatives" / "defaced" / f"{case_id}_defaced.nii.gz"


def _cta_path_multict(root: Path, case_id: str) -> Path:
    # case_id: "sub-224_acq-ctheart_ph1_ct"  -> sub_id: "sub-224"
    sub_id = case_id.split("_acq-")[0]
    return root / sub_id / f"{case_id}.nii.gz"


def _scan_seg_dir(
    seg_dir: Path,
    root: Path,
    is_ecta: bool,
    limit: int | None,
) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
    """
    Scan one seg output dir and build manifest rows.

    Returns:
        rows          : list of dicts ready for CSV
        included_ct   : {ct_type_display: count} of included cases
        skipped_ct    : {ct_type_display: count} of skipped cases (missing files)
    """
    rows: List[dict] = []
    included_ct: Dict[str, int] = defaultdict(int)
    skipped_ct:  Dict[str, int] = defaultdict(int)

    if not seg_dir.exists():
        return rows, dict(included_ct), dict(skipped_ct)

    case_dirs = sorted([p for p in seg_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if limit is not None:
        case_dirs = case_dirs[:limit]

    for case_dir in case_dirs:
        case_id = case_dir.name

        laa_mask   = case_dir / f"{case_id}_laa_vista3d.nii.gz"
        la_mask    = case_dir / f"{case_id}_left_atrium_highres.nii.gz"
        aorta_mask = case_dir / f"{case_id}_aorta_highres_ts.nii.gz"

        # sub_id: first component before any _acq- suffix
        sub_id = case_id.split("_acq-")[0]

        if is_ecta:
            cta_path = _cta_path_ecta(root, case_id)
            ct_type  = "eCTA"
        else:
            cta_path = _cta_path_multict(root, case_id)
            acq_type = _detect_acq_type(case_id)
            ct_type  = _CT_TYPE_DISPLAY.get(acq_type, acq_type)

        # Auto-extract LA from heartchambers_highres if missing at top level
        if not la_mask.exists():
            hc = _find_heartchambers(case_dir)
            if hc is not None:
                ok = _extract_label(hc, la_mask, label=2, name="heart_atrium_left")
                if ok:
                    print(f"  ✓ extracted la_mask from heartchambers for {case_id}")

        missing = []
        if not laa_mask.exists():
            missing.append("laa_mask")
        if not la_mask.exists():
            missing.append("la_mask")
        if not aorta_mask.exists():
            missing.append("aorta_mask")
        if not cta_path.exists():
            missing.append("cta")

        if missing:
            print(f"  ⚠ skip {case_id}  missing: {', '.join(missing)}")
            skipped_ct[ct_type] += 1
            continue

        rows.append({
            "sub_id":     sub_id,
            "case_id":    case_id,
            "ct_type":    ct_type,
            "cta_path":   str(cta_path),
            "laa_mask":   str(laa_mask),
            "la_mask":    str(la_mask),
            "aorta_mask": str(aorta_mask),
        })
        included_ct[ct_type] += 1

    return rows, dict(included_ct), dict(skipped_ct)


def _load_qc_statuses(qc_csv: Path) -> Dict[str, str]:
    """Return {case_id: latest_status} from a QC dashboard seg-tab CSV export.

    Takes the row with the highest timestamp string per case_id (ISO-8601 sorts lexicographically).
    Status values used by the dashboard: '' (unreviewed), 'pending', 'fixed', 'not_fixable'.
    """
    latest: Dict[str, Tuple[str, str]] = {}  # case_id -> (timestamp, status)
    with qc_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id", "").strip()
            if not cid:
                continue
            ts = row.get("timestamp", "").strip()
            status = row.get("status", "").strip()
            if cid not in latest or ts > latest[cid][0]:
                latest[cid] = (ts, status)
    return {cid: v[1] for cid, v in latest.items()}


def _apply_qc_filter(
    rows: List[dict],
    qc_statuses: Dict[str, str],
) -> Tuple[List[dict], List[str], int]:
    """Filter manifest rows by QC seg status.

    Green  (included): status == 'fixed' or '' (unreviewed / absent from QC CSV).
    Pending (excluded): status == 'pending'   — returned for separate logging.
    Not fixable (excluded): status == 'not_fixable' — counted only.

    Returns:
        green_rows       : rows that pass the filter
        pending_case_ids : case_ids excluded because status == 'pending'
        not_fixable_count: number of rows excluded because status == 'not_fixable'
    """
    green: List[dict] = []
    pending_ids: List[str] = []
    not_fixable = 0
    for row in rows:
        cid = row["case_id"]
        status = qc_statuses.get(cid, "")  # absent from QC CSV = unreviewed = green
        if status in ("", "fixed"):
            green.append(row)
        elif status == "pending":
            pending_ids.append(cid)
        elif status == "not_fixable":
            not_fixable += 1
    return green, pending_ids, not_fixable


def main() -> int:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    default_manifest = (
        "radiomics_manifest_ts_laa.csv" if args.mode == "totalseg"
        else "radiomics_manifest_all.csv"
    )
    out_manifest = (
        Path(args.out_manifest) if args.out_manifest
        else root / "derivatives" / default_manifest
    )

    print(f"Root          : {root}")
    print(f"Mode          : {args.mode}")
    print()

    all_rows: List[dict] = []
    all_included: Dict[str, int] = defaultdict(int)
    all_skipped:  Dict[str, int] = defaultdict(int)

    if args.mode == "totalseg":
        ecta_dir    = root / "derivatives" / "nudf_la_eCTA"
        multict_dir = root / "derivatives" / "nudf_la_multict"
        print(f"eCTA seg dir  : {ecta_dir}  {'(found)' if ecta_dir.exists() else '(NOT FOUND)'}")
        print(f"MultiCT dir   : {multict_dir}  {'(found)' if multict_dir.exists() else '(NOT FOUND)'}")
        print()

        ecta_rows, ecta_inc, ecta_skip = _scan_seg_dir(
            ecta_dir, root, is_ecta=True, limit=args.limit
        )
        all_rows.extend(ecta_rows)
        for k, v in ecta_inc.items():
            all_included[k] += v
        for k, v in ecta_skip.items():
            all_skipped[k] += v

        multi_rows, multi_inc, multi_skip = _scan_seg_dir(
            multict_dir, root, is_ecta=False, limit=args.limit
        )
        all_rows.extend(multi_rows)
        for k, v in multi_inc.items():
            all_included[k] += v
        for k, v in multi_skip.items():
            all_skipped[k] += v
    else:
        ecta_dir    = root / "derivatives" / "nudf_la_eCTA"
        multict_dir = root / "derivatives" / "nudf_la_multict"
        print(f"eCTA seg dir  : {ecta_dir}  {'(found)' if ecta_dir.exists() else '(NOT FOUND)'}")
        print(f"MultiCT dir   : {multict_dir}  {'(found)' if multict_dir.exists() else '(NOT FOUND)'}")
        print()

        ecta_rows, ecta_inc, ecta_skip = _scan_seg_dir(
            ecta_dir, root, is_ecta=True, limit=args.limit
        )
        all_rows.extend(ecta_rows)
        for k, v in ecta_inc.items():
            all_included[k] += v
        for k, v in ecta_skip.items():
            all_skipped[k] += v

        multi_rows, multi_inc, multi_skip = _scan_seg_dir(
            multict_dir, root, is_ecta=False, limit=args.limit
        )
        all_rows.extend(multi_rows)
        for k, v in multi_inc.items():
            all_included[k] += v
        for k, v in multi_skip.items():
            all_skipped[k] += v

    # Summary table
    all_types = list(_CT_TYPE_DISPLAY.values())  # ordered
    extra_types = sorted(
        t for t in (set(all_included) | set(all_skipped)) if t not in all_types
    )
    display_order = all_types + extra_types

    print(f"{'CT type':<16}  {'included':>8}  {'skipped (missing masks)':>24}")
    print("-" * 54)
    total_inc = total_skip = 0
    for ct in display_order:
        inc  = all_included.get(ct, 0)
        skip = all_skipped.get(ct, 0)
        if inc + skip == 0:
            continue
        print(f"  {ct:<14}  {inc:>8}  {skip:>24}")
        total_inc  += inc
        total_skip += skip
    print(f"  {'TOTAL':<14}  {total_inc:>8}  {total_skip:>24}")
    print()

    # ── QC filter ─────────────────────────────────────────────────────────────
    if args.qc_csv:
        qc_path = Path(args.qc_csv)
        if not qc_path.exists():
            raise FileNotFoundError(f"--qc-csv not found: {qc_path}")
        qc_statuses = _load_qc_statuses(qc_path)
        all_rows, pending_ids, not_fixable_count = _apply_qc_filter(all_rows, qc_statuses)
        print(f"QC filter applied  : {qc_path.name}")
        print(f"  Green (pass)     : {len(all_rows)}")
        if pending_ids:
            print(f"  Pending (excluded, {len(pending_ids)} cases):")
            for cid in sorted(pending_ids):
                print(f"    {cid}")
        else:
            print(f"  Pending (excluded): 0")
        print(f"  Not fixable (excluded): {not_fixable_count}")
        print()

    if args.dry_run:
        print(f"[dry-run] would write {len(all_rows)} rows -> {out_manifest}")
        return 0

    if not all_rows:
        print("No cases with complete masks found. Run run_full_seg_batch.py first.")
        return 1

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
