#!/usr/bin/env python
"""Select dental-eligible CTAs (head/neck) from a directory of NIfTI + JSON sidecars.

The cohort is mixed (head/neck stroke CTAs + chest/pulmonary-embolism CTAs). The
dental pipeline only makes sense on scans whose FOV includes the dentition, so this
filters by the DICOM-derived metadata in each ``*_ct.json`` sidecar and writes a
manifest of eligible CTA paths (one per line) plus a classification CSV with reasons.

Example:
  python scripts/select_dental_eligible.py \
      --dir /Volumes/DICOM5/slaobids \
      --out-manifest outputs/dental_eligible.txt \
      --out-csv outputs/dental_eligibility.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

HEAD_BODYPARTS = {"HEAD", "NECK", "HEADNECK", "HEAD_NECK", "SKULL", "FACE", "BRAIN"}
HEAD_TERMS = ("head", "neck", "brain", "stroke", "tia", "carotid",
              "circle of willis", "cow", "intracranial")
CHEST_BODYPARTS = {"CHEST", "THORAX", "ABDOMEN", "CHEST_ABDOMEN", "CHEST_TO_PELVIS"}
CHEST_TERMS = ("pulmonary", "chest", "thorax", "lung", "embol", " pe ",
               "abdomen", "runoff", "coronary")


def classify(sidecar: dict) -> tuple[str, str]:
    """Return (eligibility, reason): eligibility in {eligible, excluded, ambiguous}."""
    body = (sidecar.get("BodyPartExamined") or "").upper().replace(" ", "")
    text = " ".join(str(sidecar.get(k, "")) for k in
                    ("ProtocolName", "StudyDescription", "SeriesDescription")).lower()
    is_head = body in HEAD_BODYPARTS or any(t in text for t in HEAD_TERMS)
    is_chest = body in CHEST_BODYPARTS or any(t in text for t in CHEST_TERMS)
    if is_head and not is_chest:
        return "eligible", f"head/neck ({body or 'by-description'})"
    if is_chest and not is_head:
        return "excluded", f"chest/abdomen ({body or 'by-description'})"
    return "ambiguous", f"body={body or '?'} desc='{text[:60]}'"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="Dir of <case>_acq-CTA_ct.nii.gz + .json")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="Also include ambiguous cases in the manifest.")
    args = ap.parse_args()

    rows, eligible = [], []
    for nii in sorted(glob.glob(str(Path(args.dir) / "*_acq-CTA_ct.nii.gz"))):
        nii = Path(nii)
        case = (re.search(r"(sub-[0-9A-Za-z]+)", nii.name) or [None, nii.stem])[1]
        sidecar = nii.with_suffix("").with_suffix(".json")
        try:
            meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        except Exception:
            meta = {}
        elig, reason = classify(meta) if meta else ("ambiguous", "no sidecar")
        rows.append((case, elig, reason, str(nii)))
        if elig == "eligible" or (args.include_ambiguous and elig == "ambiguous"):
            eligible.append(str(nii))

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text("\n".join(eligible) + ("\n" if eligible else ""))
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "eligibility", "reason", "path"])
            w.writerows(rows)

    from collections import Counter
    c = Counter(r[1] for r in rows)
    print(f"scanned {len(rows)} | " + " | ".join(f"{k}={v}" for k, v in c.most_common()))
    print(f"manifest ({len(eligible)} cases): {args.out_manifest}")


if __name__ == "__main__":
    main()
