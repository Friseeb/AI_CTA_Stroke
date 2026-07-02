#!/usr/bin/env python
"""Generate real pharyngeal airway masks with TotalSegmentator (head_glands_cavities).

For each CTA it runs TS ``-ta head_glands_cavities`` (which segments
nasopharynx / oropharynx / hypopharynx — the OSA-relevant upper airway) and
unions those three labels into a single ``airway.nii.gz`` per case. The result
is a real, model-based airway far better than the stroke_cta_osa HU fallback
(no lung leak, a true anatomical min-CSA) and than the dental teeth-``pharynx``
label (which is oropharynx-only and fails in ~half of cases).

Designed to run on a CUDA box (e.g. the office DGX): ~seconds/case on GPU vs
~2.3 min/case on an Apple-MPS Mac.

Example (DGX, 4 parallel workers):
  python run_ts_airway_batch.py \
      --in-dir /path/slaobids \
      --out-dir /path/ts_airway \
      --device gpu --workers 4

Then feed the masks into stroke_cta_osa:
  # per case:  stroke-cta-osa extract CASE.nii.gz --out OUT \
  #              --external-airway-mask /path/ts_airway/<case>/airway.nii.gz
  # or batch:  stroke-cta-osa batch MANIFEST --out OUT \
  #              --airway-mask-dir /path/ts_airway     (see CLI --airway-mask-dir)

The per-case ``airway.nii.gz`` is written in the input CTA's geometry, so the
stroke pipeline consumes it directly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PHARYNX_LABELS = ("nasopharynx", "oropharynx", "hypopharynx")
TASK = "head_glands_cavities"

# BodyPartExamined values that confirm head/neck coverage
_INCLUDE_BODY_PARTS = {"NECK", "HEAD", "HEADNECK", "HEAD_NECK", "BRAIN", "CRANIOFACIAL"}
# BodyPartExamined values that definitively rule out pharyngeal coverage
_EXCLUDE_BODY_PARTS = {"CHEST", "ABDOMEN", "CHEST_ABDOMEN", "CHEST_TO_PELVIS",
                       "PELVIS", "THORAX", "HEART", "LUNG"}
# Protocol substrings (lowercase) that indicate head/neck CTA when body part is blank
_INCLUDE_PROTO_KW = {"stroke", "head", "neck", "carotid", "tia", "hyperacute", "cranial", "angio"}
# Protocol substrings (lowercase) that rule out head/neck when body part is blank
_EXCLUDE_PROTO_KW = {"chest", "pulmonar", "abdomen", "pelvis", "cap ", "cap-",
                     "heart", "robotic", " pe ", "pe-", "dissection", "aorta",
                     "renal", "portal", "liver", "colon"}


def case_id_from_path(p: Path) -> str:
    name = p.name
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    m = re.match(r"(sub-[0-9A-Za-z]+)", name)
    return m.group(1) if m else name


def is_head_neck_cta(p: Path) -> tuple[bool, str]:
    """Check BIDS JSON sidecar for BodyPartExamined / ProtocolName.

    Returns (include, reason). Falls back to True (don't drop) if no sidecar.
    """
    # Sidecar lives next to the NIfTI with the same stem
    stem = p.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    sidecar = p.parent / f"{stem}.json"
    if not sidecar.is_file():
        return True, "no sidecar — assuming head/neck CTA"

    try:
        import json as _json
        meta = _json.loads(sidecar.read_text())
    except Exception as exc:
        return True, f"sidecar unreadable ({exc}) — assuming head/neck CTA"

    body_part = (meta.get("BodyPartExamined") or "").strip().upper().replace(" ", "_")
    protocol  = (meta.get("ProtocolName") or "").strip().lower()

    if body_part in _INCLUDE_BODY_PARTS:
        return True, f"BodyPartExamined={body_part}"
    if body_part in _EXCLUDE_BODY_PARTS:
        return False, f"BodyPartExamined={body_part}"

    # Body part is blank or unknown — fall back to protocol name keywords
    if any(kw in protocol for kw in _INCLUDE_PROTO_KW):
        return True, f"ProtocolName contains head/neck keyword ({protocol!r})"
    if any(kw in protocol for kw in _EXCLUDE_PROTO_KW):
        return False, f"ProtocolName suggests non-head/neck scan ({protocol!r})"

    # Can't determine — be conservative and include (TS will return EMPTY if no pharynx)
    return True, f"undetermined body_part={body_part!r} protocol={protocol!r} — including"


def collect_inputs(manifest: Path | None, in_dir: Path | None, glob: str) -> list[Path]:
    if manifest is not None:
        lines = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
        return [Path(x) for x in lines if Path(x).exists()]
    if in_dir is not None:
        return sorted(in_dir.glob(glob))
    return []


def run_ts(cta: Path, out_dir: Path, device: str, fast: bool) -> bool | float:
    """Run TS head_glands_cavities into a temp dir, union pharyngeal labels."""
    import SimpleITK as sitk
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        cmd = ["TotalSegmentator", "-i", str(cta), "-o", td, "-ta", TASK,
               "--device", device]
        if fast:
            cmd.append("--fast")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(f"[TS FAIL] {cta.name}\n{r.stderr[-800:]}\n")
            return False
        union = None
        ref = None
        for lab in PHARYNX_LABELS:
            f = Path(td) / f"{lab}.nii.gz"
            if not f.is_file():
                continue
            im = sitk.ReadImage(str(f))
            ref = im
            a = sitk.GetArrayFromImage(im) > 0
            union = a if union is None else (union | a)
        if union is None or not union.any():
            sys.stderr.write(f"[EMPTY] {cta.name}: no pharyngeal labels\n")
            return False
        out_dir.mkdir(parents=True, exist_ok=True)
        out_img = sitk.GetImageFromArray(union.astype("uint8"))
        out_img.CopyInformation(ref)
        sitk.WriteImage(out_img, str(out_dir / "airway.nii.gz"), useCompression=True)
        vox_ml = float(np.prod(ref.GetSpacing())) / 1000.0
        return float(union.sum() * vox_ml)


def process_case(args: tuple) -> tuple[str, str, str, str]:
    """Worker target: returns (case_id, cta_path, airway_path, status)."""
    cta, out_dir_root, device, fast, skip_existing, idx, total = args
    cid = case_id_from_path(cta)
    cdir = out_dir_root / cid
    airway = cdir / "airway.nii.gz"

    prefix = f"[{idx}/{total}] {cid}"

    if skip_existing and airway.is_file():
        print(f"{prefix}  skip (exists)", flush=True)
        return cid, str(cta), str(airway), "skipped"

    # CTA verification via JSON sidecar
    ok, reason = is_head_neck_cta(cta)
    if not ok:
        print(f"{prefix}  SKIP ({reason})", flush=True)
        return cid, str(cta), "", f"skipped_not_head_neck:{reason}"

    print(f"{prefix}  segmenting...", flush=True)
    vol = run_ts(cta, cdir, device, fast)
    if vol:
        print(f"{prefix}  airway {vol:.1f} ml -> {airway}", flush=True)
        return cid, str(cta), str(airway), f"ok:{vol:.1f}ml"
    else:
        return cid, str(cta), "", "failed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="Text file, one CTA path per line.")
    src.add_argument("--in-dir", type=Path, help="Directory of CTAs.")
    ap.add_argument("--glob", default="*_acq-CTA_ct.nii.gz")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Per-case airway masks go to <out-dir>/<case_id>/airway.nii.gz")
    ap.add_argument("--device", default="gpu", help="gpu | cpu | mps (TS --device)")
    ap.add_argument("--fast", action="store_true", help="TS --fast (3mm, quicker/coarser)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel TotalSegmentator workers (default 1). "
                         "On a GPU with large unified memory (e.g. GB10) try 4.")
    args = ap.parse_args()

    inputs = collect_inputs(args.manifest, args.in_dir, args.glob)
    if not inputs:
        sys.exit("no inputs found")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = len(inputs)
    print(f"Found {total} inputs. workers={args.workers} device={args.device}", flush=True)

    job_args = [
        (cta, args.out_dir, args.device, args.fast, args.skip_existing, i + 1, total)
        for i, cta in enumerate(inputs)
    ]

    manifest_rows: list[tuple] = []
    n_ok = n_skip = n_fail = n_not_cta = 0

    if args.workers == 1:
        for a in job_args:
            cid, cta_path, airway_path, status = process_case(a)
            manifest_rows.append((cid, cta_path, airway_path, status))
            if status == "skipped":                   n_skip += 1
            elif status.startswith("ok"):             n_ok += 1
            elif status.startswith("skipped_not"):    n_not_cta += 1
            else:                                     n_fail += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_case, a): a for a in job_args}
            for fut in concurrent.futures.as_completed(futures):
                cid, cta_path, airway_path, status = fut.result()
                manifest_rows.append((cid, cta_path, airway_path, status))
                if status == "skipped":                   n_skip += 1
                elif status.startswith("ok"):             n_ok += 1
                elif status.startswith("skipped_not"):    n_not_cta += 1
                else:                                     n_fail += 1

    # sort manifest by case_id for reproducibility
    manifest_rows.sort(key=lambda r: r[0])

    man = args.out_dir / "airway_manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "cta_path", "airway_mask_path", "status"])
        w.writerows(manifest_rows)
    print(
        f"\ndone: {n_ok} ok, {n_skip} skipped, {n_fail} failed, "
        f"{n_not_cta} not-CTA. manifest -> {man}",
        flush=True,
    )


if __name__ == "__main__":
    main()
