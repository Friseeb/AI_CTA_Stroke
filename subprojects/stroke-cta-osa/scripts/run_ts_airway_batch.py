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

Example (DGX):
  python run_ts_airway_batch.py \
      --manifest /path/slao_eligible.txt \
      --out-dir  /path/ts_airway \
      --device gpu --workers 1

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
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

PHARYNX_LABELS = ("nasopharynx", "oropharynx", "hypopharynx")
TASK = "head_glands_cavities"


def case_id_from_path(p: Path) -> str:
    name = p.name
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    # sub-1023_acq-CTA_ct -> sub-1023
    import re
    m = re.match(r"(sub-[0-9A-Za-z]+)", name)
    return m.group(1) if m else name


def collect_inputs(manifest: Path | None, in_dir: Path | None, glob: str) -> list[Path]:
    if manifest is not None:
        lines = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
        return [Path(x) for x in lines if Path(x).exists()]
    if in_dir is not None:
        return sorted(in_dir.glob(glob))
    return []


def run_ts(cta: Path, out_dir: Path, device: str, fast: bool) -> bool:
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
    args = ap.parse_args()

    inputs = collect_inputs(args.manifest, args.in_dir, args.glob)
    if not inputs:
        sys.exit("no inputs found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_ok = n_skip = n_fail = 0
    for i, cta in enumerate(inputs, 1):
        cid = case_id_from_path(cta)
        cdir = args.out_dir / cid
        airway = cdir / "airway.nii.gz"
        if args.skip_existing and airway.is_file():
            print(f"[{i}/{len(inputs)}] {cid}  skip (exists)")
            manifest_rows.append((cid, str(cta), str(airway), "skipped"))
            n_skip += 1
            continue
        print(f"[{i}/{len(inputs)}] {cid}  segmenting...", flush=True)
        vol = run_ts(cta, cdir, args.device, args.fast)
        if vol:
            print(f"    airway {vol:.1f} ml -> {airway}")
            manifest_rows.append((cid, str(cta), str(airway), f"ok:{vol:.1f}ml"))
            n_ok += 1
        else:
            manifest_rows.append((cid, str(cta), "", "failed"))
            n_fail += 1

    man = args.out_dir / "airway_manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "cta_path", "airway_mask_path", "status"])
        w.writerows(manifest_rows)
    print(f"\ndone: {n_ok} ok, {n_skip} skipped, {n_fail} failed. manifest -> {man}")


if __name__ == "__main__":
    main()
