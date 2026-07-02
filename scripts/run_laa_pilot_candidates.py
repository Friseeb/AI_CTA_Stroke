#!/usr/bin/env python3
"""Generate VISTA3D LAA first-mask candidates for the LAA pilot cohort.

For each pilot case, runs NV-Segment-CT / VISTA3D (label 108 = LAA) on the CTA
and writes the mask into the case's
  <case_dir>/laa_annotation/<reader>/candidate_masks/vista3d_laa.nii.gz
updating the case session.json. Skips cases whose candidate already exists
(e.g. sub-138 done, sub-547 has prior-fusion masks).

Run inside the env that has monai + transformers<5:
  conda run -n nv-segment-ct python scripts/run_laa_pilot_candidates.py \
    --pilot-root outputs/laa_pilot --reader readerA --device auto
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
VISTA_SCRIPT = REPO / "scripts" / "run_nv_segment_ct_laa.py"
MODEL_DIR = REPO / "external" / "nv_segment_ct"


def _voxels(path: Path) -> int:
    img = nib.load(str(path))
    return int((np.asarray(img.dataobj) > 0).sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-root", type=Path, default=REPO / "outputs" / "laa_pilot")
    ap.add_argument("--reader", default="readerA")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--label-id", default="108")
    ap.add_argument("--force", action="store_true", help="re-run even if candidate exists")
    args = ap.parse_args(argv)

    case_dirs = sorted(d for d in args.pilot_root.glob("sub-*") if d.is_dir())
    print(f"pilot cases: {[d.name for d in case_dirs]}")
    results = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        sess_path = case_dir / "laa_annotation" / args.reader / "logs" / f"{case_id}_session.json"
        if not sess_path.exists():
            print(f"[skip] {case_id}: no session.json")
            continue
        session = json.loads(sess_path.read_text())
        cand_dir = case_dir / "laa_annotation" / args.reader / "candidate_masks"
        cand_dir.mkdir(parents=True, exist_ok=True)
        out = cand_dir / "vista3d_laa.nii.gz"

        if out.exists() and not args.force:
            print(f"[have] {case_id}: {out.name} ({_voxels(out)} vox) — skip")
            results.append((case_id, "exists", _voxels(out)))
            continue
        # sub-547 already has staged consensus/vista3d masks
        if (cand_dir / "consensus_laa.nii.gz").exists() and not args.force:
            print(f"[have] {case_id}: consensus_laa already staged — skip")
            results.append((case_id, "exists_consensus", _voxels(cand_dir / "consensus_laa.nii.gz")))
            continue

        ct = session.get("cta_path", "")
        if not ct or not Path(ct).exists():
            print(f"[ERR ] {case_id}: CTA missing ({ct})")
            results.append((case_id, "no_cta", 0))
            continue

        log = case_dir / "vista3d.log"
        t0 = time.time()
        print(f"[run ] {case_id}: VISTA3D label {args.label_id} -> {out.name} ...", flush=True)
        cmd = [
            sys.executable, str(VISTA_SCRIPT),
            "--input", ct, "--output", str(out),
            "--label-id", str(args.label_id),
            "--model-dir", str(MODEL_DIR), "--device", args.device,
        ]
        with log.open("w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        # clean temp work dirs the pipeline leaves behind
        for tmp in cand_dir.glob("nv_segment_ct_*"):
            shutil.rmtree(tmp, ignore_errors=True)

        if rc != 0 or not out.exists():
            print(f"[FAIL] {case_id}: rc={rc} (see {log})")
            results.append((case_id, "fail", 0))
            continue
        vox = _voxels(out)
        session["candidate"] = f"candidate_masks/{out.name}"
        session["candidate_source"] = f"vista3d_label{args.label_id}"
        session["candidate_voxels"] = vox
        sess_path.write_text(json.dumps(session, indent=2))
        print(f"[ok  ] {case_id}: {vox} vox in {dt:.0f}s")
        results.append((case_id, "ok", vox))

    print("\n=== summary ===")
    for cid, status, vox in results:
        print(f"  {cid:10s} {status:16s} vox={vox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
