#!/usr/bin/env python3
"""Single-case driver: heart crop (TotalSegmentator, skip-if-done) + HolOrama whole-heart segmentation.

  python run_heart_segmentation_case.py --cta <sub-N_acq-CTA_ct.nii.gz> --out-root outputs/heart_segmentation

Crop mask, first match wins:
  1. --crop-mask <nifti>            any binary mask ({case_id} placeholder allowed, e.g. to reuse
                                    outputs/valve_calcium/{case_id}/total_fast/heart.nii.gz)
  2. <out-root>/<case>/total_fast/heart.nii.gz   from an earlier run
  3. TotalSegmentator `total --fast -rs heart`    written to 2. (skip with --no-totalseg)
  --no-crop segments the whole volume (slow; refused above --max-memory-fraction of RAM unless --allow-large).
Outputs land in <out-root>/<case>/ (see heart_segmentation.py). Serial by design: one case, one process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heart_segmentation import DEFAULT_WEIGHTS_DIR, SUFFIX, Params, segment_case  # noqa: E402
from nnunet_onnx import DEVICES, SegmentationUnavailable  # noqa: E402

TS_DEFAULT = "/opt/anaconda3/envs/totalseg-mac/bin/TotalSegmentator"


def run_totalseg_heart(ts_bin: str, cta: Path, out: Path, device: str) -> str:
    target = out / "heart.nii.gz"
    if target.is_file():
        return "skipped"
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ts_bin, "-i", str(cta), "-o", str(out), "-ta", "total", "--fast", "-rs", "heart", "-d", device]
    log = out.parent / "ts_total_fast.log"
    with log.open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not target.is_file():
        raise RuntimeError(f"TotalSegmentator heart crop failed (see {log})")
    return f"{time.time() - t0:.0f}s"


def resolve_crop_mask(a, case_id: str, case_dir: Path) -> Path | None:
    if a.no_crop:
        return None
    if a.crop_mask:
        p = Path(str(a.crop_mask).format(case_id=case_id))
        if not p.is_file():
            raise SystemExit(f"--crop-mask not found: {p}")
        return p
    local = case_dir / "total_fast" / "heart.nii.gz"
    if local.is_file():
        return local
    if a.no_totalseg:
        raise SystemExit("No crop mask: pass --crop-mask, drop --no-totalseg, or use --no-crop")
    return None  # caller runs TotalSegmentator into `local`


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cta", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--case-id")
    ap.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    ap.add_argument("--crop-mask", help="binary NIfTI; {case_id} placeholder allowed")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-totalseg", action="store_true", help="never run TotalSegmentator for the crop")
    ap.add_argument("--ts-bin", default=TS_DEFAULT)
    ap.add_argument("--ts-device", default="cpu")
    ap.add_argument("--crop-margin-mm", type=float, default=15.0)
    ap.add_argument("--device", default="cpu", choices=DEVICES)
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tile-step-size", type=float, default=None, help="sliding-window step as a fraction of the patch (default: model's 0.75)")
    ap.add_argument("--no-structures", action="store_true", help="skip the per-structure masks")
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--max-memory-fraction", type=float, default=0.5)
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo a case whose outputs exist")
    a = ap.parse_args(argv)

    case_id = a.case_id or a.cta.name.replace(".nii.gz", "").replace(".nii", "")
    case_dir = a.out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    status: dict = {"case_id": case_id, "cta": str(a.cta.resolve())}
    manifest_path = case_dir / f"{case_id}_{SUFFIX}_manifest.json"
    seg_path = case_dir / f"{case_id}_{SUFFIX}_seg.nii.gz"
    try:
        crop = resolve_crop_mask(a, case_id, case_dir)
        if crop is None and not a.no_crop:
            status["total_fast"] = run_totalseg_heart(a.ts_bin, a.cta, case_dir / "total_fast", a.ts_device)
            crop = case_dir / "total_fast" / "heart.nii.gz"
        status["crop_mask"] = str(crop) if crop else None
        if manifest_path.is_file() and seg_path.is_file() and not a.force:
            status["segmentation"] = "skipped"
        else:
            params = Params(
                crop_margin_mm=a.crop_margin_mm,
                device=a.device,
                precision=a.precision,
                num_threads=a.threads,
                tile_step_size=a.tile_step_size,
                write_structures=not a.no_structures,
                write_qc=not a.no_qc,
                max_memory_fraction=a.max_memory_fraction,
                allow_large=a.allow_large,
            )
            t0 = time.time()
            manifest = segment_case(a.cta, case_dir, case_id, params, crop_mask_path=crop, weights_dir=a.weights_dir)
            status["segmentation"] = f"{time.time() - t0:.0f}s"
            status["structures_present"] = len(manifest["structures_present"])
            status["structures_truncated_by_scan"] = manifest["structures_truncated_by_scan"]
        status["status"] = "ok"
    except SegmentationUnavailable as e:
        status.update(status="failed", error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - recorded in case_status.json for the batch
        status.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (case_dir / "case_status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status))
    return 0 if status.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
