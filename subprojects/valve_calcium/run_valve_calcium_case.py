#!/usr/bin/env python3
"""Single-case driver: TotalSegmentator chain (skip-if-done) + valve calcium analysis.

  python run_valve_calcium_case.py --cta <sub-N_acq-CTA_ct.nii.gz> --out-root outputs/valve_calcium

Steps (each skipped if its output folder is complete):
  1. total --fast, roi_subset heart/aorta/IVC   -> crop masks shared by the next tasks
  2. heartchambers_highres                       (LA, LV, myocardium, aorta)
  3. aortic_sinuses                              (LVOT + 3 cusps)
  4. coronary_arteries                           (excluded from both ROIs; --no-coronary to skip)
  5. STOPSTORM model 603 (Zenodo 13752275) on the heart crop -> Valve_Mitral (--no-stopstorm to skip)
  6. valve_calcium.run_case (aortic valve from TS sinuses; mitral: geometric band + STOPSTORM label)
TotalSegmentator binary / device via --ts-bin / --device.  Serial by design (one case, one process).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from valve_calcium import Params, run_case, CHAMBER_FILES, SINUS_FILES  # noqa: E402
from stopstorm_postprocess import make_heart_crop, run_stopstorm, postprocess  # noqa: E402

TS_DEFAULT = "/opt/anaconda3/envs/totalseg-mac/bin/TotalSegmentator"


def ts(ts_bin, cta, out, task, device, extra=(), required=()):
    if out.is_dir() and all((out / f).is_file() for f in required):
        return "skipped"
    t0 = time.time()
    cmd = [ts_bin, "-i", str(cta), "-o", str(out), "-ta", task, "-d", device, *extra]
    log = out.parent / f"ts_{task}.log"
    with log.open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not all((out / f).is_file() for f in required):
        raise RuntimeError(f"TotalSegmentator {task} failed (see {log})")
    return f"{time.time() - t0:.0f}s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cta", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--case-id")
    ap.add_argument("--ts-bin", default=TS_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-coronary", action="store_true")
    ap.add_argument("--no-stopstorm", action="store_true")
    ap.add_argument("--force-analysis", action="store_true")
    a = ap.parse_args(argv)
    case_id = a.case_id or a.cta.name.replace(".nii.gz", "")
    case_dir = a.out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    status = {"case_id": case_id, "cta": str(a.cta.resolve())}
    crop = case_dir / "total_fast"
    status["total_fast"] = ts(a.ts_bin, a.cta, crop, "total", a.device,
                              ["--fast", "-rs", "heart", "aorta", "inferior_vena_cava"], ["heart.nii.gz", "aorta.nii.gz"])
    status["heartchambers"] = ts(a.ts_bin, a.cta, case_dir / "heartchambers", "heartchambers_highres", a.device,
                                 ["-cp", str(crop)], list(CHAMBER_FILES.values()))
    status["aortic_sinuses"] = ts(a.ts_bin, a.cta, case_dir / "aortic_sinuses", "aortic_sinuses", a.device,
                                  ["-cp", str(crop)], list(SINUS_FILES.values()))
    cor_dir = None
    if not a.no_coronary:
        cor_dir = case_dir / "coronary"
        status["coronary"] = ts(a.ts_bin, a.cta, cor_dir, "coronary_arteries", a.device, ["-cp", str(crop)], ["coronary_arteries.nii.gz"])
    ss_dir = None
    if not a.no_stopstorm:
        sdir = case_dir / "stopstorm"
        ss_dir = sdir / "full_grid"
        if (ss_dir / f"{case_id}_Valve_Mitral.nii.gz").is_file():
            status["stopstorm"] = "skipped"
        else:
            t0 = time.time()
            if not list(sdir.glob("output/*.nii.gz")):
                make_heart_crop(a.cta, crop / "heart.nii.gz", sdir)
                run_stopstorm(sdir, a.device)
            postprocess(case_dir, a.cta)
            status["stopstorm"] = f"{time.time() - t0:.0f}s"
    feat = case_dir / "analysis" / f"{case_id}_valve_calcium_features.csv"
    has_ss = feat.is_file() and "stopstorm" in feat.read_text()
    if feat.is_file() and not a.force_analysis and (a.no_stopstorm or has_ss):
        status["analysis"] = "skipped"
    else:
        t0 = time.time()
        run_case(a.cta, case_dir / "heartchambers", case_dir / "aortic_sinuses", cor_dir, case_dir / "analysis", Params(), case_id, ss_dir)
        status["analysis"] = f"{time.time() - t0:.0f}s"
    status["status"] = "ok"
    (case_dir / "case_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
