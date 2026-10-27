#!/usr/bin/env python3
"""Convert the confirmed calcified-emboli cohort (DICOMDIR-style exports) to NIfTI, serially, straight from the drive.

Per case picks: NCCT = thin (<= 0.7 mm) ORIGINAL/PRIMARY/AXIAL series without contrast, most slices;
                CTA  = ORIGINAL/PRIMARY/AXIAL series whose description contains 'CTA' (if any);
                NCCT_thick = 5 mm NC head (optional, for reader comparison).
Writes outputs/<out>/<case>/raw/sub-cce<PIN>_acq-{NCCT,CTA}_ct.nii.gz + json via dcm2niix, and a manifest.
One sequential pass per series; nothing else is copied from the drive.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import pydicom

DCM2NIIX = "/opt/anaconda3/bin/dcm2niix"
SKIP = ("IHE_PDI", "JRE", "PLUGINS", "HELP", "REPORT", "XTR_CONT")


def scan_series(case_dir: Path):
    out = []
    for d, dirs, files in os.walk(case_dir):
        if any(s in d for s in SKIP):
            continue
        files = [f for f in files if not f.startswith("._")]
        if len(files) < 5:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(d, files[0]), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        it = [str(x).upper() for x in getattr(ds, "ImageType", [])]
        out.append({"dir": d, "n": len(files), "desc": str(getattr(ds, "SeriesDescription", "")), "thk": float(getattr(ds, "SliceThickness", 0) or 0),
                    "contrast": str(getattr(ds, "ContrastBolusAgent", "") or ""), "original": "ORIGINAL" in it and "PRIMARY" in it and "AXIAL" in it,
                    "kvp": str(getattr(ds, "KVP", "")), "series_uid": str(getattr(ds, "SeriesInstanceUID", ""))})
    return out


def pick(series):
    orig = [s for s in series if s["original"]]
    ncct = [s for s in orig if not s["contrast"] and s["thk"] <= 0.7 and "CTA" not in s["desc"].upper()]
    cta = [s for s in orig if "CTA" in s["desc"].upper()]
    thick = [s for s in orig if not s["contrast"] and 2.0 <= s["thk"] <= 5.5 and "NC" in s["desc"].upper().replace(" ", "")]
    choice = {}
    if ncct:
        choice["NCCT"] = max(ncct, key=lambda s: s["n"])
    if cta:
        choice["CTA"] = max(cta, key=lambda s: s["n"])
    if thick:
        choice["NCCTthick"] = max(thick, key=lambda s: s["n"])
    return choice


def convert(series_dir: str, out_dir: Path, stem: str):
    work = Path(tempfile.mkdtemp(prefix="dcm2niix_"))
    t0 = time.time()
    r = subprocess.run([DCM2NIIX, "-z", "y", "-f", stem, "-o", str(work), "-w", "1", series_dir], capture_output=True, text=True)
    produced = sorted(work.glob("*.nii.gz"))
    if r.returncode != 0 or not produced:
        shutil.rmtree(work, ignore_errors=True)
        return None, r.stdout[-800:] + r.stderr[-400:]
    # Tilted GE head CTs: dcm2niix writes the raw (sheared) volume AND a tilt-corrected / slice-equalised one
    # ("*_Tilt_*", "*_Tilt_Eq_*"). Prefer the corrected one; otherwise the largest volume (splits, localizers).
    tilt = [p for p in produced if "_Tilt" in p.name]
    best = max(tilt, key=lambda p: p.stat().st_size) if tilt else max(produced, key=lambda p: p.stat().st_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{stem}.nii.gz"; shutil.move(str(best), final)
    jsons = sorted(work.glob("*.json"))
    js = next((j for j in jsons if j.stem == best.name.replace(".nii.gz", "")), jsons[0] if jsons else None)
    if js is not None:
        shutil.move(str(js), out_dir / f"{stem}.json")
    extra = [p.name for p in produced if p != best]
    shutil.rmtree(work, ignore_errors=True)
    return final, f"{time.time() - t0:.0f}s kept={best.name}" + (f" discarded={extra}" if extra else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("/Volumes/SANDISK USB/Calcified emboli"))
    ap.add_argument("--out-root", type=Path, required=True); ap.add_argument("--limit", type=int)
    ap.add_argument("--prefix", default="sub-cce", help="case id prefix, e.g. sub-neg for a negative-control folder")
    ap.add_argument("--cases", nargs="*", help="only these PIN folders")
    a = ap.parse_args(argv)
    cases = sorted([p for p in a.source.iterdir() if p.is_dir() and p.name[0].isdigit()], key=lambda p: int(p.name))
    if a.cases:
        cases = [c for c in cases if c.name in set(a.cases)]
    cases = cases[: a.limit]
    manifest = []
    for case in cases:
        case_id = f"{a.prefix}{case.name}"
        series = scan_series(case); choice = pick(series)
        rec = {"case_id": case_id, "pin_folder": case.name, "series_found": len(series), "converted": {}}
        for kind, s in choice.items():
            stem = f"{case_id}_acq-{kind}_ct"
            dest = a.out_root / case_id / "raw"
            if (dest / f"{stem}.nii.gz").is_file():
                rec["converted"][kind] = {"path": str(dest / f"{stem}.nii.gz"), "status": "exists"}; continue
            final, info = convert(s["dir"], dest, stem)
            rec["converted"][kind] = {"path": str(final) if final else None, "status": "ok" if final else "failed", "info": info,
                                      "desc": s["desc"], "slices": s["n"], "thk_mm": s["thk"], "kvp": s["kvp"], "contrast": s["contrast"]}
            print(case_id, kind, rec["converted"][kind]["status"], s["desc"], s["n"], info[:120], flush=True)
        manifest.append(rec)
        a.out_root.mkdir(parents=True, exist_ok=True)
        (a.out_root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
