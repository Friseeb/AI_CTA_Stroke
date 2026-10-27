#!/usr/bin/env python3
"""Serial, resumable cohort runner: per case, TotalSegmentator (brain/skull, brain_structures, ventricle_parts) on the
NCCT, then search_intracranial_calcium.py in NCCT mode (and CTA mode when a CTA exists, with the CTA's own masks).

  python run_cce_cohort.py --root outputs/calcified_emboli_cohort [--limit N] [--no-cta]
"""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TS = "/opt/anaconda3/envs/totalseg-mac/bin/TotalSegmentator"
PY = sys.executable


def ts(image: Path, out: Path, task: str, extra, required, log: Path) -> str:
    if out.is_dir() and all((out / f).is_file() for f in required):
        return "skipped"
    t0 = time.time()
    with log.open("a") as fh:
        r = subprocess.run([TS, "-i", str(image), "-o", str(out), "-ta", task, "-d", "cpu", *extra], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not all((out / f).is_file() for f in required):
        raise RuntimeError(f"TotalSegmentator {task} failed on {image.name}")
    return f"{time.time() - t0:.0f}s"


def run_modality(case_dir: Path, image: Path, modality: str, status: dict) -> None:
    m = case_dir / modality.lower(); m.mkdir(exist_ok=True); log = m / "ts.log"
    status[f"{modality}_total"] = ts(image, m / "total", "total", ["-rs", "brain", "skull", "vertebrae_C1", "vertebrae_C2", "vertebrae_C3"], ["brain.nii.gz", "skull.nii.gz"], log)
    status[f"{modality}_brain_structures"] = ts(image, m / "brain_structures", "brain_structures", ["-cp", str(m / "total")], ["ventricle.nii.gz"], log)
    status[f"{modality}_ventricle_parts"] = ts(image, m / "ventricle_parts", "ventricle_parts", ["-cp", str(m / "total")], ["third_ventricle.nii.gz"], log)
    t0 = time.time()
    cmd = [PY, str(HERE / "search_intracranial_calcium.py"), "--image", str(image), "--total-dir", str(m / "total"), "--brain-structures-dir", str(m / "brain_structures"),
           "--ventricle-parts-dir", str(m / "ventricle_parts"), "--out-dir", str(m / "search"), "--modality", modality.lower() if modality == "NCCT" else "cta", "--case-id", case_dir.name]
    with (m / "search.log").open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"search failed on {image.name} (see {m / 'search.log'})")
    status[f"{modality}_search"] = f"{time.time() - t0:.0f}s"
    counts = json.loads((m / "search" / f"{case_dir.name}_intracranial_calcium_manifest.json").read_text())["counts"]
    status[f"{modality}_distal_candidates"] = counts.get("distal_candidate", 0); status[f"{modality}_proximal"] = counts.get("proximal_intraluminal", 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--limit", type=int); ap.add_argument("--no-cta", action="store_true")
    ap.add_argument("--prefix", default="sub-cce", help="case folder prefix to process (e.g. sub-neg)")
    a = ap.parse_args(argv)
    cases = sorted([p for p in a.root.iterdir() if p.is_dir() and p.name.startswith(a.prefix)])[: a.limit]
    status_path = a.root / f"cohort_status_{a.prefix.rstrip('-')}.csv"; rows = []
    for case_dir in cases:
        st = {"case_id": case_dir.name, "status": "ok", "error": ""}; t0 = time.time()
        try:
            ncct = case_dir / "raw" / f"{case_dir.name}_acq-NCCT_ct.nii.gz"
            cta = case_dir / "raw" / f"{case_dir.name}_acq-CTA_ct.nii.gz"
            if ncct.is_file():
                run_modality(case_dir, ncct, "NCCT", st)
            else:
                st["NCCT_search"] = "no NCCT"
            if cta.is_file() and not a.no_cta:
                run_modality(case_dir, cta, "CTA", st)
        except Exception as exc:
            st["status"] = "failed"; st["error"] = str(exc)[:300]
        st["seconds"] = f"{time.time() - t0:.0f}"; rows.append(st)
        print(json.dumps(st), flush=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case_id", k))
        with status_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    return 0


if __name__ == "__main__":
    sys.exit(main())
