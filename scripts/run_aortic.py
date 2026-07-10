#!/usr/bin/env python
"""Single-case aorta CTA pipeline for the dashboard runner.

Two stages, each in its own conda env (the heavy deps do not co-exist):
  1. VISTA3D aorta segmentation (label 6)  -> nv-segment-ct env
  2. aorta radiomics stages (calcium/fat/wall/protrusions/wall-thickness) ->
     aorta-cta-radiomics env  (via aorta_cta_radiomics/scripts/run_single_case.py)

Then writes ``<out>/report.json`` and copies representative task masks to
``<out>/<case>_<task>.nii.gz`` so the Streamlit Results Viewer can render them.

This orchestrator itself only needs the stdlib (+ pandas for the report), so it
can run from any env; it spawns the correct env python for each stage.

RESEARCH PROTOTYPE - NOT FOR CLINICAL DIAGNOSIS.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-stage env pythons (overridable). These are the envs that actually carry
# the stage's dependencies on this installation.
DEFAULT_NV_PYTHON = Path("/opt/anaconda3/envs/nv-segment-ct/bin/python")
DEFAULT_AORTA_PYTHON = Path("/opt/anaconda3/envs/aorta-cta-radiomics/bin/python")
DEFAULT_MODEL_DIR = REPO_ROOT / "external" / "nv_segment_ct"

# Representative mask per UI "task" (glob suffix under masks/<case>/).
TASK_MASKS = {
    "calcium": "{case}_calcification_aorta_wall_dynamic_seed500HU.nii.gz",
    "fat": "{case}_periaortic_fat.nii.gz",
    "wall": "{case}_aorta_wall_band.nii.gz",
}


def _run(cmd: list[str]) -> int:
    """Stream a subprocess line-by-line to stdout; return its exit code."""
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    proc.wait()
    return proc.returncode


def _first_python(*candidates: Path) -> str:
    for c in candidates:
        if Path(c).exists():
            return str(c)
    return sys.executable  # last resort: current interpreter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CTA NIfTI (.nii.gz).")
    ap.add_argument("--out", required=True, help="Output directory for this case.")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tasks", nargs="*", default=["calcium", "fat", "wall"])
    ap.add_argument("--nv-python", type=Path, default=DEFAULT_NV_PYTHON)
    ap.add_argument("--aorta-python", type=Path, default=DEFAULT_AORTA_PYTHON)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--config", default=None, help="Optional aorta pipeline YAML config.")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    case = args.case_id
    out.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        print(f"[aortic] ERROR: input not found: {inp}", flush=True)
        return 2

    nv_python = _first_python(args.nv_python)
    aorta_python = _first_python(args.aorta_python)
    # VISTA3D auto-detects CUDA/MPS/CPU; map the UI's gpu/cpu onto its vocabulary.
    vista_device = "auto" if args.device in ("gpu", "auto", "") else args.device

    # ---- Stage 1: VISTA3D aorta segmentation (label 6) --------------------
    mask = out / "vista_aorta" / f"{case}_aorta6.nii.gz"
    mask.parent.mkdir(parents=True, exist_ok=True)
    print(f"--- Stage 1/2: VISTA3D aorta segmentation (label 6)  [{Path(nv_python).parent.parent.name}] ---", flush=True)
    rc = _run([
        nv_python, str(REPO_ROOT / "scripts" / "run_nv_segment_ct_laa.py"),
        "--input", str(inp), "--output", str(mask),
        "--label-id", "6", "--model-dir", str(args.model_dir),
        "--device", vista_device,
    ])
    if rc != 0 or not mask.exists():
        print(f"[aortic] ERROR: VISTA3D segmentation failed (rc={rc}); no aorta mask produced.", flush=True)
        return rc or 3

    # ---- Stage 2: aorta radiomics stages ---------------------------------
    print(f"--- Stage 2/2: aorta radiomics stages  [{Path(aorta_python).parent.parent.name}] ---", flush=True)
    cmd = [
        aorta_python, str(REPO_ROOT / "aorta_cta_radiomics" / "scripts" / "run_single_case.py"),
        "--image", str(inp), "--aorta-mask", str(mask),
        "--case-id", case, "--outdir", str(out),
    ]
    if args.config:
        cmd += ["--config", str(args.config)]
    rc = _run(cmd)
    if rc != 0:
        print(f"[aortic] ERROR: aorta radiomics stages failed (rc={rc}).", flush=True)
        return rc

    # ---- Surface results for the dashboard viewer ------------------------
    masks_dir = out / "masks" / case
    for task in args.tasks:
        suffix = TASK_MASKS.get(task.lower())
        if not suffix:
            continue
        src = masks_dir / suffix.format(case=case)
        if src.exists():
            dst = out / f"{case}_{task.lower()}.nii.gz"
            try:
                shutil.copy2(src, dst)
                print(f"[aortic] surfaced {task} mask -> {dst.name}", flush=True)
            except OSError as exc:
                print(f"[aortic] WARN: could not copy {task} mask: {exc}", flush=True)

    report: dict = {"case_id": case, "tasks": [t.lower() for t in args.tasks],
                    "aorta_mask": str(mask), "masks_dir": str(masks_dir)}
    feats = out / "features" / "case_level_features.csv"
    if feats.exists():
        try:
            import pandas as pd
            row = pd.read_csv(feats)
            row = row[row.get("case_id", row.iloc[:, 0]).astype(str).str.contains(case)] if len(row) else row
            if len(row):
                r = row.iloc[0]
                for col in ["aorta_volume_ml", "aortic_length_cm",
                            "aorta_wall_dynamic__calcification__calcium_volume__thr_dynamic_lumen_referenced_seed500HU",
                            "aortic_wall__wall_thickness__wall_mean_mm"]:
                    if col in r.index:
                        key = col.split("__")[-1] if "__" in col else col
                        report[key] = round(float(r[col]), 3) if str(r[col]).replace('.', '', 1).replace('-', '', 1).isdigit() else r[col]
            report["n_feature_columns"] = int(row.shape[1])
        except Exception as exc:  # noqa: BLE001
            report["features_note"] = f"could not parse case_level_features.csv: {exc}"
    else:
        report["features_note"] = "case_level_features.csv not found"

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[aortic] wrote {out / 'report.json'}", flush=True)
    print("[aortic] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
