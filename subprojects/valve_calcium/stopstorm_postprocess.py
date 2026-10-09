#!/usr/bin/env python3
"""Map a STOPSTORM (Zenodo 13752275, model 603) prediction on a heart crop back to the full CTA grid,
write one NIfTI per label, and compare Valve_Mitral with the geometric mitral band.

  python stopstorm_postprocess.py --case-dir outputs/valve_calcium/<case_id> --cta <full CTA>
"""
import argparse, json, sys
from pathlib import Path
import nibabel as nib, numpy as np
from scipy import ndimage

LABELS = {1: "Valve_Aortic", 2: "Valve_Mitral", 3: "Valve_Pulmonic", 4: "Valve_Tricuspid", 5: "A_LAD", 6: "A_LCX", 7: "A_LM", 8: "A_RCA"}


STOPSTORM_RESULTS = Path(__file__).resolve().parents[2] / "external" / "stopstorm" / "nnUNet_results"
NNUNET_PREDICT = "/opt/anaconda3/envs/totalseg-mac/bin/nnUNetv2_predict"


def make_heart_crop(cta: Path, heart_mask: Path, sdir: Path, margin_mm: float = 25.0) -> Path:
    """Crop the CTA to the TotalSegmentator heart bbox (+margin) as nnU-Net input; records the offset."""
    img = nib.load(str(cta)); heart = np.asarray(nib.load(str(heart_mask)).dataobj) > 0
    sp = img.header.get_zooms()[:3]; idx = np.where(heart)
    pad = [int(margin_mm / s) for s in sp]
    sl = tuple(slice(max(idx[i].min() - pad[i], 0), min(idx[i].max() + pad[i] + 1, heart.shape[i])) for i in range(3))
    data = np.asarray(img.dataobj)[sl].astype(np.int16)
    aff = img.affine.copy(); aff[:3, 3] = (img.affine @ np.array([sl[0].start, sl[1].start, sl[2].start, 1]))[:3]
    inp = sdir / "input"; inp.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data, aff); out.header.set_zooms(sp)
    nib.save(out, str(inp / "case_0000.nii.gz"))
    (sdir / "crop.json").write_text(json.dumps({"crop_start_ijk": [int(s.start) for s in sl], "crop_shape": [int(x) for x in data.shape],
                                                "source": str(cta), "margin_mm": margin_mm}, indent=2))
    return inp


def run_stopstorm(sdir: Path, device: str = "cpu", predict_bin: str = NNUNET_PREDICT, results: Path = STOPSTORM_RESULTS) -> Path:
    import os, subprocess
    out = sdir / "output"; out.mkdir(exist_ok=True)
    env = dict(os.environ, nnUNet_results=str(results), nnUNet_raw=str(sdir / "_raw"), nnUNet_preprocessed=str(sdir / "_pre"))
    cmd = [predict_bin, "-i", str(sdir / "input"), "-o", str(out), "-d", "603", "-c", "3d_fullres", "-f", "all",
           "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "-device", device, "-npp", "2", "-nps", "2"]
    with (sdir / "predict.log").open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    preds = list(out.glob("*.nii.gz"))
    if r.returncode != 0 or not preds:
        raise RuntimeError(f"STOPSTORM prediction failed (see {sdir / 'predict.log'})")
    return preds[0]


def postprocess(case_dir: Path, cta_path: Path) -> dict:
    sdir = case_dir / "stopstorm"
    crop = json.loads((sdir / "crop.json").read_text())
    pred_path = next(sdir.glob("output/*.nii.gz"))
    pred = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.uint8)
    cta = nib.load(str(cta_path)); sp = cta.header.get_zooms()[:3]; vox = float(np.prod(sp))
    full = np.zeros(cta.shape, np.uint8)
    s = crop["crop_start_ijk"]; sh = pred.shape
    full[s[0]:s[0] + sh[0], s[1]:s[1] + sh[1], s[2]:s[2] + sh[2]] = pred
    case_id = case_dir.name
    out = sdir / "full_grid"; out.mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(full, cta.affine), str(out / f"{case_id}_stopstorm_labels.nii.gz"))
    report = {"case_id": case_id, "prediction": str(pred_path), "volumes_ml": {}}
    for lid, name in LABELS.items():
        m = full == lid
        report["volumes_ml"][name] = m.sum() * vox / 1000.0
        nib.save(nib.Nifti1Image(m.astype(np.uint8), cta.affine), str(out / f"{case_id}_{name}.nii.gz"))
    # compare with geometric ROIs
    for name, lid, roi_file in (("mitral", 2, f"analysis/{case_id}_mv_roi.nii.gz"),):
        roi_p = case_dir / roi_file
        if not roi_p.is_file():
            continue
        roi = np.asarray(nib.load(str(roi_p)).dataobj) > 0; m = full == lid
        inter = (roi & m).sum()
        cm_m = np.array(ndimage.center_of_mass(m)) if m.any() else None
        cm_r = np.array(ndimage.center_of_mass(roi)) if roi.any() else None
        dist = float(np.sqrt((((cm_m - cm_r) * np.array(sp)) ** 2).sum())) if cm_m is not None and cm_r is not None else None
        report[f"{name}_vs_geometric"] = {
            "stopstorm_ml": m.sum() * vox / 1000, "geometric_roi_ml": roi.sum() * vox / 1000,
            "dice": 2 * inter / max(m.sum() + roi.sum(), 1), "fraction_of_stopstorm_inside_roi": inter / max(m.sum(), 1),
            "fraction_of_roi_inside_stopstorm": inter / max(roi.sum(), 1), "centroid_distance_mm": dist,
            "stopstorm_z_range": [int(x) for x in np.flatnonzero(m.any(axis=(0, 1)))[[0, -1]]] if m.any() else None,
        }
    (sdir / "stopstorm_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--case-dir", type=Path, required=True); ap.add_argument("--cta", type=Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(postprocess(a.case_dir, a.cta), indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
