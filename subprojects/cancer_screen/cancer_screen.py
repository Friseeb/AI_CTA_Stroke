#!/usr/bin/env python3
"""Opportunistic incidental-finding / cancer screen on head-neck-chest CTA.  v0, research only.

Consumes TotalSegmentator outputs and converts them into guideline-style flags:

  lung      : `lung_nodules` task -> per-nodule size (max axial diameter, equivalent diameter), volume,
              mean HU, lobe, Fleischner 2017 bin (<6 mm / 6-8 mm / >8 mm; solid vs sub-solid by HU)
  thyroid   : `total` thyroid_gland -> volume, left/right asymmetry, HU heterogeneity, hypodense/hyperdense
              focal candidates >= `thyroid_nodule_min_mm` (ACR white paper: report >=1 cm <35 y, >=1.5 cm >=35 y)
  glands    : `head_glands_cavities` parotid / submandibular left-right volume asymmetry (mass surrogate)
  pharynx   : `head_glands_cavities` oro/naso/hypopharynx airway volume (narrowing / asymmetry surrogate)
  lymph     : NOT segmented by any installed model; placeholder that records the mediastinal / cervical
              search regions (between carotids/IJVs; mediastinum from `trunk_cavities` if present) so a
              nodal model can be dropped in later.  No nodal flags are produced in v0.

Outputs per case: findings CSV (long format), per-lesion table, Slicer fiducials for nodules, manifest.
Thresholds are configurable; nothing here is a diagnosis.
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

__version__ = "0.1.0"

LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
PAIRED_GLANDS = [("parotid_gland_left", "parotid_gland_right"), ("submandibular_gland_left", "submandibular_gland_right")]
PHARYNX = ["nasopharynx", "oropharynx", "hypopharynx"]


@dataclass
class Params:
    nodule_min_mm: float = 3.0
    nodule_solid_hu: float = -300.0          # mean HU above this = solid; below = ground-glass / part-solid
    nodule_inside_lung_min: float = 0.5      # fraction of the component inside the (2 mm dilated) lung mask
    thyroid_nodule_min_mm: float = 8.0
    thyroid_hypo_delta_hu: float = -60.0     # focal region this much below gland median (contrast CT: nodules often hypo)
    thyroid_hyper_delta_hu: float = 80.0
    thyroid_calcification_hu: float = 450.0  # above this a hyperdense focus is reported as calcification, not a nodule
    thyroid_core_erode_mm: float = 2.5       # ignore the gland boundary (partial volume with strap muscles / vessels / cartilage)
    thyroid_cartilage_exclude_mm: float = 2.0
    thyroid_smooth_mm: float = 1.5
    gland_asymmetry_ratio: float = 1.5       # larger/smaller volume ratio to flag


def load(path: Path, ref: nib.Nifti1Image) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = nib.load(str(path))
    if img.shape != ref.shape:
        raise ValueError(f"{path.name}: grid differs from CTA")
    return np.asarray(img.dataobj) > 0


def ball(spacing, r_mm):
    r = [max(int(np.ceil(r_mm / s)), 0) for s in spacing]
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return sum((g[i] * spacing[i]) ** 2 for i in range(3)) <= r_mm ** 2


def components(mask, spacing):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    out = []
    for i in range(1, n + 1):
        comp = lab == i; ii = np.where(comp)
        vol = comp.sum() * float(np.prod(spacing))
        # max axial diameter = largest in-plane extent over slices (approximation of RECIST long axis)
        ax = 0.0
        for k in np.unique(ii[2]):
            s = comp[:, :, k]; jj = np.where(s)
            ax = max(ax, float(np.hypot((jj[0].max() - jj[0].min() + 1) * spacing[0], (jj[1].max() - jj[1].min() + 1) * spacing[1])))
        out.append({"label": i, "mask": comp, "volume_mm3": vol, "equiv_diameter_mm": (6 * vol / np.pi) ** (1 / 3), "max_axial_mm": ax,
                    "centroid_ijk": [float(x) for x in ndimage.center_of_mass(comp)]})
    return out


def fleischner_bin(d_mm: float) -> str:
    return "<6mm" if d_mm < 6 else "6-8mm" if d_mm <= 8 else ">8mm"


def lung_findings(image, nodules, lobes, spacing, p: Params, case_id):
    rows, lesions = [], []
    if nodules is None:
        rows.append({"region": "lung", "feature": "nodule_task_available", "value": 0}); return rows, lesions
    comps = components(nodules, spacing)
    lung_any = np.zeros(nodules.shape, bool)
    for m in lobes.values():
        if m is not None:
            lung_any |= m
    lung_any = ndimage.binary_dilation(lung_any, structure=ball(spacing, 2.0)) if lung_any.any() else lung_any
    n_flag = 0; n_extra = 0
    for c in comps:
        d = max(c["max_axial_mm"], c["equiv_diameter_mm"])
        if d < p.nodule_min_mm:
            continue
        inside = float((c["mask"] & lung_any).sum() / c["mask"].sum()) if lung_any.any() else 1.0
        if inside < p.nodule_inside_lung_min:
            n_extra += 1          # mediastinum / chest wall / heart labelled by the nodule model: not a lung nodule
            continue
        hu = float(image[c["mask"]].mean())
        lobe = next((name for name, m in lobes.items() if m is not None and (m & c["mask"]).sum() > 0.3 * c["mask"].sum()), "unassigned")
        n_flag += 1
        lesions.append({"case_id": case_id, "organ": "lung", "lesion_id": n_flag, "max_axial_mm": c["max_axial_mm"], "equiv_diameter_mm": c["equiv_diameter_mm"],
                        "volume_mm3": c["volume_mm3"], "mean_hu": hu, "attenuation": "solid" if hu >= p.nodule_solid_hu else "subsolid",
                        "location": lobe, "fleischner_bin": fleischner_bin(d), "centroid_ijk": c["centroid_ijk"]})
    rows += [{"region": "lung", "feature": "nodule_task_available", "value": 1}, {"region": "lung", "feature": "nodules_total", "value": len(comps)},
             {"region": "lung", "feature": f"nodules_ge_{p.nodule_min_mm:g}mm", "value": n_flag},
             {"region": "lung", "feature": "extrapulmonary_components_rejected", "value": n_extra},
             {"region": "lung", "feature": "nodules_6_8mm", "value": sum(l["fleischner_bin"] == "6-8mm" for l in lesions)},
             {"region": "lung", "feature": "nodules_gt_8mm", "value": sum(l["fleischner_bin"] == ">8mm" for l in lesions)},
             {"region": "lung", "feature": "largest_nodule_mm", "value": max([l["max_axial_mm"] for l in lesions], default=0.0)}]
    return rows, lesions


def thyroid_findings(image, thyroid, spacing, p: Params, case_id, cartilage=None, blood_ref_hu=None):
    rows, lesions = [], []
    if thyroid is None or not thyroid.any():
        rows.append({"region": "thyroid", "feature": "gland_found", "value": 0}); return rows, lesions
    vox = float(np.prod(spacing)); vals = image[thyroid]
    med = float(np.median(vals))
    rows += [{"region": "thyroid", "feature": "gland_found", "value": 1}, {"region": "thyroid", "feature": "volume_ml", "value": thyroid.sum() * vox / 1000},
             {"region": "thyroid", "feature": "median_hu", "value": med}, {"region": "thyroid", "feature": "iqr_hu", "value": float(np.subtract(*np.percentile(vals, [75, 25])))}]
    # left/right split at the gland's x-centroid (RAS x ~ array axis 0 in these volumes)
    cx = ndimage.center_of_mass(thyroid)[0]; g = np.mgrid[:thyroid.shape[0], :thyroid.shape[1], :thyroid.shape[2]][0]
    vl, vr = (thyroid & (g > cx)).sum() * vox, (thyroid & (g <= cx)).sum() * vox
    rows.append({"region": "thyroid", "feature": "lobe_volume_ratio", "value": max(vl, vr) / max(min(vl, vr), 1)})
    idx = np.where(thyroid); pad = 4
    sl = tuple(slice(max(idx[i].min() - pad, 0), idx[i].max() + pad + 1) for i in range(3))
    sm = ndimage.gaussian_filter(image[sl], sigma=[p.thyroid_smooth_mm / s for s in spacing]); tm = thyroid[sl]
    core = ndimage.binary_erosion(tm, structure=ball(spacing, p.thyroid_core_erode_mm))
    if cartilage is not None and cartilage.any():
        core &= ~ndimage.binary_dilation(cartilage[sl], structure=ball(spacing, p.thyroid_cartilage_exclude_mm))
    raw = image[sl]
    hyper = core & (sm >= med + p.thyroid_hyper_delta_hu) & (raw < p.thyroid_calcification_hu)
    if blood_ref_hu is not None:
        hyper &= raw < blood_ref_hu - 50          # enhancing vessels inside the gland mask are not nodules
    calc = core & (raw >= p.thyroid_calcification_hu)
    n_calc = 0
    for kind, cand in (("hypodense", core & (sm <= med + p.thyroid_hypo_delta_hu)), ("hyperenhancing", hyper), ("calcification", calc)):
        for c in components(cand, spacing):
            d = max(c["max_axial_mm"], c["equiv_diameter_mm"])
            if kind == "calcification":
                if d >= 3.0:
                    n_calc += 1
                continue
            if d < p.thyroid_nodule_min_mm:
                continue
            lesions.append({"case_id": case_id, "organ": "thyroid", "lesion_id": len(lesions) + 1, "max_axial_mm": c["max_axial_mm"], "equiv_diameter_mm": c["equiv_diameter_mm"],
                            "volume_mm3": c["volume_mm3"], "mean_hu": float(raw[c["mask"]].mean()), "attenuation": kind, "location": "thyroid",
                            "fleischner_bin": "", "centroid_ijk": [c["centroid_ijk"][k] + sl[k].start for k in range(3)]})
    rows.append({"region": "thyroid", "feature": "calcifications_ge_3mm", "value": n_calc})
    rows.append({"region": "thyroid", "feature": f"focal_candidates_ge_{p.thyroid_nodule_min_mm:g}mm", "value": len(lesions)})
    rows.append({"region": "thyroid", "feature": "largest_focal_mm", "value": max([l["max_axial_mm"] for l in lesions], default=0.0)})
    return rows, lesions


def head_findings(masks, spacing, p: Params):
    rows = []; vox = float(np.prod(spacing))
    for l, r in PAIRED_GLANDS:
        ml, mr = masks.get(l), masks.get(r)
        if ml is None or mr is None:
            continue
        vl, vr = ml.sum() * vox / 1000, mr.sum() * vox / 1000
        base = l.replace("_left", "")
        rows += [{"region": "head", "feature": f"{base}_left_ml", "value": vl}, {"region": "head", "feature": f"{base}_right_ml", "value": vr},
                 {"region": "head", "feature": f"{base}_asymmetry_ratio", "value": max(vl, vr) / max(min(vl, vr), 1e-3)},
                 {"region": "head", "feature": f"{base}_asymmetry_flag", "value": float(max(vl, vr) / max(min(vl, vr), 1e-3) >= p.gland_asymmetry_ratio)}]
    for ph in PHARYNX:
        m = masks.get(ph)
        if m is not None:
            rows.append({"region": "head", "feature": f"{ph}_air_ml", "value": m.sum() * vox / 1000})
    return rows


def qc_montage(image, lesions, spacing, out_png: Path, case_id: str) -> None:
    if not lesions:
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = sorted(lesions, key=lambda l: -l["max_axial_mm"])[:9]
    cols = 3; rows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows)); axes = np.atleast_1d(axes).ravel()
    for ax, l in zip(axes, top):
        ci, cj, ck = [int(round(x)) for x in l["centroid_ijk"]]
        half = int(max(30.0, l["max_axial_mm"]) / spacing[0])
        i0, i1 = max(ci - half, 0), ci + half; j0, j1 = max(cj - half, 0), cj + half
        win = (-1000, 300) if l["organ"] == "lung" else (-100, 400)
        im = np.clip((image[i0:i1, j0:j1, ck].T[::-1] - win[0]) / (win[1] - win[0]), 0, 1)
        ax.imshow(im, cmap="gray"); ax.plot([ci - i0], [im.shape[0] - (cj - j0)], "r+", ms=14, mew=1.5)
        ax.set_title(f"{l['organ']} #{l['lesion_id']} {l['max_axial_mm']:.0f} mm {l['attenuation']} {l['mean_hu']:.0f} HU", fontsize=8); ax.axis("off")
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle(f"{case_id}: incidental-finding candidates (reader review)", fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def write_fiducials(lesions, affine, path: Path):
    pts = []
    for l in lesions:
        ras = (affine @ np.array([*l["centroid_ijk"], 1.0]))[:3]
        pts.append({"id": f"{l['organ']}-{l['lesion_id']}", "label": f"{l['organ']} {l['max_axial_mm']:.1f}mm {l['attenuation']} {l['location']}",
                    "position": [float(x) for x in ras], "orientation": [-1, 0, 0, 0, -1, 0, 0, 0, 1], "selected": True, "locked": False, "visibility": True})
    doc = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
           "markups": [{"type": "Fiducial", "coordinateSystem": "RAS", "coordinateUnits": "mm", "locked": False, "labelFormat": "%N-%d", "controlPoints": pts,
                        "display": {"color": [1.0, 0.8, 0.1], "glyphScale": 2.0, "textScale": 2.5}}]}
    path.write_text(json.dumps(doc, indent=1))


def run_case(cta: Path, ts_root: Path, out_dir: Path, p: Params, case_id: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(cta)); spacing = tuple(float(s) for s in ref.header.get_zooms()[:3]); image = np.asarray(ref.dataobj).astype(np.float32)
    total = ts_root / "total"; nod = ts_root / "lung_nodules"; glands = ts_root / "head_glands"  # + headneck_vessels (optional)
    lobes = {n: load(total / f"{n}.nii.gz", ref) for n in LOBES}
    thyroid = load(total / "thyroid_gland.nii.gz", ref)
    nodules = load(nod / "lung_nodules.nii.gz", ref)
    head = {n: load(glands / f"{n}.nii.gz", ref) for n in [g for pair in PAIRED_GLANDS for g in pair] + PHARYNX}
    hnv = ts_root / "headneck_vessels"
    cart = None
    for n in ("thyroid_cartilage", "cricoid_cartilage"):
        m = load(hnv / f"{n}.nii.gz", ref)
        if m is not None:
            cart = m if cart is None else (cart | m)
    blood_ref = None
    for n in ("internal_carotid_artery_left", "internal_carotid_artery_right"):
        m = load(hnv / f"{n}.nii.gz", ref)
        if m is not None and m.any():
            v = float(np.median(image[m]))
            if v >= 150.0:                    # an enhanced ICA is >= ~250 HU; lower means the mask missed the vessel
                blood_ref = v if blood_ref is None else min(blood_ref, v)
    rows, lesions = [], []
    r, l = lung_findings(image, nodules, lobes, spacing, p, case_id); rows += r; lesions += l
    r, l = thyroid_findings(image, thyroid, spacing, p, case_id, cartilage=cart, blood_ref_hu=blood_ref); rows += r; lesions += l
    rows.append({"region": "thyroid", "feature": "carotid_blood_reference_hu", "value": blood_ref if blood_ref is not None else float("nan")})
    rows += head_findings(head, spacing, p)
    rows.append({"region": "lymph_nodes", "feature": "model_available", "value": 0})
    df = pd.DataFrame(rows); df.insert(0, "case_id", case_id); df["software_version"] = __version__
    df.to_csv(out_dir / f"{case_id}_cancer_screen_features.csv", index=False)
    les = pd.DataFrame(lesions)
    les.to_csv(out_dir / f"{case_id}_cancer_screen_lesions.csv", index=False)
    qc_montage(image, lesions, spacing, out_dir / f"{case_id}_cancer_screen_qc.png", case_id)
    write_fiducials(lesions, ref.affine, out_dir / f"{case_id}_cancer_screen_lesions.mrk.json")
    (out_dir / f"{case_id}_cancer_screen_manifest.json").write_text(json.dumps({"case_id": case_id, "cta": str(cta), "ts_root": str(ts_root), "params": asdict(p),
        "inputs_present": {"lung_nodules": nodules is not None, "thyroid": thyroid is not None, "head_glands": any(v is not None for v in head.values())},
        "software_version": __version__, "intended_use": "research; incidental-finding candidates for reader review, not a diagnosis"}, indent=2))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cta", type=Path, required=True); ap.add_argument("--ts-root", type=Path, required=True, help="folder with total/, lung_nodules/, head_glands/")
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--case-id")
    for f, v in asdict(Params()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args(argv); p = Params(**{f: getattr(a, f) for f in asdict(Params())})
    case_id = a.case_id or a.cta.name.replace(".nii.gz", "")
    df = run_case(a.cta, a.ts_root, a.out_dir, p, case_id)
    print(df[["region", "feature", "value"]].round(2).to_string(index=False)); return 0


if __name__ == "__main__":
    sys.exit(main())
