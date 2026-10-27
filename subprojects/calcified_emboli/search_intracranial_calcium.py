#!/usr/bin/env python3
"""Whole-intracranial calcium search with anatomical classification.  v0, research only.

Instead of looking only where the circle-of-Willis mask is, find every calcium-dense focus inside the brain
mask and classify it by anatomy so that the embolus candidates are what is *left over*:

  skull / dural            within `bone_mm` of the TotalSegmentator skull (and venous sinuses if available)
  arterial_wall            touching the CoW labelmap but eccentric / elongated (atheroma: ICA siphon, M1, BA)
  proximal_intraluminal    touching the CoW labelmap, compact and mostly inside the lumen (the old detector)
  choroid_plexus           inside the ventricles dilated by `ventricle_mm` (lateral trigones, 3rd, 4th)
  pineal_habenula          within `pineal_mm` posterior-superior of the 3rd ventricle
  basal_ganglia            inside lentiform / caudate / thalamus masks (physiological, often bilateral)
  symmetric_bilateral      has a mirror-image counterpart within `mirror_mm` across the midsagittal plane
  distal_candidate         everything else in the parenchyma / subarachnoid space: compact, 1-8 mm -> embolus
                           candidate for reader review (M2-M4, ACA, PCA branches that no mask covers)

Modality presets: --modality cta  (>= 700 HU tier-1, >= 500 tier-2, over an enhanced-vessel floor)
                  --modality ncct (>= 130 HU tier-1, >= 90 tier-2)
Inputs are TotalSegmentator `total` (brain, skull), `brain_structures`, `ventricle_parts` folders and the TopCoW
labelmap; any missing mask simply disables that rule and is recorded in the manifest.
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
TOPCOW_LABELS = {1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA", 8: "R-Pcom",
                 9: "L-Pcom", 10: "Acom", 11: "R-ACA", 12: "L-ACA", 15: "3rd-A2"}


@dataclass
class Params:
    modality: str = "cta"
    tier1_hu: float = 700.0
    tier2_hu: float = 500.0
    brain_erode_mm: float = 1.0
    bone_mm: float = 2.0
    sinus_mm: float = 1.5
    ventricle_mm: float = 2.0
    choroid_inside_min: float = 0.5   # choroid plexus calcium must lie mostly INSIDE the undilated ventricle lumen
    ventricle_max_ml: float = 80.0    # larger ventricle masks are over-segmented (sulci/CSF); then only the inside rule applies
    pineal_mm: float = 12.0
    vessel_touch_mm: float = 1.5
    mirror_mm: float = 6.0
    wall_adjacent_mm: float = 2.5     # a small compact focus this close to a larger wall-calcium component is its partial-volume tail
    d_min_mm: float = 1.0
    d_max_mm: float = 10.0
    compact_min: float = 0.25
    inside_lumen_min: float = 0.3
    smooth_mm: float = 0.0            # optional Gaussian smoothing before thresholding; 0 = off (0.6 mm lost faint 1.7 mm emboli in testing)
    tier2_d_min_mm: float = 1.4       # tier-2 (low-HU) foci must be at least this large to be reported as candidates
    low_conf_d_mm: float = 1.5        # candidates smaller than this AND fainter than low_conf_hu -> distal_candidate_low (kept, not reviewed)
    low_conf_hu: float = 170.0


def preset(p: Params) -> Params:
    if p.modality == "ncct":
        p.tier1_hu, p.tier2_hu = 130.0, 90.0
    return p


def ball(spacing, r_mm):
    r = [max(int(np.ceil(r_mm / s)), 0) for s in spacing]
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return sum((g[i] * spacing[i]) ** 2 for i in range(3)) <= r_mm ** 2


def dil(m, spacing, mm):
    """Dilate by mm via a Euclidean distance transform (much faster than a large ball on 0.3 mm slices)."""
    if mm <= 0 or m is None or not m.any():
        return m
    return ndimage.distance_transform_edt(~m, sampling=spacing) <= mm


def load(path: Path | None, ref: nib.Nifti1Image, labels=None):
    if path is None or not path.is_file():
        return None
    img = nib.load(str(path))
    if img.shape != ref.shape:
        raise ValueError(f"{path.name}: grid differs from the reference image")
    a = np.asarray(img.dataobj)
    return (a > 0) if labels is None else np.isin(a, labels)


def search(image, masks: dict, spacing, p: Params):
    brain = masks["brain"]
    idx = np.where(brain); pad = [int(np.ceil(p.pineal_mm / s)) + 2 for s in spacing]
    sl = tuple(slice(max(idx[i].min() - pad[i], 0), min(idx[i].max() + pad[i] + 1, brain.shape[i])) for i in range(3))
    img = image[sl]; m = {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in masks.items()}
    region = (ndimage.distance_transform_edt(m["brain"], sampling=spacing) > p.brain_erode_mm) if p.brain_erode_mm > 0 else m["brain"]
    bone_d = dil(m.get("skull"), spacing, p.bone_mm)
    sinus_d = dil(m.get("venous_sinuses"), spacing, p.sinus_mm)
    vent = m.get("ventricle"); vent_d = dil(vent, spacing, p.ventricle_mm)
    vent_ml = float(vent.sum() * np.prod(spacing) / 1000.0) if vent is not None else 0.0
    vent_ok = vent is not None and vent_ml <= p.ventricle_max_ml
    third = m.get("third_ventricle")
    pineal_zone = None
    if third is not None and third.any():
        # posterior-superior cap around the 3rd ventricle: dilate, then keep the half that lies posterior to its centroid
        cz = ndimage.center_of_mass(third); pz = dil(third, spacing, p.pineal_mm)
        g = np.mgrid[:pz.shape[0], :pz.shape[1], :pz.shape[2]]
        post = g[1] < cz[1] if masks.get("anterior_is_plus_j", True) else g[1] > cz[1]
        pineal_zone = pz & post
    bg = m.get("basal_ganglia")
    cow = m.get("cow"); cow_bin = cow > 0 if cow is not None else None
    cow_d = dil(cow_bin, spacing, p.vessel_touch_mm) if cow_bin is not None else None
    midline_i = ndimage.center_of_mass(m["brain"])[0]

    img_s = ndimage.gaussian_filter(img, sigma=[p.smooth_mm / s for s in spacing]) if p.smooth_mm > 0 else img
    bright = region & (img_s >= p.tier2_hu)
    lab, n = ndimage.label(bright, structure=np.ones((3, 3, 3)))
    comps = []
    vox = float(np.prod(spacing))
    for i in range(1, n + 1):
        c = lab == i; nv = int(c.sum()); vol = nv * vox; d_eq = (6 * vol / np.pi) ** (1 / 3)
        if d_eq < p.d_min_mm:
            continue
        ii = np.where(c); ext = [(ii[k].max() - ii[k].min() + 1) * spacing[k] for k in range(3)]; dmax = max(ext)
        compact = vol / (np.pi / 6 * dmax ** 3) if dmax > 0 else 0
        hu = img[c]; com = ndimage.center_of_mass(c)
        comps.append({"comp": c, "mask_id": i, "volume_mm3": vol, "equiv_diameter_mm": d_eq, "max_extent_mm": dmax, "compactness": compact,
                      "mean_hu": float(hu.mean()), "max_hu": float(hu.max()), "tier": 1 if hu.max() >= p.tier1_hu else 2, "centroid": com,
                      "side": "L" if com[0] > midline_i else "R"})
    # classification
    def frac(c, mm):
        return float((c & mm).sum() / c.sum()) if mm is not None else 0.0
    for c in comps:
        cm = c["comp"]; cat = None; vessel = ""
        if frac(cm, bone_d) > 0.2:
            cat = "skull_dural"
        elif sinus_d is not None and frac(cm, sinus_d) > 0.3:
            cat = "dural_sinus"
        elif cow_d is not None and frac(cm, cow_d) > 0.0:
            touch = np.unique(cow[ndimage.binary_dilation(cm, structure=ball(spacing, p.vessel_touch_mm)) & cow_bin])
            vessel = ",".join(TOPCOW_LABELS.get(int(t), str(t)) for t in touch if t > 0)
            inside = frac(cm, cow_bin)
            cat = "proximal_intraluminal" if (inside >= p.inside_lumen_min and c["compactness"] >= p.compact_min and c["equiv_diameter_mm"] <= 8) else "arterial_wall"
        elif vent is not None and (frac(cm, vent) >= p.choroid_inside_min or (vent_ok and frac(cm, vent_d) > 0.3 and c["compactness"] < 0.4)):
            cat = "choroid_plexus"   # inside the lumen, or (trustworthy mask) hugging the margin and irregular; compact foci next to the mask stay candidates
        elif pineal_zone is not None and frac(cm, pineal_zone) > 0.3:
            cat = "pineal_habenula"
        elif bg is not None and frac(cm, bg) > 0.3:
            cat = "basal_ganglia"
        c["category"] = cat; c["vessel_labels"] = vessel
    # small compact foci hugging a larger wall / skull calcification are partial-volume tails, not emboli
    big = [c for c in comps if c["category"] in ("arterial_wall", "skull_dural", "dural_sinus")]
    for c in comps:
        if c["category"] not in ("proximal_intraluminal", None):
            continue
        for o in big:
            if o["volume_mm3"] <= c["volume_mm3"]:
                continue
            dist = np.sqrt(sum(((c["centroid"][k] - o["centroid"][k]) * spacing[k]) ** 2 for k in range(3))) - o["max_extent_mm"] / 2
            if dist <= p.wall_adjacent_mm:
                c["category"] = o["category"] + "_adjacent"; break
    # bilateral symmetry among the still-unclassified foci
    free = [c for c in comps if c["category"] is None]
    for c in free:
        mirror = (2 * midline_i - c["centroid"][0], c["centroid"][1], c["centroid"][2])
        for o in free:
            if o is c:
                continue
            dist = np.sqrt(sum(((mirror[k] - o["centroid"][k]) * spacing[k]) ** 2 for k in range(3)))
            if dist <= p.mirror_mm:
                c["category"] = "symmetric_bilateral"; break
    for c in comps:
        if c["category"] is None:
            d_min = p.d_min_mm if c["tier"] == 1 else max(p.d_min_mm, p.tier2_d_min_mm)
            ok = d_min <= c["equiv_diameter_mm"] <= p.d_max_mm and c["compactness"] >= p.compact_min
            c["category"] = "distal_candidate" if ok else ("parenchymal_other" if c["tier"] == 1 or c["equiv_diameter_mm"] > p.d_max_mm else "subthreshold_noise")
            if c["category"] == "distal_candidate" and c["equiv_diameter_mm"] < p.low_conf_d_mm and c["max_hu"] < p.low_conf_hu:
                c["category"] = "distal_candidate_low"   # tiny and faint: the profile of noise / vessel specks on the negative control
    out = np.zeros(image.shape, np.uint16); cat_ids = {"distal_candidate": 1, "proximal_intraluminal": 2, "arterial_wall": 3, "choroid_plexus": 4,
                                                       "pineal_habenula": 5, "basal_ganglia": 6, "symmetric_bilateral": 7, "skull_dural": 8, "dural_sinus": 9, "parenchymal_other": 10,
                                                       "arterial_wall_adjacent": 11, "skull_dural_adjacent": 12, "dural_sinus_adjacent": 13, "subthreshold_noise": 14, "distal_candidate_low": 15}
    rows = []
    for c in comps:
        out[sl][c["comp"]] = cat_ids[c["category"]]
        rows.append({k: v for k, v in c.items() if k not in ("comp", "mask_id")} | {"centroid_ijk": [float(c["centroid"][k] + sl[k].start) for k in range(3)]})
    for r in rows:
        r.pop("centroid", None)
    meta = {"components_ge_tier2": len(comps), "counts": {k: sum(r["category"] == k for r in rows) for k in cat_ids},
            "ventricle_mask_ml": round(vent_ml, 1), "ventricle_mask_trusted": bool(vent_ok),
            "rules_active": {"skull": bone_d is not None, "venous_sinuses": sinus_d is not None, "ventricles": vent_d is not None,
                             "pineal_zone": pineal_zone is not None, "basal_ganglia": bg is not None, "cow": cow is not None}}
    return rows, out, meta, cat_ids


def write_fiducials(rows, affine, path: Path):
    pts = []
    for n, r in enumerate(rows, 1):
        if r["category"] not in ("distal_candidate", "proximal_intraluminal"):
            continue
        ras = (affine @ np.array([*r["centroid_ijk"], 1.0]))[:3]
        pts.append({"id": str(n), "label": f"{r['category']} {r['side']} {r['equiv_diameter_mm']:.1f}mm {r['max_hu']:.0f}HU T{r['tier']} {r['vessel_labels']}".strip(),
                    "position": [float(x) for x in ras], "orientation": [-1, 0, 0, 0, -1, 0, 0, 0, 1], "selected": True, "locked": False, "visibility": True})
    doc = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
           "markups": [{"type": "Fiducial", "coordinateSystem": "RAS", "coordinateUnits": "mm", "locked": False, "labelFormat": "%N-%d", "controlPoints": pts,
                        "display": {"color": [1.0, 0.2, 0.2], "glyphScale": 2.0, "textScale": 2.5}}]}
    path.write_text(json.dumps(doc, indent=1))


def qc_montage(image, rows, spacing, out_png, case_id):
    show = [r for r in rows if r["category"] in ("distal_candidate", "proximal_intraluminal")]
    show = sorted(show, key=lambda r: (-r["tier"] * -1, -r["max_hu"]))[:12]
    if not show:
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = 4; nrows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 3.6 * nrows)); axes = np.atleast_1d(axes).ravel()
    half = int(15 / spacing[0])
    for ax, r in zip(axes, show):
        ci, cj, ck = [int(round(x)) for x in r["centroid_ijk"]]
        i0, j0 = max(ci - half, 0), max(cj - half, 0)
        im = np.clip((image[i0:ci + half, j0:cj + half, ck].T[::-1] + 50) / 650, 0, 1)
        ax.imshow(im, cmap="gray"); ax.plot([ci - i0], [im.shape[0] - (cj - j0)], "r+", ms=12, mew=1.2)
        ax.set_title(f"{r['category'][:9]} {r['side']} {r['equiv_diameter_mm']:.1f}mm {r['max_hu']:.0f}HU T{r['tier']}", fontsize=7); ax.axis("off")
    for ax in axes[len(show):]:
        ax.axis("off")
    fig.suptitle(f"{case_id}: intracranial calcium, embolus candidates after anatomical exclusion", fontsize=9); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="CTA or NCCT NIfTI (reference grid)")
    ap.add_argument("--total-dir", type=Path, required=True, help="TotalSegmentator total output with brain.nii.gz and skull.nii.gz")
    ap.add_argument("--brain-structures-dir", type=Path); ap.add_argument("--ventricle-parts-dir", type=Path)
    ap.add_argument("--cow", type=Path, help="TopCoW labelmap on the same grid")
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--case-id")
    for f, v in asdict(Params()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv); p = preset(Params(**{f: getattr(a, f) for f in asdict(Params())}))
    case_id = a.case_id or a.image.name.replace(".nii.gz", "")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(a.image)); spacing = tuple(float(s) for s in ref.header.get_zooms()[:3]); image = np.asarray(ref.dataobj).astype(np.float32)
    bs = a.brain_structures_dir; vp = a.ventricle_parts_dir
    masks = {"brain": load(a.total_dir / "brain.nii.gz", ref), "skull": load(a.total_dir / "skull.nii.gz", ref),
             "venous_sinuses": load(bs / "venous_sinuses.nii.gz", ref) if bs else None,
             "ventricle": load(bs / "ventricle.nii.gz", ref) if bs else None,
             "third_ventricle": load(vp / "third_ventricle.nii.gz", ref) if vp else None,
             "cow": None, "anterior_is_plus_j": bool(ref.affine[1, 1] > 0)}
    if bs:
        parts = [load(bs / f"{n}.nii.gz", ref) for n in ("lentiform_nucleus", "caudate_nucleus", "thalamus")]
        parts = [x for x in parts if x is not None]
        masks["basal_ganglia"] = np.logical_or.reduce(parts) if parts else None
    if a.cow:
        cimg = nib.load(str(a.cow)); assert cimg.shape == ref.shape, "CoW labelmap grid differs"
        masks["cow"] = np.asarray(cimg.dataobj).astype(np.int32)
    if masks["brain"] is None:
        raise SystemExit("brain.nii.gz missing from --total-dir")
    rows, out, meta, cat_ids = search(image, masks, spacing, p)
    nib.save(nib.Nifti1Image(out, ref.affine), str(a.out_dir / f"{case_id}_intracranial_calcium_classes.nii.gz"))
    df = pd.DataFrame(rows); df.insert(0, "case_id", case_id); df.to_csv(a.out_dir / f"{case_id}_intracranial_calcium.csv", index=False)
    write_fiducials(rows, ref.affine, a.out_dir / f"{case_id}_embolus_candidates.mrk.json")
    qc_montage(image, rows, spacing, a.out_dir / f"{case_id}_embolus_candidates_qc.png", case_id)
    (a.out_dir / f"{case_id}_intracranial_calcium_manifest.json").write_text(json.dumps({"case_id": case_id, "image": str(a.image), "params": asdict(p),
        "class_ids": cat_ids, **meta, "software_version": __version__, "intended_use": "research candidate generation; not a diagnosis"}, indent=2))
    print(json.dumps(meta["counts"]))
    if len(df):
        print(df[df.category.isin(["distal_candidate", "proximal_intraluminal"])].drop(columns=["centroid_ijk"]).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
