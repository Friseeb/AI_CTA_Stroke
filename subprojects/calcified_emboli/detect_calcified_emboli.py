#!/usr/bin/env python3
"""Rule-based candidate detector for calcified cerebral emboli (CCE) on CTA.  v0, research only.

Idea: a calcified embolus is a small, compact, very-high-attenuation object sitting *inside* an
intracranial (or cervical) arterial lumen, i.e. within or immediately adjacent to a vessel mask,
away from bone, and brighter than the contrast-opacified lumen by a wide margin.  Atheromatous
wall calcification is eccentric/crescentic and hugs the vessel boundary, dural/choroid/pineal
calcifications are outside the arterial tree, and skull-base bone is excluded by a bone mask.

Inputs
  --cta           contrast CTA (NIfTI)
  --vessels       arterial labelmap on the same grid (TopCoW CLAIM circle-of-Willis labels, FLOWCAT,
                  or any binary vessel mask); other grids are resampled with nearest neighbour
  --bone          optional bone/skull mask (TotalSegmentator skull + vertebrae) to exclude
  --out-dir       outputs: candidates CSV, candidate labelmap, Slicer fiducials (.mrk.json), QC montage

Rules (all tunable):
  lumen reference  = median HU inside the eroded vessel mask (per case)
  bright           = HU >= max(min_hu, lumen_ref + margin_hu)             (default 700 / +300)
  near vessel      = within `search_mm` of the vessel mask (dilated search ROI)
  size             = equivalent diameter between d_min and d_max mm        (default 1.0-8.0)
  compact          = sphericity-like ratio (volume / bounding-sphere) >= `compactness_min`
  intraluminal     = fraction of the candidate inside the vessel mask >= `inside_fraction_min`
  bone-free        = no overlap with the (dilated) bone mask
Each candidate gets a score 0-1 from these and a nearest vessel label; readers confirm in Slicer.
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
    min_hu: float = 700.0
    margin_hu: float = 300.0
    search_mm: float = 2.0
    d_min_mm: float = 1.0
    d_max_mm: float = 8.0
    compactness_min: float = 0.25
    inside_fraction_min: float = 0.3
    bone_exclude_mm: float = 2.0
    vessel_erode_mm: float = 0.8


def ball(spacing, r_mm):
    r = [max(int(np.ceil(r_mm / s)), 0) for s in spacing]
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return sum((g[i] * spacing[i]) ** 2 for i in range(3)) <= r_mm ** 2


def resample_to(ref: nib.Nifti1Image, img: nib.Nifti1Image) -> np.ndarray:
    """Nearest-neighbour resample a labelmap onto the reference grid."""
    if img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3):
        return np.asarray(img.dataobj)
    M = np.linalg.inv(img.affine) @ ref.affine
    out = ndimage.affine_transform(np.asarray(img.dataobj), M[:3, :3], M[:3, 3], output_shape=ref.shape, order=0, mode="constant", cval=0)
    return out


def detect(image: np.ndarray, vessels: np.ndarray, spacing, p: Params, bone: np.ndarray | None = None):
    vox = float(np.prod(spacing))
    vmask = vessels > 0
    if not vmask.any():
        raise RuntimeError("empty vessel mask")
    # bounding box work region
    idx = np.where(vmask); pad = [int(np.ceil((p.search_mm + p.d_max_mm) / s)) + 2 for s in spacing]
    sl = tuple(slice(max(idx[i].min() - pad[i], 0), min(idx[i].max() + pad[i] + 1, vmask.shape[i])) for i in range(3))
    img = image[sl]; vm = vmask[sl]; vl = vessels[sl]
    core = ndimage.binary_erosion(vm, structure=ball(spacing, p.vessel_erode_mm)) if p.vessel_erode_mm > 0 else vm
    ref_vals = img[core] if core.any() else img[vm]
    lumen_ref = float(np.median(ref_vals)); lumen_sd = float(np.std(ref_vals))
    thr = max(p.min_hu, lumen_ref + p.margin_hu)
    search = ndimage.binary_dilation(vm, structure=ball(spacing, p.search_mm))
    bright = search & (img >= thr)
    if bone is not None:
        b = ndimage.binary_dilation(bone[sl] > 0, structure=ball(spacing, p.bone_exclude_mm)) if p.bone_exclude_mm > 0 else bone[sl] > 0
        bright &= ~b
    lab, n = ndimage.label(bright, structure=np.ones((3, 3, 3)))
    rows = []; keep = np.zeros_like(lab)
    for i in range(1, n + 1):
        comp = lab == i
        nvox = int(comp.sum()); vol = nvox * vox
        d_eq = (6 * vol / np.pi) ** (1 / 3)
        if not (p.d_min_mm <= d_eq <= p.d_max_mm):
            continue
        ii = np.where(comp)
        ext = [(ii[k].max() - ii[k].min() + 1) * spacing[k] for k in range(3)]
        d_max_ext = max(ext)
        compact = vol / (np.pi / 6 * d_max_ext ** 3) if d_max_ext > 0 else 0.0
        inside = float((comp & vm).sum() / nvox)
        if compact < p.compactness_min or inside < p.inside_fraction_min:
            continue
        hu = img[comp]
        touch_labels = np.unique(vl[ndimage.binary_dilation(comp, structure=np.ones((3, 3, 3))) & vm])
        touch_labels = [int(t) for t in touch_labels if t > 0]
        label_name = ",".join(TOPCOW_LABELS.get(t, str(t)) for t in touch_labels) or "unlabelled"
        c = ndimage.center_of_mass(comp)
        score = float(np.clip(0.35 * min(inside / 0.8, 1) + 0.25 * min(compact / 0.5, 1) + 0.25 * min((hu.max() - thr) / 500, 1)
                              + 0.15 * (1 - abs(d_eq - 3.0) / 5.0), 0, 1))
        rows.append({"candidate_id": len(rows) + 1, "equiv_diameter_mm": d_eq, "max_extent_mm": d_max_ext, "volume_mm3": vol,
                     "mean_hu": float(hu.mean()), "max_hu": float(hu.max()), "compactness": compact, "inside_vessel_fraction": inside,
                     "vessel_labels": label_name, "score": score,
                     "centroid_ijk": [float(c[k] + sl[k].start) for k in range(3)]})
        keep[comp] = len(rows)
    out = np.zeros(image.shape, np.uint16); out[sl] = keep
    meta = {"lumen_reference_hu": lumen_ref, "lumen_sd_hu": lumen_sd, "threshold_hu": thr, "raw_components": int(n), "candidates": len(rows)}
    return rows, out, meta


def write_fiducials(rows, affine, path: Path, case_id: str):
    pts = []
    for r in rows:
        ras = (affine @ np.array([*r["centroid_ijk"], 1.0]))[:3]
        pts.append({"id": str(r["candidate_id"]), "label": f"CCE-{r['candidate_id']} {r['vessel_labels']} {r['equiv_diameter_mm']:.1f}mm s={r['score']:.2f}",
                    "position": [float(x) for x in ras], "orientation": [-1, 0, 0, 0, -1, 0, 0, 0, 1], "selected": True, "locked": False, "visibility": True})
    doc = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
           "markups": [{"type": "Fiducial", "coordinateSystem": "RAS", "coordinateUnits": "mm", "locked": False, "labelFormat": "%N-%d",
                        "controlPoints": pts, "display": {"color": [1.0, 0.2, 0.2], "glyphScale": 2.0, "textScale": 2.5}}]}
    path.write_text(json.dumps(doc, indent=1))


def qc_montage(image, cand, rows, spacing, out_png, case_id):
    if not rows:
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = sorted(rows, key=lambda r: -r["score"])[:9]
    cols = 3; nrows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 4 * nrows)); axes = np.atleast_1d(axes).ravel()
    half = int(15 / spacing[0])
    for ax, r in zip(axes, top):
        ci, cj, ck = [int(round(x)) for x in r["centroid_ijk"]]
        i0, i1 = max(ci - half, 0), ci + half; j0, j1 = max(cj - half, 0), cj + half
        im = np.clip((image[i0:i1, j0:j1, ck].T[::-1] + 100) / 1100, 0, 1)
        ax.imshow(im, cmap="gray"); ax.contour((cand[i0:i1, j0:j1, ck] == r["candidate_id"]).T[::-1], levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(f"#{r['candidate_id']} {r['vessel_labels']} {r['equiv_diameter_mm']:.1f} mm {r['max_hu']:.0f} HU s={r['score']:.2f}", fontsize=8); ax.axis("off")
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle(f"{case_id}: calcified embolus candidates (top by score)", fontsize=10); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cta", type=Path, required=True); ap.add_argument("--vessels", type=Path, required=True)
    ap.add_argument("--bone", type=Path); ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--case-id")
    for f, v in asdict(Params()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args(argv)
    p = Params(**{f: getattr(a, f) for f in asdict(Params())})
    case_id = a.case_id or a.cta.name.replace(".nii.gz", "")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(a.cta)); spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    image = np.asarray(ref.dataobj).astype(np.float32)
    vessels = resample_to(ref, nib.load(str(a.vessels))).astype(np.int32)
    bone = resample_to(ref, nib.load(str(a.bone))) if a.bone else None
    rows, cand, meta = detect(image, vessels, spacing, p, bone)
    nib.save(nib.Nifti1Image(cand, ref.affine), str(a.out_dir / f"{case_id}_cce_candidates.nii.gz"))
    df = pd.DataFrame(rows); df.insert(0, "case_id", case_id)
    df.to_csv(a.out_dir / f"{case_id}_cce_candidates.csv", index=False)
    write_fiducials(rows, ref.affine, a.out_dir / f"{case_id}_cce_candidates.mrk.json", case_id)
    qc_montage(image, cand, rows, spacing, a.out_dir / f"{case_id}_cce_qc.png", case_id)
    (a.out_dir / f"{case_id}_cce_manifest.json").write_text(json.dumps({"case_id": case_id, "cta": str(a.cta), "vessels": str(a.vessels),
        "bone": str(a.bone) if a.bone else None, "params": asdict(p), **meta, "software_version": __version__,
        "intended_use": "research candidate generation for reader review; not a diagnosis"}, indent=2))
    print(json.dumps(meta)); 
    if rows:
        print(df.drop(columns=["centroid_ijk"]).sort_values("score", ascending=False).round(2).head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
