#!/usr/bin/env python3
"""Opportunistic aortic valve (AVC) and mitral annular (MAC) calcium on contrast CTA.

Inputs are TotalSegmentator outputs (per-structure NIfTI folders):
  heartchambers_highres : heart_atrium_left, heart_ventricle_left, heart_myocardium, aorta
  aortic_sinuses        : left_ventricular_outflow_tract, right/left/non_coronary_cusp
  coronary_arteries     : coronary_arteries (optional; excluded from both ROIs)
  STOPSTORM (optional)  : <case>_Valve_Mitral.nii.gz on the CTA grid -> learned mitral region (second definition)

ROIs (all on the CTA grid, in mm):
  aortic valve complex = cusps  U  (LVOT within `av_lvot_band_mm` of the cusps), dilated by
                         `av_dilate_mm`, minus dilated coronaries.
  mitral annulus       = band where dilate(LA) and dilate(LV) meet (`mv_band_mm`), restricted to
                         the heart (LA U LV U myocardium) dilated by `mv_dilate_mm`,
                         minus the aortic valve complex (aorto-mitral continuity), minus coronaries.
  mitral (STOPSTORM)   = Valve_Mitral label dilated `stopstorm_dilate_mm`, minus aortic complex, minus coronaries.
                         The aortic valve uses the TotalSegmentator sinuses only (no second definition).

Calcium (contrast-adaptive, same logic family as aorta_cta_radiomics.calcification):
  blood reference = median HU of the eroded blood pool (LVOT + cusps for AV; LA + LV for MV)
  candidate >= max(min_candidate_hu, ref + margin); seed >= max(seed_hu, ref + margin)
  calcium = candidate voxels connected to a seed.  Fixed thresholds also reported.

Outputs per case: ROI + calcium masks, features CSV (long format), QC montage PNG, manifest JSON.
Research use only. Non-gated CTA: scores are 'Agatston-like', not Agatston.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

__version__ = "0.1.0"

CHAMBER_FILES = {
    "la": "heart_atrium_left.nii.gz",
    "lv": "heart_ventricle_left.nii.gz",
    "myo": "heart_myocardium.nii.gz",
    "aorta": "aorta.nii.gz",
}
STOPSTORM_FILES = {"ss_mitral": "{case}_Valve_Mitral.nii.gz"}
SINUS_FILES = {
    "lvot": "left_ventricular_outflow_tract.nii.gz",
    "rcc": "right_coronary_cusp.nii.gz",
    "lcc": "left_coronary_cusp.nii.gz",
    "ncc": "non_coronary_cusp.nii.gz",
}


@dataclass
class Params:
    av_lvot_band_mm: float = 5.0
    av_dilate_mm: float = 2.0
    mv_band_mm: float = 4.0
    mv_dilate_mm: float = 2.0
    coronary_exclude_mm: float = 1.5
    blood_erode_mm: float = 3.0
    seed_hu: float = 500.0
    min_candidate_hu: float = 300.0
    lumen_margin_hu: float = 150.0
    fixed_thresholds_hu: tuple[int, ...] = (300, 500, 600, 800)
    min_component_mm3: float = 1.0
    stopstorm_dilate_mm: float = 2.0


# --------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask(path: Path, ref: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{path.name}: grid differs from the CTA (shape {img.shape} vs {ref.shape})")
    return np.asarray(img.dataobj) > 0


def ball(spacing: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    r = [max(int(np.ceil(radius_mm / s)), 0) for s in spacing]
    if max(r) == 0:
        return np.ones((1, 1, 1), bool)
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    d = sum((g[i] * spacing[i]) ** 2 for i in range(3))
    return d <= radius_mm ** 2


def dilate(mask: np.ndarray, spacing, mm: float) -> np.ndarray:
    if mm <= 0 or not mask.any():
        return mask.copy()
    return _in_bbox(mask, spacing, mm, lambda m: ndimage.binary_dilation(m, structure=ball(spacing, mm)))


def erode(mask: np.ndarray, spacing, mm: float) -> np.ndarray:
    if mm <= 0 or not mask.any():
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=ball(spacing, mm))


def _in_bbox(mask, spacing, pad_mm, fn):
    """Apply a morphology op only inside the padded bounding box (full-volume ops are slow at 0.45 mm)."""
    idx = np.where(mask)
    pad = [int(np.ceil(pad_mm / s)) + 1 for s in spacing]
    sl = tuple(slice(max(idx[i].min() - pad[i], 0), min(idx[i].max() + pad[i] + 1, mask.shape[i])) for i in range(3))
    out = np.zeros_like(mask)
    out[sl] = fn(mask[sl])
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def remove_small(mask: np.ndarray, voxel_mm3: float, min_mm3: float) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1)) * voxel_mm3
    keep = np.flatnonzero(sizes >= min_mm3) + 1
    return np.isin(lab, keep)


def connected_to_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(candidate, structure=np.ones((3, 3, 3)))
    if n == 0:
        return candidate
    seeded = np.unique(lab[seed & candidate])
    seeded = seeded[seeded > 0]
    return np.isin(lab, seeded)


def density_factor(max_hu: float) -> int:
    return 4 if max_hu >= 400 else 3 if max_hu >= 300 else 2 if max_hu >= 200 else 1 if max_hu >= 130 else 0


def agatston_like(image: np.ndarray, calcium: np.ndarray, spacing, axis: int = 2) -> float:
    """Slice-wise Agatston-style score (area x density factor per lesion per axial slice).
    Non-gated, thin-slice CTA: reported as 'agatston_like', not comparable to gated 3 mm Agatston."""
    area_mm2 = spacing[0] * spacing[1]
    score = 0.0
    for k in np.flatnonzero(calcium.any(axis=(0, 1))):
        sl = calcium[:, :, k]
        lab, n = ndimage.label(sl)
        for i in range(1, n + 1):
            comp = lab == i
            score += comp.sum() * area_mm2 * density_factor(float(image[:, :, k][comp].max()))
    # classic Agatston used 3 mm slices; scale to a 3 mm-equivalent so numbers are on a familiar range
    return score * (spacing[2] / 3.0)


# --------------------------------------------------------------------------- core
def build_rois(masks: dict[str, np.ndarray], spacing, p: Params) -> dict[str, np.ndarray]:
    cusps = masks["rcc"] | masks["lcc"] | masks["ncc"]
    lvot = masks["lvot"]
    la, lv, myo = masks["la"], masks["lv"], masks["myo"]
    coronary = masks.get("coronary")

    cusps_band = dilate(cusps, spacing, p.av_lvot_band_mm)
    av_core = cusps | (lvot & cusps_band)
    av_roi = dilate(av_core, spacing, p.av_dilate_mm)

    la_d = dilate(la, spacing, p.mv_band_mm)
    lv_d = dilate(lv, spacing, p.mv_band_mm)
    heart = dilate(la | lv | myo, spacing, p.mv_dilate_mm)
    mv_band = la_d & lv_d & heart
    mv_roi = mv_band & ~dilate(av_roi, spacing, 1.0)
    mv_roi = largest_component(mv_roi) if mv_roi.any() else mv_roi

    if coronary is not None and coronary.any():
        cor_d = dilate(coronary, spacing, p.coronary_exclude_mm)
        av_roi &= ~cor_d
        mv_roi &= ~cor_d

    cor_d = dilate(coronary, spacing, p.coronary_exclude_mm) if coronary is not None and coronary.any() else None
    extra = {}
    if masks.get("ss_mitral") is not None and masks["ss_mitral"].any():
        m = dilate(masks["ss_mitral"], spacing, p.stopstorm_dilate_mm) & ~dilate(av_roi, spacing, 1.0)
        extra["mv_stopstorm_roi"] = m & ~cor_d if cor_d is not None else m

    av_blood = erode(lvot | cusps, spacing, p.blood_erode_mm)
    mv_blood = erode(la, spacing, p.blood_erode_mm) | erode(lv, spacing, p.blood_erode_mm)
    return {"av_roi": av_roi, "mv_roi": mv_roi, "av_blood": av_blood, "mv_blood": mv_blood, **extra}


def fov_flags(masks: dict[str, np.ndarray], rois: dict[str, np.ndarray], spacing) -> dict[str, float]:
    """Flag inferior/superior truncation of the heart: extended-arm head-neck CTAs often end through the
    ventricles, so the mitral annulus can sit at or beyond the scan edge."""
    nz = masks["lv"].shape[2]
    out: dict[str, float] = {}
    for name in ("lv", "la", "myo"):
        z = np.flatnonzero(masks[name].any(axis=(0, 1)))
        out[f"{name}_touches_inferior_edge"] = float(z.size > 0 and z.min() == 0)
        out[f"{name}_touches_superior_edge"] = float(z.size > 0 and z.max() == nz - 1)
    for name in ("av_roi", "mv_roi"):
        z = np.flatnonzero(rois[name].any(axis=(0, 1)))
        out[f"{name}_min_distance_to_inferior_edge_mm"] = float(z.min() * spacing[2]) if z.size else float("nan")
        out[f"{name}_z_extent_mm"] = float((z.max() - z.min() + 1) * spacing[2]) if z.size else 0.0
    out["mv_roi_possibly_truncated"] = float(out["lv_touches_inferior_edge"] and out["mv_roi_min_distance_to_inferior_edge_mm"] < 10.0)
    return out


def score_calcium(image: np.ndarray, roi: np.ndarray, blood: np.ndarray, spacing, p: Params) -> tuple[np.ndarray, dict]:
    voxel_mm3 = float(np.prod(spacing))
    blood_vals = image[blood] if blood.any() else image[roi]
    ref = float(np.median(blood_vals)) if blood_vals.size else float("nan")
    ref_sd = float(np.std(blood_vals)) if blood_vals.size else float("nan")
    cand_thr = max(p.min_candidate_hu, ref + p.lumen_margin_hu) if np.isfinite(ref) else p.min_candidate_hu
    seed_thr = max(p.seed_hu, cand_thr)
    candidate = roi & (image >= cand_thr)
    seed = roi & (image >= seed_thr)
    calcium = connected_to_seed(candidate, seed)
    calcium = remove_small(calcium, voxel_mm3, p.min_component_mm3)

    feats: dict[str, float] = {
        "roi_volume_ml": roi.sum() * voxel_mm3 / 1000.0,
        "blood_reference_hu": ref, "blood_reference_sd_hu": ref_sd,
        "candidate_threshold_hu": cand_thr, "seed_threshold_hu": seed_thr,
        "calcium_volume_mm3": calcium.sum() * voxel_mm3,
        "calcium_mean_hu": float(image[calcium].mean()) if calcium.any() else 0.0,
        "calcium_max_hu": float(image[calcium].max()) if calcium.any() else 0.0,
        "calcium_mass_score_mg_eq": float(image[calcium].sum()) * voxel_mm3 / 1000.0 if calcium.any() else 0.0,
        "calcium_components": int(ndimage.label(calcium, structure=np.ones((3, 3, 3)))[1]),
        "agatston_like_3mm_equiv": agatston_like(image, calcium, spacing),
    }
    for t in p.fixed_thresholds_hu:
        m = remove_small(roi & (image >= t), voxel_mm3, p.min_component_mm3)
        feats[f"fixed_{t}hu_volume_mm3"] = m.sum() * voxel_mm3
    # literature-style patient-specific thresholds: blood-pool mean + k*SD (Eberhard 2017, Laohachewin 2024, Dong 2025)
    if blood_vals.size:
        mean = float(blood_vals.mean())
        for k in (3, 4):
            t = mean + k * ref_sd
            m = remove_small(roi & (image >= t), voxel_mm3, p.min_component_mm3)
            feats[f"bloodpool_mean_plus{k}sd_threshold_hu"] = t
            feats[f"bloodpool_mean_plus{k}sd_volume_mm3"] = m.sum() * voxel_mm3
    return calcium, feats


def qc_montage(image, rois, calc, spacing, out_png: Path, case_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    sets = [("aortic valve", rois["av_roi"], calc["av"]), ("mitral annulus", rois["mv_roi"], calc["mv"])]
    if "mv_stopstorm" in calc:
        sets.append(("mitral valve (STOPSTORM)", rois["mv_stopstorm_roi"], calc["mv_stopstorm"]))
    for name, roi, cal in sets:
        if not roi.any():
            continue
        zs = np.flatnonzero(roi.any(axis=(0, 1)))
        for frac in (0.25, 0.5, 0.75):
            k = int(zs[min(int(frac * len(zs)), len(zs) - 1)])
            panels.append((name, k, roi[:, :, k], cal[:, :, k]))
    if not panels:
        return
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    any_roi = rois["av_roi"] | rois["mv_roi"]
    idx = np.where(any_roi)
    pad = int(25 / spacing[0])
    x0, x1 = max(idx[0].min() - pad, 0), idx[0].max() + pad
    y0, y1 = max(idx[1].min() - pad, 0), idx[1].max() + pad
    for ax, (name, k, roi2, cal2) in zip(axes, panels):
        im = np.clip((image[x0:x1, y0:y1, k].T[::-1] + 150) / 900, 0, 1)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.contour(roi2[x0:x1, y0:y1].T[::-1], levels=[0.5], colors="cyan", linewidths=0.8)
        if cal2.any():
            ov = np.zeros((*cal2[x0:x1, y0:y1].T.shape, 4)); ov[cal2[x0:x1, y0:y1].T[::-1]] = (1, 0.2, 0.2, 0.9)
            ax.imshow(ov)
        ax.set_title(f"{name}  z-index {k}", fontsize=9); ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{case_id}  cyan = ROI, red = calcium", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def run_case(cta: Path, chambers_dir: Path, sinuses_dir: Path, coronary_dir: Path | None, out_dir: Path, p: Params, case_id: str,
             stopstorm_dir: Path | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(cta))
    spacing = tuple(float(s) for s in ref.header.get_zooms()[:3])
    image = np.asarray(ref.dataobj).astype(np.float32)
    masks = {k: load_mask(chambers_dir / f, ref) for k, f in CHAMBER_FILES.items()}
    masks.update({k: load_mask(sinuses_dir / f, ref) for k, f in SINUS_FILES.items()})
    if stopstorm_dir is not None:
        for k, f in STOPSTORM_FILES.items():
            pth = stopstorm_dir / f.format(case=case_id)
            if pth.is_file():
                masks[k] = load_mask(pth, ref)
    cor_path = coronary_dir / "coronary_arteries.nii.gz" if coronary_dir else None
    if cor_path and cor_path.is_file():
        masks["coronary"] = load_mask(cor_path, ref)
    qc = {k: float(v.sum() * np.prod(spacing) / 1000.0) for k, v in masks.items()}
    missing = [k for k in ("rcc", "lcc", "ncc", "lvot", "la", "lv") if not masks[k].any()]
    if missing:
        raise RuntimeError(f"{case_id}: empty segmentation for {missing}; valve ROIs cannot be built")

    rois = build_rois(masks, spacing, p)
    fov = fov_flags(masks, rois, spacing)
    calc, feats = {}, {}
    calc["av"], feats["aortic_valve"] = score_calcium(image, rois["av_roi"], rois["av_blood"], spacing, p)
    calc["mv"], feats["mitral_annulus"] = score_calcium(image, rois["mv_roi"], rois["mv_blood"], spacing, p)
    to_save = [("av_roi", rois["av_roi"]), ("mv_roi", rois["mv_roi"]), ("av_calcium", calc["av"]), ("mv_calcium", calc["mv"])]
    if "mv_stopstorm_roi" in rois:
        calc["mv_stopstorm"], feats["mitral_valve_stopstorm"] = score_calcium(image, rois["mv_stopstorm_roi"], rois["mv_blood"], spacing, p)
        to_save += [("mv_stopstorm_roi", rois["mv_stopstorm_roi"]), ("mv_stopstorm_calcium", calc["mv_stopstorm"])]
        fov["mv_stopstorm_touches_inferior_edge"] = float(np.flatnonzero(rois["mv_stopstorm_roi"].any(axis=(0, 1))).min() == 0)

    for name, arr in to_save:
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), ref.affine), str(out_dir / f"{case_id}_{name}.nii.gz"))
    qc_montage(image, rois, calc, spacing, out_dir / f"{case_id}_valve_calcium_qc.png", case_id)

    rows = []
    feats["field_of_view"] = fov
    for region, fd in feats.items():
        for k, v in fd.items():
            rows.append({"case_id": case_id, "region": region, "feature": k, "value": v, "software_version": __version__})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{case_id}_valve_calcium_features.csv", index=False)
    manifest = {
        "case_id": case_id, "cta": str(cta), "cta_sha256": sha256(cta), "spacing_mm": spacing,
        "chambers_dir": str(chambers_dir), "sinuses_dir": str(sinuses_dir), "coronary_dir": str(coronary_dir) if coronary_dir else None,
        "coronary_excluded": "coronary" in masks, "stopstorm_dir": str(stopstorm_dir) if stopstorm_dir else None,
        "stopstorm_mitral_used": "ss_mitral" in masks, "params": asdict(p), "segmentation_volumes_ml": qc,
        "software_version": __version__, "intended_use": "research; non-gated contrast CTA; not an Agatston score",
    }
    (out_dir / f"{case_id}_valve_calcium_manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cta", type=Path, required=True)
    ap.add_argument("--chambers-dir", type=Path, required=True)
    ap.add_argument("--sinuses-dir", type=Path, required=True)
    ap.add_argument("--coronary-dir", type=Path)
    ap.add_argument("--stopstorm-dir", type=Path, help="folder with <case>_Valve_Mitral.nii.gz on the CTA grid (optional)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--case-id")
    for f, v in asdict(Params()).items():
        if f == "fixed_thresholds_hu":
            continue
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args(argv)
    p = Params(**{f: getattr(a, f) for f in asdict(Params()) if f != "fixed_thresholds_hu"})
    case_id = a.case_id or a.cta.name.replace(".nii.gz", "")
    df = run_case(a.cta, a.chambers_dir, a.sinuses_dir, a.coronary_dir, a.out_dir, p, case_id, a.stopstorm_dir)
    print(df.pivot(index="feature", columns="region", values="value").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
