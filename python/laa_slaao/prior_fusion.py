"""Prior fusion for thrombus-inclusive LAA segmentation.

Combines NUDF, VISTA3D (nv_segment_ct), and TotalSegmentator masks into:
- Consensus LAA mask (majority vote across available priors)
- Union / intersection masks
- Disagreement map (where priors disagree)
- Positive anatomical priors (LA + LAA cavity)
- Negative anatomical priors (aorta, lungs, myocardium, coronaries, etc.)
- Distance transform from negative prior boundary
- Exclusion mask

Source priority for negative priors:
  heartchambers_highres  > total (myocardium, aorta, pulmonary_artery)
  coronary_arteries task > total (coronary_arteries)
  VISTA3D                > total (lungs, pulmonary_vein)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


# ------------------------------------------------------------------
# Label maps
# ------------------------------------------------------------------

# TotalSegmentator heartchambers_highres task — per-structure file names
_HC_HIGHRES_LABELS = {
    "heart_myocardium":     "heart_myocardium.nii.gz",
    "heart_atrium_left":    "heart_atrium_left.nii.gz",
    "heart_ventricle_left": "heart_ventricle_left.nii.gz",
    "heart_atrium_right":   "heart_atrium_right.nii.gz",
    "heart_ventricle_right":"heart_ventricle_right.nii.gz",
    "aorta":                "aorta.nii.gz",
    "pulmonary_artery":     "pulmonary_artery.nii.gz",
}

# TotalSegmentator coronary_arteries task — per-structure file names
_CORONARY_LABELS = {
    "coronary_arteries":    "coronary_arteries.nii.gz",
    "right_coronary_cusp":  "right_coronary_cusp.nii.gz",
    "left_coronary_cusp":   "left_coronary_cusp.nii.gz",
    "non_coronary_cusp":    "non_coronary_cusp.nii.gz",
}

# TotalSegmentator total task — fallback per-structure file names
_TOTAL_NEG_FALLBACKS = [
    "aorta.nii.gz",
    "pulmonary_artery.nii.gz",
    "pulmonary_vein.nii.gz",
    "lung_upper_lobe_left.nii.gz",
    "lung_lower_lobe_left.nii.gz",
    "lung_upper_lobe_right.nii.gz",
    "lung_lower_lobe_right.nii.gz",
    "lung_middle_lobe_right.nii.gz",
    "heart_myocardium.nii.gz",
    "coronary_arteries.nii.gz",
    "pericardium.nii.gz",
    "vertebrae_T1.nii.gz",
    "vertebrae_T2.nii.gz",
    "vertebrae_T3.nii.gz",
    "vertebrae_T4.nii.gz",
    "sternum.nii.gz",
    "rib_left_1.nii.gz",
    "rib_right_1.nii.gz",
]

# TotalSegmentator total task — positive prior file names
_TOTAL_POS_STRUCTURES = [
    "left_atrium.nii.gz",
    "atrial_appendage_left.nii.gz",
]

# VISTA3D (nv_segment_ct) label IDs in the combined multi-label output
VISTA3D_LABEL_IDS = {
    # positive
    "laa":            108,
    "heart":          115,
    # negative
    "aorta":            6,
    "lung":            20,
    "left_lung_upper": 28,
    "left_lung_lower": 29,
    "right_lung_upper":30,
    "right_lung_mid":  31,
    "right_lung_lower":32,
    "pulmonary_vein": 119,
}
VISTA3D_NEG_LABEL_IDS = [6, 20, 28, 29, 30, 31, 32, 119]
VISTA3D_POS_LABEL_IDS = [108, 115]


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------

@dataclass
class PriorFusionResult:
    """All outputs from prior fusion for a single case."""
    case_id: str
    shape: tuple
    affine: np.ndarray

    # Individual LAA masks (binary uint8)
    nudf_laa: Optional[np.ndarray] = None
    vista3d_laa: Optional[np.ndarray] = None
    totalseg_laa: Optional[np.ndarray] = None

    # Fused LAA representations
    union_laa: Optional[np.ndarray] = None
    intersection_laa: Optional[np.ndarray] = None
    consensus_laa: Optional[np.ndarray] = None     # majority vote
    disagreement_map: Optional[np.ndarray] = None  # float 0–1

    # Anatomical priors
    positive_prior: Optional[np.ndarray] = None
    negative_prior: Optional[np.ndarray] = None
    negative_distance: Optional[np.ndarray] = None  # mm from negative boundary

    sources_used: list = field(default_factory=list)

    def n_sources(self) -> int:
        return len(self.sources_used)

    def summary(self) -> dict:
        def vox(arr):
            return int(arr.sum()) if arr is not None else None
        return {
            "case_id": self.case_id,
            "sources_used": self.sources_used,
            "nudf_laa_voxels": vox(self.nudf_laa),
            "vista3d_laa_voxels": vox(self.vista3d_laa),
            "totalseg_laa_voxels": vox(self.totalseg_laa),
            "union_voxels": vox(self.union_laa),
            "intersection_voxels": vox(self.intersection_laa),
            "consensus_voxels": vox(self.consensus_laa),
            "positive_prior_voxels": vox(self.positive_prior),
            "negative_prior_voxels": vox(self.negative_prior),
            "disagreement_mean": (
                float(self.disagreement_map.mean())
                if self.disagreement_map is not None else None
            ),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_binary(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return (img.get_fdata() > 0).astype(np.uint8)


def _load_label(path: Path, label_id: int) -> np.ndarray:
    img = nib.load(str(path))
    data = np.round(img.get_fdata()).astype(np.int32)
    return (data == label_id).astype(np.uint8)


def _load_labels_union(path: Path, label_ids: list[int]) -> np.ndarray:
    """Load multiple label IDs from a multi-label NIfTI and return their union."""
    img = nib.load(str(path))
    data = np.round(img.get_fdata()).astype(np.int32)
    out = np.zeros(data.shape[:3], dtype=np.uint8)
    for lid in label_ids:
        out |= (data == lid).astype(np.uint8)
    return out


def _try_load(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return _load_binary(path)


def _try_struct(base_dir: Optional[Path], filename: str, ref_shape: tuple) -> Optional[np.ndarray]:
    if base_dir is None or not base_dir.is_dir():
        return None
    p = base_dir / filename
    if not p.exists():
        return None
    m = _load_binary(p)
    return m if m.shape == ref_shape else None


def _get_voxel_spacing(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def fuse_priors(
    case_id: str,
    *,
    # LAA source masks
    nudf_laa_path: Optional[Path] = None,
    vista3d_laa_path: Optional[Path] = None,
    vista3d_label_id: int = 108,
    totalseg_laa_path: Optional[Path] = None,
    totalseg_la_label_id: Optional[int] = None,
    # High-res TotalSegmentator task dirs (per-structure NIfTI files)
    totalseg_total_dir: Optional[Path] = None,          # 'total' task output
    totalseg_heart_dir: Optional[Path] = None,          # 'heartchambers_highres' task output
    totalseg_coronary_dir: Optional[Path] = None,       # 'coronary_arteries' task output
    # VISTA3D combined multi-label output (all 133 classes)
    vista3d_combined_path: Optional[Path] = None,
    # Reference image
    ref_image_path: Optional[Path] = None,
) -> PriorFusionResult:
    """Fuse LAA priors and build positive/negative anatomical prior maps.

    Source priority for negative priors (first found wins per structure):
      heartchambers_highres dir > total dir  (myocardium, aorta, pulmonary_artery)
      coronary_arteries dir     > total dir  (coronary_arteries)
      VISTA3D combined          > total dir  (lungs, pulmonary_vein)
    """
    # --- reference shape/affine ---
    ref_path = (
        nudf_laa_path or vista3d_laa_path or totalseg_laa_path
        or vista3d_combined_path or ref_image_path
    )
    if ref_path is None:
        raise ValueError("At least one mask path or ref_image_path must be provided.")
    ref_img = nib.load(str(ref_path))
    affine = ref_img.affine
    ref_shape = ref_img.get_fdata().shape[:3]

    result = PriorFusionResult(case_id=case_id, shape=ref_shape, affine=affine)
    masks = []

    # --- load LAA masks from each source ---
    if nudf_laa_path is not None and nudf_laa_path.exists():
        result.nudf_laa = _load_binary(nudf_laa_path)
        masks.append(result.nudf_laa)
        result.sources_used.append("nudf")

    if vista3d_laa_path is not None and vista3d_laa_path.exists():
        result.vista3d_laa = _load_label(vista3d_laa_path, vista3d_label_id)
        masks.append(result.vista3d_laa)
        result.sources_used.append("vista3d")
    elif vista3d_combined_path is not None and vista3d_combined_path.exists():
        # Extract LAA from the combined VISTA3D multi-label output
        laa = _load_label(vista3d_combined_path, VISTA3D_LABEL_IDS["laa"])
        if laa.sum() > 0:
            result.vista3d_laa = laa
            masks.append(laa)
            result.sources_used.append("vista3d_combined")

    if totalseg_laa_path is not None and totalseg_laa_path.exists():
        if totalseg_la_label_id is not None:
            result.totalseg_laa = _load_label(totalseg_laa_path, totalseg_la_label_id)
        else:
            result.totalseg_laa = _load_binary(totalseg_laa_path)
        masks.append(result.totalseg_laa)
        result.sources_used.append("totalseg")

    if not masks:
        raise ValueError(f"No valid LAA masks found for case {case_id}.")

    # --- fused LAA representations ---
    stack = np.stack(masks, axis=0).astype(np.float32)
    result.union_laa = (stack.max(axis=0) > 0).astype(np.uint8)
    result.intersection_laa = (stack.min(axis=0) > 0).astype(np.uint8)
    vote = stack.mean(axis=0)
    result.consensus_laa = (vote >= 0.5).astype(np.uint8)

    majority = result.consensus_laa.astype(np.float32)
    disagreement = np.zeros(ref_shape, dtype=np.float32)
    for m in masks:
        disagreement += np.abs(m.astype(np.float32) - majority)
    result.disagreement_map = (disagreement / len(masks)).astype(np.float32)

    # --- positive anatomical priors ---
    pos = result.consensus_laa.copy()

    # heartchambers_highres: LA body (best quality)
    la_highres = _try_struct(totalseg_heart_dir, "heart_atrium_left.nii.gz", ref_shape)
    if la_highres is not None:
        pos |= la_highres
    else:
        # fallback: total task LA
        la_total = _try_struct(totalseg_total_dir, "left_atrium.nii.gz", ref_shape)
        if la_total is not None:
            pos |= la_total

    # total task: LAA from TotalSegmentator (atrial_appendage_left)
    laa_ts = _try_struct(totalseg_total_dir, "atrial_appendage_left.nii.gz", ref_shape)
    if laa_ts is not None:
        pos |= laa_ts

    # VISTA3D: LAA (108) and heart (115) — additional positive support
    if vista3d_combined_path is not None and vista3d_combined_path.exists():
        vista_pos = _load_labels_union(vista3d_combined_path, VISTA3D_POS_LABEL_IDS)
        if vista_pos.shape == ref_shape:
            pos |= vista_pos

    result.positive_prior = pos

    # --- negative anatomical priors ---
    neg = np.zeros(ref_shape, dtype=np.uint8)

    def _add_neg(mask: Optional[np.ndarray]) -> None:
        if mask is not None and mask.shape == ref_shape:
            neg.__ior__(mask)

    # Myocardium: heartchambers_highres > total
    myo = _try_struct(totalseg_heart_dir, "heart_myocardium.nii.gz", ref_shape)
    if myo is None:
        myo = _try_struct(totalseg_total_dir, "heart_myocardium.nii.gz", ref_shape)
    _add_neg(myo)

    # Aorta: heartchambers_highres > total
    aorta = _try_struct(totalseg_heart_dir, "aorta.nii.gz", ref_shape)
    if aorta is None:
        aorta = _try_struct(totalseg_total_dir, "aorta.nii.gz", ref_shape)
    _add_neg(aorta)

    # Pulmonary artery: heartchambers_highres > total
    pa = _try_struct(totalseg_heart_dir, "pulmonary_artery.nii.gz", ref_shape)
    if pa is None:
        pa = _try_struct(totalseg_total_dir, "pulmonary_artery.nii.gz", ref_shape)
    _add_neg(pa)

    # Coronary arteries: dedicated task (507) > total
    coronary = _try_struct(totalseg_coronary_dir, "coronary_arteries.nii.gz", ref_shape)
    if coronary is None:
        coronary = _try_struct(totalseg_total_dir, "coronary_arteries.nii.gz", ref_shape)
    _add_neg(coronary)

    # Cusps from dedicated coronary task
    for fname in ("right_coronary_cusp.nii.gz", "left_coronary_cusp.nii.gz", "non_coronary_cusp.nii.gz"):
        _add_neg(_try_struct(totalseg_coronary_dir, fname, ref_shape))

    # Pulmonary veins: VISTA3D (119) > total
    if vista3d_combined_path is not None and vista3d_combined_path.exists():
        pv_v = _load_label(vista3d_combined_path, VISTA3D_LABEL_IDS["pulmonary_vein"])
        if pv_v.shape == ref_shape:
            _add_neg(pv_v)
    pv_total = _try_struct(totalseg_total_dir, "pulmonary_vein.nii.gz", ref_shape)
    _add_neg(pv_total)

    # Lungs: VISTA3D lobes (28-32) > total
    if vista3d_combined_path is not None and vista3d_combined_path.exists():
        lung_ids = [VISTA3D_LABEL_IDS[k] for k in ("left_lung_upper","left_lung_lower","right_lung_upper","right_lung_mid","right_lung_lower")]
        lung_v = _load_labels_union(vista3d_combined_path, lung_ids)
        if lung_v.shape == ref_shape:
            _add_neg(lung_v)
    for fname in ("lung_upper_lobe_left.nii.gz","lung_lower_lobe_left.nii.gz",
                  "lung_upper_lobe_right.nii.gz","lung_middle_lobe_right.nii.gz","lung_lower_lobe_right.nii.gz"):
        _add_neg(_try_struct(totalseg_total_dir, fname, ref_shape))

    # Other total-task fallbacks (bones, pericardium)
    for fname in ("pericardium.nii.gz","vertebrae_T1.nii.gz","vertebrae_T2.nii.gz",
                  "vertebrae_T3.nii.gz","vertebrae_T4.nii.gz","sternum.nii.gz",
                  "rib_left_1.nii.gz","rib_right_1.nii.gz"):
        _add_neg(_try_struct(totalseg_total_dir, fname, ref_shape))

    # Strip overlap with positive prior (positive takes precedence)
    neg = np.where(pos.astype(bool), np.uint8(0), neg).astype(np.uint8)
    result.negative_prior = neg

    # --- distance transform from negative boundary ---
    if neg.sum() > 0:
        spacing = _get_voxel_spacing(affine)
        result.negative_distance = distance_transform_edt(
            1 - neg, sampling=spacing
        ).astype(np.float32)

    return result


def save_fusion_outputs(result: PriorFusionResult, out_dir: Path) -> dict[str, Path]:
    """Save all arrays in result as NIfTI files. Returns dict of name -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = result.affine
    saved = {}

    def _save(arr: Optional[np.ndarray], name: str, dtype=None):
        if arr is None:
            return
        if dtype is not None:
            arr = arr.astype(dtype)
        path = out_dir / f"{result.case_id}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), str(path))
        saved[name] = path

    _save(result.nudf_laa,         "nudf_laa",            np.uint8)
    _save(result.vista3d_laa,      "vista3d_laa",         np.uint8)
    _save(result.totalseg_laa,     "totalseg_laa",        np.uint8)
    _save(result.union_laa,        "union_laa",           np.uint8)
    _save(result.intersection_laa, "intersection_laa",    np.uint8)
    _save(result.consensus_laa,    "consensus_laa",       np.uint8)
    _save(result.disagreement_map, "disagreement_map",    np.float32)
    _save(result.positive_prior,   "positive_prior",      np.uint8)
    _save(result.negative_prior,   "negative_prior",      np.uint8)
    _save(result.negative_distance,"negative_distance_mm",np.float32)

    summary_path = out_dir / f"{result.case_id}_prior_fusion_summary.json"
    summary_path.write_text(json.dumps(result.summary(), indent=2))
    saved["summary_json"] = summary_path

    return saved
