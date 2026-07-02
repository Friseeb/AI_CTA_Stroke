# Soft-tissue metrics (soft palate, uvula, palatine tonsils, lateral pharyngeal wall)

These features come from the soft-palate module
([stroke_cta_osa/soft_palate.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/soft_palate.py)).
Each block is *independently optional* — if the user supplies only a soft-palate
mask, only the soft-palate columns populate.

## Inputs

| Input | Used for |
|---|---|
| `soft_palate_mask` | volume, mean HU, length, thickness max / mean |
| `uvula_mask` | volume + length / width |
| `palatine_tonsil_left_mask`, `palatine_tonsil_right_mask` | per-side volumes + total |
| `landmarks` (PNS, uvula_tip, retropalatal_level) | landmark length fallback for soft palate + lateral wall anchor |
| `airway` + `body_mask` | lateral pharyngeal wall thickness |

## Soft palate

* **Length** — z extent of the mask in mm. Fallback: physical distance from
  `posterior_nasal_spine` to `uvula_tip` when no mask is available and
  `allow_landmark_length_fallback=True`.
* **Thickness (max, mean)** — per-slice maximum and mean of the mask's y
  extent.
* **Volume** — voxel count × voxel volume.
* **Mean HU** — over the mask.
* **`soft_palate_inferior_tip_z_mm`** — the most inferior slice's physical z.

## Uvula

* **Volume** — from mask.
* **Length** — z extent.
* **Width** — x extent.

## Palatine tonsils

* **`palatine_tonsil_left_volume_ml`** / `_right_volume_ml` — independent
  per-side volumes from supplied masks.
* **`palatine_tonsil_total_volume_ml`** — sum of L and R if either side is
  present.
* **`tonsil_to_retropalatal_airway_ratio`** — populated downstream when the
  airway area at the retropalatal level is known.

## Lateral pharyngeal wall thickness

Requires an airway mask and a body mask. For each axial slice in a window
around the retropalatal (or retroglossal) level, the algorithm:

1. takes the slice's airway L/R extreme x at the y-midline;
2. sweeps outward toward the body silhouette, counting body voxels along the
   sweep;
3. converts the count to mm by multiplying by `spacing_x`.

The reported thickness is the **median** of those per-slice sweeps, separately
for left and right.

* `lateral_pharyngeal_wall_left_thickness_mm`
* `lateral_pharyngeal_wall_right_thickness_mm`
* `lateral_pharyngeal_wall_mean_thickness_mm` — mean over both sides.
* `lateral_pharyngeal_wall_asymmetry_index` — `(R - L) / (R + L)` in
  `[-1, +1]`. Positive = right thicker than left.

The window length is `lateral_wall_axial_window_mm` (default 20 mm); the
lateral search band is `lateral_wall_band_mm` (default 15 mm).

## Why not segment these from scratch

Soft-palate / uvula / tonsil masks need careful annotation under contrast and
benefit from oral landmarks that CTA-only DL detectors don't have. This
module exists to consume masks coming from a dental/CBCT pipeline or manual
QA, not to produce them.
