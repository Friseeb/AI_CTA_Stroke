# Landmarks

The landmarks module
([stroke_cta_osa/landmarks.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/landmarks.py))
canonicalises the small set of points, z-levels, and planes that anchor every
region-specific feature in the pipeline. It is the only place where
voxel↔physical conversion happens, so downstream modules can operate in either
frame without re-implementing affine math.

## Schema

[stroke_cta_osa/landmark_schema.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/landmark_schema.py)
defines three kinds of landmarks:

* **Points** — `(z, y, x)` voxel index plus optional physical mm position.
* **Z-levels** — a single axial slice index marking an anatomical level
  (e.g. `retropalatal_level`).
* **Planes** — physical-mm representations of anatomical planes, either as
  `(point, normal)` or as three point references.

The canonical names live as module-level tuples
(`POINT_LANDMARKS`, `Z_LEVEL_LANDMARKS`, `PLANE_LANDMARKS`). The loader
silently drops unknown names so consumers can write forward-compatible JSON.

## Provider chain

`build_landmark_bundle()` runs four providers in priority order. Earlier
providers always win; later providers only fill slots the earlier ones left
empty.

1. **Explicit JSON** — user-supplied bundle file via `--landmarks`.
2. **Dental adapter** — sibling-pipeline output via `--dental-landmarks`.
3. **Heuristic from airway** — conservative estimator that only fires when the
   airway is present and tall enough (≥ 60 voxels and ≥ 60 mm extent).
4. **Empty bundle** — every landmark missing; features depending on them
   become NaN.

The heuristic estimator populates only:

* `retroglossal_level` ≈ z of the airway's minimum CSA, provided that minimum
  sits in the lower 2/3 of the airway extent;
* `tongue_base_level` ≈ retroglossal_level + 5 mm;
* `hard_palate_plane` ≈ z of the widest CSA in the upper 1/3 of the airway.

Each carries `source='heuristic_airway'` and a low `confidence` value so
downstream consumers can filter.

## File contract

A bundle on disk is JSON with this shape:

```json
{
  "case_id": "stu_001",
  "coord_system": "voxel_zyx",
  "image_shape_zyx": [80, 80, 80],
  "image_affine": null,
  "points": {
    "hyoid_centroid": {
      "voxel_zyx": [20, 44, 40],
      "physical_mm": null,
      "source": "external_json",
      "confidence": 0.9
    }
  },
  "z_levels": {
    "retropalatal_level": {"z_voxel": 55, "z_physical_mm": null,
                           "source": "external_json", "confidence": 0.8}
  },
  "planes": {
    "mandibular_plane": {
      "point_names": ["menton", "gonion_left", "gonion_right"],
      "point_phys_mm": null,
      "normal_phys_mm": null,
      "source": "external_json"
    }
  }
}
```

Either `voxel_zyx` or `physical_mm` can be set on a point. `fill_physical_coords()`
populates the missing one from the image affine.

## Validation

`validate_landmarks(bundle, image)` returns a list of human-readable warnings
without raising. It checks:

* image_shape_zyx in the bundle matches the loaded image;
* every point has at least one of `voxel_zyx` / `physical_mm`;
* voxel indices lie inside the image volume;
* z-levels lie in `[0, sz)`;
* planes have a usable representation;
* planes that reference named points only reference points present in the
  bundle.

`validate-landmarks` (CLI subcommand) wraps this with the canonical schema
and prints warnings + counts of populated landmarks per kind.

## Coordinate transforms

* `voxel_to_physical(image, (z, y, x)) → (x, y, z)` in physical mm — applies
  the image origin, spacing and direction (matches SimpleITK's
  `TransformIndexToPhysicalPoint`).
* `_physical_to_voxel(image, (x, y, z)) → (z, y, x)` voxel index. Solves the
  3×3 linear system for general direction matrices; falls back to identity
  on singularity.
* `transform_landmarks_between_image_spaces(bundle, source, target)` moves a
  bundle from one sampling grid to another sharing the same physical frame.
  Points and z-levels round-trip via physical mm; planes pass through
  unchanged (they're already in physical mm).

## What we deliberately don't do

* **No automatic landmark detection.** Landmarks are user-provided, from the
  dental adapter, or from the conservative airway heuristic above.
* **No clinical scoring.** Landmark distances feed downstream features;
  they're not interpreted as anatomic norms themselves.
