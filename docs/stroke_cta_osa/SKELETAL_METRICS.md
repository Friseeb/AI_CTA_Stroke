# Skeletal / hyoid geometry metrics

The skeletal module
([stroke_cta_osa/skeletal.py](../../subprojects/stroke-cta-osa/stroke_cta_osa/skeletal.py))
computes hyoid-anchored distances, neck-length proxies, and a cervicomandibular
ring area. All distances are in physical mm; the module operates on the
landmark bundle plus optional airway and mandible inputs.

## Inputs

| Argument | Used for |
|---|---|
| `landmarks` | hyoid centroid, C2/C3/C4, PNS, epiglottis tip, menton |
| `airway` (optional) | hyoid → airway posterior wall distance |
| `mandible_mask` (optional) | cervicomandibular ring area |
| `mandibular_plane_to_hyoid_distance_mm` (optional) | forwarded from the mandible module so both modules can read it |

## Per-point provenance

`hyoid_detected` is the only boolean flag — it goes True only when the
landmark bundle carries `hyoid_centroid` with a populated `physical_mm`. The
three coordinate columns (`hyoid_centroid_x/y/z_mm`) are populated from the
same point.

## Distances

| Column | Definition |
|---|---|
| `hyoid_to_c2_distance_mm` | Euclidean distance, hyoid_centroid → c2_centroid |
| `hyoid_to_c3_distance_mm` | analogous |
| `hyoid_to_c4_distance_mm` | analogous |
| `hyoid_to_epiglottis_distance_mm` | analogous |
| `hard_palate_to_hyoid_distance_mm` | posterior_nasal_spine → hyoid_centroid |
| `posterior_nasal_spine_to_epiglottis_distance_mm` | PNS → epiglottis_tip |

Every distance falls back to NaN when either landmark is missing.

## Positions relative to other landmarks

* `hyoid_vertical_position_relative_to_mandible_mm` — signed difference
  `hyoid_z - menton_z` in physical mm. Negative usually means the hyoid is
  superior to the menton (depends on patient orientation).
* `hyoid_ap_position_relative_to_cervical_spine_mm` — signed
  `hyoid_y - c3_centroid_y`. Negative usually means anterior to C3.

## Derived neck features

* `neck_length_mm` — alias of `hard_palate_to_hyoid_distance_mm` so consumers
  reading neck length don't have to know about the underlying landmarks.
* `laryngeal_descent_mm` — alias of `hyoid_to_c4_distance_mm` for the same
  reason.

## Cervicomandibular ring

Coarse axial-plane proxy: take the mandible mask's inferior 1/10 in z, then
the bounding-box area (`y_span × x_span`) of those voxels. Method string:
`mandible_inferior_bbox_proxy`. NaN when no mandible mask is supplied.

## Hyoid → posterior pharyngeal wall

When the airway mask is present, we use the slice at the hyoid's z. The
posterior wall is the airway's `max(y)` at that slice; the distance is
`|hyoid_y − airway_max_y| × spacing_y`. NaN when the airway slice is empty.

## Forwarded mandibular plane distance

If the caller already computed `mandibular_plane_to_hyoid_distance_mm` in the
mandible module, it's surfaced here too so both rows hold the same value
without re-deriving the plane.
