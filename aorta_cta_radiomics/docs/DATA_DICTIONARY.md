# Data dictionary: vessel-centred CTA perivascular phenotyping

## Conventions

- Case IDs are de-identified strings. Patient identifiers, names, accession numbers, and dates are not output features.
- NIfTI arrays use nibabel `(i, j, k)` index order. Physical coordinates use RAS millimetres unless an object explicitly declares another convention.
- FLOWCAT's `centerline_segments_array.npy` uses translation-normalized LPI-corner-relative millimetres. It must be converted through the adapter and checked against the lumen before it can be called physical CTA space.
- Distances and lengths are millimetres; areas are mm²; volumes are mm³; attenuation is HU; curvature and torsion are mm⁻¹.
- Missing values mean unavailable or non-assessable. They are not zero. Label `14` means uncertain/conflicting artery assignment, not a named artery.
- Radius is `MaximumInscribedSphereRadius` when supplied by FLOWCAT/VMTK. Diameter derived as `2 * radius` is labelled explicitly and is not a Feret, NASCET, ECST, area-equivalent, or histological outer-wall diameter.
- `other_nonfat_soft_tissue` is a residual CTA compartment and must not be renamed arterial wall.

## Volumetric label dictionaries

### Anatomical artery volume

| Value | Name | Definition |
|---:|---|---|
| 0 | background | Outside the arterial lumen. |
| 1 | aorta / AA | Aorta or FLOWCAT aortic-arch class. |
| 2 | brachiocephalic / BT | Brachiocephalic artery. |
| 3 | RCCA | Right common carotid artery. |
| 4 | LCCA | Left common carotid artery. |
| 5 | RSA | Right subclavian artery. |
| 6 | LSA | Left subclavian artery. |
| 7 | RVA | Right vertebral artery. |
| 8 | LVA | Left vertebral artery. |
| 9 | RICA | Right internal carotid artery. |
| 10 | LICA | Left internal carotid artery. |
| 11 | RECA | Right external carotid artery. |
| 12 | LECA | Left external carotid artery. |
| 13 | other artery | FLOWCAT `other` and basilar artery when the local 0–14 output convention has no separate BA value. Source label remains in the graph manifest. |
| 14 | uncertain/conflicting | Bifurcation exclusion, method disagreement, inadequate confidence, or incompatible assignment. |

The operational carotid bulb is not a separate FLOWCAT class. It is a configured longitudinal window spanning the distal CCA/bifurcation/proximal ICA and is reported with its derivation and bifurcation exclusions.

### Perivascular tissue composition

| Value | Name | Definition |
|---:|---|---|
| 0 | outside/background | Outside the evaluated envelope. |
| 1 | target artery | Target lumen; excluded from the external tissue denominator. |
| 2 | other artery | Non-target arterial lumen. |
| 3 | vein | Venous mask, including internal jugular when available. |
| 4 | bone | Bone exclusion mask. |
| 5 | airway | Air-containing airway mask. |
| 6 | thyroid/gland | Thyroid or configured gland mask. |
| 7 | skeletal muscle | Named muscle when available, otherwise configured combined muscle. |
| 8 | adipose tissue | CTA voxels inside the configured fat HU window after higher-priority exclusions. |
| 9 | other non-fat soft tissue | Residual assessable soft tissue after explicit classes. |
| 10 | high-density contamination | Configured high-attenuation/blooming/artifact candidate. |
| 11 | uncertain/non-assessable | Missing context, artifact, non-finite data, unresolved overlap, or configured exclusion. |

The deterministic priority is stored in `config/carotid_perivascular.yaml` and written to provenance. Sensitivity runs with another hierarchy receive a different configuration hash.

## Case-level output

Primary file: `case_summary.json`.

| Field | Type | Unit/values | Meaning |
|---|---|---|---|
| `case_id` | string | de-identified | Stable case identifier. |
| `cta_path` | string | path | Processing path; sharing/export policy may redact it. |
| `input_sha256` | string/null | SHA-256 | Input integrity digest. |
| `cta_metadata` | object | DICOM/NIfTI metadata | Permitted acquisition metadata with missingness explicit. |
| `original_spacing_xyz_mm` | 3 floats | mm | Native voxel spacing. |
| `original_dimensions_ijk` | 3 integers | voxels | Native image matrix. |
| `orientation` | 3 strings | axis codes | NIfTI orientation codes such as LPS or RAS. |
| `affine` | 4×4 floats | index-to-RAS mm | Original NIfTI affine. |
| `available_vessel_segments` | array | vessel codes | Anatomical segments actually available after QC. |
| `software_versions` | object | versions | Pipeline and dependency versions. |
| `model_versions` | object | versions/hashes | Model identifiers and checkpoint hashes. |
| `flowcat_repository` | string | URL | FLOWCAT upstream. |
| `flowcat_branch` | string | Git branch | Expected `models/2026-03`. |
| `flowcat_commit` | string | Git SHA | Exact checked-out commit. |
| `configuration_sha256` | string | SHA-256 | Canonical configuration digest. |
| `processing_status` | string | status enum | Completed, partial, failed, or visual-pending state. |
| `start_time`, `end_time` | string | ISO 8601 UTC | Processing bounds. |
| `processing_duration_seconds` | float | s | Wall time. |
| `cpu_information` | string | text | CPU/architecture. |
| `gpu_information` | string/null | text | GPU/device, if used. |
| `peak_ram_bytes` | integer/null | bytes | Peak resident memory when measured. |
| `peak_gpu_memory_bytes` | integer/null | bytes | Peak accelerator memory when measured. |
| `global_qc_flags` | array | flag codes | Automated and visual QC flags. |
| `failed_modules` | array | module names | Failed stages; never silently omitted. |
| `missing_outputs` | array | output names | Required or optional absent products. |

## Vessel-level output

Primary file: `vessel_level_features.csv`. One row per anatomical vessel/branch; bulb windows may be additional explicitly derived rows.

| Field group | Fields | Unit/meaning |
|---|---|---|
| Identity | `case_id`, `vessel_name`, `laterality`, `branch_id` | De-identified case and graph identity. |
| Label certainty | `vessel_label_confidence` | Probability/score in [0,1] when available. |
| Coverage | `total_length_mm`, `assessable_length_mm`, `nonassessable_length_mm` | Centreline lengths. |
| Radius | `mean_radius_mm`, `minimum_radius_mm`, `maximum_radius_mm`, `radius_method` | Explicit radius method. |
| Diameter | `diameter_method` | Method label; numerical local profile remains at sample level. |
| Geometry | `tortuosity`, `mean_curvature_per_mm` | Secondary lumen geometry. |
| Calcium | `calcium_burden_mm3` | Vessel-local high-attenuation calcium candidate with ambiguity/QC retained elsewhere. |
| Envelope | `total_shell_volume_mm3`, `valid_shell_volume_mm3` | Raw versus valid denominator. |
| Fat | `pvat_volume_mm3`, `pvat_fraction`, `mean_pvat_attenuation_hu` | CTA fat-range candidate; physiological PVAT qualification depends on QC. |
| Residual tissue | `perivascular_soft_tissue_volume_mm3`, `other_soft_tissue_fraction` | Residual non-fat soft tissue, not arterial wall. |
| Contact | `muscle_contact_fraction`, `vein_contact_fraction`, `bone_contact_fraction`, `thyroid_contact_fraction` | Fraction of sampled arterial surface at configured contact distance. |
| Uncertainty | `uncertain_fraction`, `qc_flags`, `manual_review_required` | Missingness/quality state. |
| Profiles | `radial_gradients`, `longitudinal_summaries`, `circumferential_summaries` | Serialized summaries; sample rows remain authoritative. |

Fractions use the valid denominator after target lumen and explicit non-assessable exclusions. Each feature record carries its radial band/contact distance either in its field name or nested profile metadata.

## Sample-point-level output

Primary file: `sample_point_features.parquet`; an explicit CSV fallback is permitted only when no Parquet engine is installed and is declared in `output_manifest.json`.

| Field | Type | Unit/values | Meaning |
|---|---|---|---|
| `case_id` | string | de-identified | Case. |
| `vessel_name` | string | vessel code | Anatomical identity. |
| `laterality` | string/null | left/right | Side. |
| `branch_id` | string | graph ID | Source branch. |
| `path_distance_mm` | float | mm | Cumulative distance along resampled centreline. |
| `physical_ras_x_mm`, `physical_ras_y_mm`, `physical_ras_z_mm` | floats | mm | Original CTA physical point. |
| `tangent_ras` | 3 floats | unit vector | Centreline tangent. |
| `first_normal_ras`, `second_normal_ras` | 3 floats | unit vectors | Stable parallel-transport frame. |
| `local_radius_mm` | float/null | mm | Local radius. |
| `radius_method` | string | method | Radius definition. |
| `local_diameter_mm` | float/null | mm | Explicitly derived local diameter. |
| `diameter_method` | string | method | Diameter definition. |
| `curvature_per_mm`, `torsion_per_mm` | floats/null | mm⁻¹ | Local geometry where numerically stable. |
| `distance_to_bifurcation_mm` | float/null | mm | Nearest branch point. |
| `radial_band_measurements` | object | values by band | Composition/attenuation by physical radial band. |
| `sector_measurements` | object | values by sector | Circumferential composition. |
| `surface_contact_measurements` | object | values by tissue/distance | Contact area/fraction/profile. |
| `calcium_measurements` | object | values | Calcium burden/proximity/ambiguity. |
| `tissue_confidence` | float/null | [0,1] | Aggregate tissue certainty. |
| `uncertain_fraction` | float/null | [0,1] | Uncertain/non-assessable fraction. |
| `qc_flags` | array | flag codes | Point/interval QC. |

## FLOWCAT output manifest

Each record includes `output_name`, `expected_path`, `actual_path`, `format`, `coordinate_convention`, `units`, `important_fields`, `currently_generated`, `currently_used_downstream`, `successfully_tested`, `known_problems`, SHA-256, byte size, and recommendation. The case manifest also records repository URL, branch, commit, model payload hashes, adapter version, trust policy, discovery time, and case status.

Pickle and object-array inputs are unsafe when untrusted. Loading requires explicit trusted-local opt-in, and the adapter then validates the NetworkX/object schema. A successful hash/schema check does not establish anatomical plausibility.

## Shell and sector records

Every shell record includes vessel identity, band name, inner/outer distance in mm, adaptive equation parameters when used, native spacing, interpolation policy, raw voxel count/volume, filtered voxel count/valid volume, exclusion counts/fractions, image-boundary truncation, bifurcation overlap, adjacent-artery overlap, and QC flags.

Every sector record includes vessel, branch, path position, radial band, sector count, sector index, angular bounds, absolute-or-relative orientation state, total and valid volume, class volumes/fractions, fat and CTA attenuation summaries/percentiles, valid voxel count, uncertainty, and QC state.

## QC flag namespace

Flags are stable lowercase snake-case codes. Core flags include `missing_arterial_mask`, `empty_arterial_mask`, `disconnected_arterial_components`, `missing_native_centerline`, `missing_anatomical_graph`, `centreline_outside_lumen`, `graph_label_conflict`, `left_right_conflict`, `implausible_radius`, `branch_truncation`, `coordinate_mismatch`, `affine_mismatch`, `shell_truncated_at_fov`, `adjacent_artery_overlap`, `bifurcation_contamination`, `excessive_vein_contamination`, `excessive_bone_contamination`, `insufficient_fat_voxels`, `calcium_blooming`, `dental_artifact`, `shoulder_artifact`, `low_intravascular_contrast`, `motion_artifact`, `inadequate_resolution`, `excessive_interpolation`, and `manual_review_required`.

## Output status vocabulary

- `complete_visual_pass`: all required outputs and documented visual pass.
- `automated_pass_visual_pending`: automated checks pass; reader review is required.
- `partial_missing_optional_context`: core named arterial outputs pass but one or more optional anatomy masks are absent and uncertainty is retained.
- `blocked_missing_native_flowcat_outputs`: branch/centreline/anatomical graph absent.
- `failed_integrity`: shape, affine, finite-value, hash, schema, or required-output failure.
- `development_fallback_not_anatomical`: skeleton/EDT or other technical fallback; never a named-vessel pass.
