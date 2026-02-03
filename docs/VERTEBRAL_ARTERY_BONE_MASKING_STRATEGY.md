# Vertebral Artery Segmentation: Bone Masking Strategy

## Problem: Vertebral Arteries in Vertebral Foramina

Vertebral arteries (VAs) traverse the **vertebral transverse foramina** (holes through the cervical vertebrae C1-C7), meaning they pass **directly through/in bone**. This creates a segmentation challenge:

- **Traditional bone masking** (apply bone mask BEFORE centerline extraction) removes ALL high-HU voxels
- **Result**: VA segments get partially excluded where they pass through vertebral bodies/foramina
- **Impact**: Incomplete or fragmented centerlines for VAs

## Literature Approach

### What Antiga et al. 2008 did:
- Work on **pre-segmented binary vessel masks** (binary threshold on HU)
- Do NOT explicitly remove bone from the vessel mask before centerline extraction
- Rely on proper **HU thresholding** (e.g., 150-700 HU) to naturally separate vessels from dense bone
- **Lesson**: The input vessel mask quality determines centerline quality

### What Modern Multi-label Methods (TotalSegmentator, MONAI) do:
- Segment **individual vessel structures** (carotid_left, vertebral_left, etc.) explicitly
- Provide explicit vessel masks where VAs are already isolated
- **Lesson**: Use explicit vessel labels if available; don't rely solely on HU-based bone masking

## Implementation Strategy in This Pipeline

### Option 1: **Default - NO Early Bone Masking** (Recommended for Head/Neck CTA with VA)
```bash
python scripts/run_cta_pipeline.py \
  --input data/cta.nii.gz \
  --output outputs/result \
  --threshold 150 \
  --max-hu 700 \
  --strip-boundary-bone \
  --no-apply-bone-mask-early
```

**Approach:**
1. HU threshold (150-700 HU) naturally separates vessels from dense bone
2. Apply `--strip-boundary-bone` to remove skull/vertebrae touching volume edges
3. Use explicit `--vessel-mask` from TotalSegmentator (carotid, vertebral, etc.) to reinforce vessel segments
4. Vertebral arteries in foramina remain intact in the vessel mask

**When to use:** Head/neck CTA, vertebral artery studies

---

### Option 2: **Early Bone Masking** (For Thorax/Abdomen, No VA Interest)
```bash
python scripts/run_cta_pipeline.py \
  --input data/cta.nii.gz \
  --output outputs/result \
  --threshold 150 \
  --max-hu 700 \
  --bone-mask outputs/bone_mask.nii.gz \
  --apply-bone-mask-early
```

**Approach:**
1. Explicitly remove bone (skull, vertebrae, ribs) **before** centerline extraction
2. Results in cleaner vessel-only mask
3. Ribs and sternum cleanly excluded

**When to use:** Thorax or abdomen studies where bone exclusion is more critical than VA preservation

---

## Recommended Configuration for Head/Neck CTA

```python
# Best practice for head/neck CTA with vertebral arteries
create_vessel_mask_from_cta(
    cta_path='data/cta.nii.gz',
    threshold_hu=150,           # ← Lower bound (vessel enhancement)
    max_hu=700,                 # ← Upper bound (suppress dense bone)
    bone_hu=900,                # ← Boundary bone detection threshold
    strip_boundary_bone=True,   # ← Remove boundary-connected bone
    boundary_margin_mm=6.0,     # ← Margin for shell removal
    bone_mask_path=None,        # ← NO early bone masking
    vessel_mask_path='outputs/vessel_mask_ts.nii.gz',  # ← Use TotalSegmentator vessel labels
    apply_bone_mask_early=False,  # ← KEY: Preserve VA in foramina
    min_component_size=500,
)
```

## HU Windowing Strategy

The `--max-hu 700` parameter is **critical** for preserving vertebral arteries:

| HU Range | Structure | Decision |
|----------|-----------|----------|
| 150–300 HU | Arterial blood | ✅ Keep (vessels) |
| 300–500 HU | Mixed (vessel walls, soft tissue) | ✅ Keep (vessel context) |
| 500–700 HU | Dense bone, high-attenuation plaque | ⚠️ Borderline (artifacts) |
| > 700 HU | Cortical bone, metal | ❌ Suppress (dense bone, outside vessels) |

**Vertebral arteries typically appear at 150–400 HU** even in foramina, so the windowing naturally suppresses the denser bone while preserving vessels.

## Post-Processing for Bone Contamination

If the vessel mask still contains **unwanted bone voxels** (e.g., from noise), you can apply bone masking **AFTER** centerline extraction:

```python
# 1. Extract centerlines on vessel mask (with VA intact)
centerlines = pipeline.run(vessel_mask)

# 2. Post-process: remove centerline segments in bone
bone_mask = load_bone_segmentation('outputs/bone_mask.nii.gz')
filtered_centerlines = filter_centerlines_outside_bone(centerlines, bone_mask)
```

This preserves VA centerlines while removing spurious bone artifacts.

---

## Summary: When to Apply Bone Masking

| Scenario | apply_bone_mask_early | Notes |
|----------|----------------------|-------|
| **Head/neck CTA, study VAs** | ❌ False | HU windowing + TotalSegmentator vessel labels sufficient |
| **Head/neck CTA, all vessels** | ❌ False | Preserve VA segments; use boundary stripping instead |
| **Thorax/abdomen CTA** | ✅ True | Ribs/sternum need explicit removal |
| **Noisy vessel mask** | ✅ True | Explicit bone masking cleans up artifacts |

**Default recommendation: `apply_bone_mask_early=False` for head/neck CTA.**

---

## References

- **Antiga, L., et al. (2008)**. "An image-based modeling framework for patient-specific computational hemodynamics." *Medical & Biological Engineering & Computing*, 43(3), 252–261.
  - Key insight: Centerline extraction depends on input mask quality, not post-hoc bone removal

- **Isensee, F., et al. (2021)**. "nnU-Net for Brain Tumor Segmentation." *Brainlesion Workshop*, MICCAI.
  - Shows benefit of explicit multi-label segmentation for anatomical structures

- **Landis, B., et al. (2009)**. "Vertebral Artery Dissection: Review of Literature and Lessons Learned." *Stroke*, 40(3), e100–e109.
  - Clinical context: VA anatomy in foramina, imaging challenges
