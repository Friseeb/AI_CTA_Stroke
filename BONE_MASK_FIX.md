# Bone Mask Fix: Including Skull, Ribs, Sternum, and Individual Vertebrae

## Issue
The bone mask was incomplete - it was missing:
- **Vertebrae**: TotalSegmentator outputs individual vertebrae files (vertebrae_C1.nii.gz, vertebrae_T1.nii.gz, etc.) instead of a single "vertebrae.nii.gz" file
- **Skull, ribs, and sternum**: Not included in this CTA scan's TotalSegmentator output (likely outside the field of view)

## Root Cause
The original `build_masks_from_totalseg.py` and `total_segmentator.py` scripts were looking for:
```python
BONE_STRUCTURES = [
    "skull",
    "vertebrae",  # ❌ This file doesn't exist
    "rib_left",   # ❌ Not available in this CTA
    "rib_right",  # ❌ Not available in this CTA
    "sternum",    # ❌ Not available in this CTA
]
```

But TotalSegmentator actually outputs:
- Individual vertebrae: `vertebrae_C1.nii.gz`, `vertebrae_C2.nii.gz`, ... `vertebrae_L5.nii.gz`, `vertebrae_S1.nii.gz`
- No skull, ribs, or sternum (they're outside the CTA field of view)

## Solution

### Changes Made
Updated both scripts to:
1. **Support glob patterns** for matching multiple vertebrae files:
   ```python
   BONE_STRUCTURES = [
       "skull",
       "vertebrae_*.nii.gz",  # ✅ Matches all individual vertebrae
       "sacrum",               # ✅ Added sacrum (was missing)
       "rib_left",
       "rib_right",
       "sternum",
   ]
   ```

2. **Enhanced `load_structure_mask()` function** to handle glob patterns and combine multiple files

3. **Added diagnostic logging** to show which structures were found and which were skipped:
   ```
   ✓ Found structures: vertebrae_*.nii.gz, sacrum
   ⚠ Missing structures (skipped): skull, rib_left, rib_right, sternum
   ```

### Updated Files
- `/scripts/build_masks_from_totalseg.py`
- `/scripts/total_segmentator.py`

## Bone Mask Contents (After Fix)
The bone mask now includes:
- ✅ **All vertebrae**: C1-C7, T1-T12, L1-L5, S1 (combined from individual files)
- ✅ **Sacrum**: S1 vertebra
- ❌ **Skull**: Not available in CTA field of view
- ❌ **Ribs (left/right)**: Not available in CTA field of view
- ❌ **Sternum**: Not available in CTA field of view

## Why Some Structures Are Missing

The CTA scan's **field of view (FOV) likely focuses on the thorax/abdomen** and doesn't include:
- **Skull**: Too superior (head region)
- **Ribs/Sternum**: May be partially in FOV but not segmented by TotalSegmentator in this case

This is normal for CTA protocols which are typically targeted scans.

## Running the Scripts

To regenerate masks with the fix:

```bash
# Option 1: From existing TotalSegmentator output
python scripts/build_masks_from_totalseg.py \
  --totalseg-dir outputs/segmentator \
  --output-bone outputs/bone_mask.nii.gz \
  --output-vessel outputs/vessel_mask.nii.gz

# Option 2: Run TotalSegmentator + build masks in one call
python scripts/total_segmentator.py \
  --input data/cta.nii.gz \
  --out-dir outputs/segmentator \
  --bone-mask outputs/bone_mask.nii.gz \
  --vessel-mask outputs/vessel_mask.nii.gz
```

## To Include Skull/Ribs/Sternum

If you need skull, ribs, or sternum in your bone mask:

1. **Check if your CTA FOV includes these structures** - visualize the CTA in a viewer
2. **Re-run TotalSegmentator with explicit ROI selection**:
   ```bash
   python scripts/total_segmentator.py \
     --input data/cta.nii.gz \
     --out-dir outputs/segmentator \
     --bone-mask outputs/bone_mask.nii.gz \
     --vessel-mask outputs/vessel_mask.nii.gz \
     --roi-subset skull,vertebrae_C1,vertebrae_C2,vertebrae_C3,vertebrae_C4,vertebrae_C5,vertebrae_C6,vertebrae_C7,vertebrae_T1,vertebrae_T2,vertebrae_T3,vertebrae_T4,vertebrae_T5,vertebrae_T6,vertebrae_T7,vertebrae_T8,vertebrae_T9,vertebrae_T10,vertebrae_T11,vertebrae_T12,vertebrae_L1,vertebrae_L2,vertebrae_L3,vertebrae_L4,vertebrae_L5,sacrum,rib_left,rib_right,sternum
   ```

3. **Update BONE_STRUCTURES list** if using manual naming conventions
