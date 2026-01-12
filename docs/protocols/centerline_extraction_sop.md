# Centerline Extraction Pipeline

## Purpose
Extract vessel centerlines from CTA segmentations for segmental plaque analysis.

## Input
- 3D vessel segmentation masks (from ResNet model)
- CTA images in BIDS format

## Steps

1. **Preprocessing**
   - Load vessel mask
   - Apply morphological closing to fill small gaps
   - Remove isolated components (< 100 voxels)

2. **Skeletonization**
   - Compute 3D distance transform
   - Apply thinning algorithm preserving topology
   - Extract skeleton voxels

3. **Centerline Smoothing**
   - Fit B-spline curves to skeleton points
   - Resample to uniform spacing (0.5mm)
   - Validate smoothness (max curvature threshold)

4. **Territory Labeling**
   - Identify branch points
   - Label segments: CCA, ICA, ECA, VA
   - Store labels in JSON sidecar

5. **Cross-sectional Analysis**
   - Extract perpendicular planes along centerline
   - Measure lumen diameter and wall thickness
   - Quantify plaque burden per segment

## Software
```python
import SimpleITK as sitk
import vtk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d

def extract_centerline(vessel_mask):
    # Distance transform
    dist = distance_transform_edt(vessel_mask)
    
    # Skeletonization
    skeleton = skeletonize_3d(vessel_mask > 0)
    
    # Smooth with splines
    centerline = fit_spline(skeleton)
    
    return centerline
```

## Outputs
- `centerlines.nii.gz` - Binary centerline mask
- `centerlines.json` - Anatomical labels and metrics
- `plaque_burden_segmental.csv` - Plaque per segment

## Quality Control
- Visual overlay on CTA
- Check branch point detection
- Validate segment labels against atlas
