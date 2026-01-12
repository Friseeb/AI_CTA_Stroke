# Vascular Centerline Extraction Methods

## Overview

Centerline extraction is a fundamental preprocessing step for vascular analysis in CTA imaging. This document summarizes key methods from literature for extracting vessel centerlines and applying them to plaque burden quantification.

### Related GitHub (perecanals)
- Profile: https://github.com/perecanals
- Repos to review for vascular/stroke code or preprocessing routines:
  - https://github.com/perecanals/ICAD (multimodal stroke classification)
  - https://github.com/perecanals/EVC (CTA/vascular; check for vessel utilities)
  - https://github.com/perecanals/arterial_gnet (arterial access graph net)
  - https://github.com/perecanals/posterior_hypodensity_analysis (stroke imaging)
  - https://github.com/perecanals/DTFA (analysis notebooks/utilities)
  - https://github.com/perecanals/generic_classification_framework (training infra)

Action: inspect these for reusable centerline, vessel mask, or preprocessing code before reimplementing.

## Key Papers

### 1. Computerized Medical Imaging and Graphics (2022)
**Reference:** 1-s2.0-S0895611122001409-main-3.pdf

**Key Methods:**
- Skeleton-based centerline extraction from 3D vascular segmentations
- Topological analysis to preserve vessel connectivity
- Distance transform methods for centerline computation
- Applications to coronary and carotid artery analysis

**Relevance to Project:**
- Necessary for segmental plaque burden analysis
- Enables mapping plaque to specific arterial territories
- Foundation for automated calcium scoring adapted to cervical vessels

### 2. Medical & Biological Engineering & Computing (2008)
**Reference:** s11517-008-0420-1.pdf

**Key Methods:**
- Model-based centerline extraction
- Vessel tracking algorithms
- Cross-sectional analysis along centerline paths
- Stenosis quantification techniques

**Relevance to Project:**
- Automated measurement of vessel narrowing (stenosis)
- Enables territory-specific plaque burden metrics
- Critical for SPARKLE classification integration

## Implementation Strategy (VMTK-Based)

Following the EVC pipeline (Antiga et al., 2003-2008), centerline extraction is performed using the Vascular Modelling Toolkit (VMTK):

### Phase 1: Vessel Segmentation
- Use 3D ResNet to segment supraaortic vessels from CTA
- Output: Binary vessel masks (voxel-wise binary map)

### Phase 2: Surface Model Extraction
- Threshold binary segmentation map
- Smooth surface using Laplacian smoothing
- Remove small islands and noise
- Exclude intracranial arteries
- Output: Triangulated surface model (STL/VTK format)

### Phase 3: Centerline Extraction via Shortest Path Tracing
**Method:**
1. **Extremal Point Detection**: Automatically identify start/endpoints at vascular structure ends
2. **Voronoi Diagram Computation**: Generate Voronoi diagram from binary segmentation
3. **Eikonal Equation Optimization**: Compute shortest paths minimizing wave propagation integral
   - Eikonal equation: $|\nabla \phi| = 1$
   - Numerically solved via fast marching or level-set methods
   - Paths constrained to interior of vascular tubes (Voronoi skeleton)
4. **Maximal Inscribed Spheres**: Compute radius at each centerline point
5. **Branch Splitting**: Identify bifurcations using tube containment relationships

**Tools:**
- VMTK v1.4+ (Antiga et al., 2008)
- Custom modules for robust endpoint detection
- Circular centerline tracing for smooth paths

### Phase 4: Graph Generation
- Convert branched centerline model to graph structure
- **Nodes**: Centerlines of individual vascular segments
- **Edges**: Connect immediately proximal/distal segments at bifurcations
- Extract node features: position (x,y,z), radius, vessel attributes
- Output: Pickled NetworkX graph or JSON representation

### Phase 5: Territory Labeling
- Map centerline segments to anatomical territories:
  - Common carotid artery (CCA)
  - Internal carotid artery (ICA) - cervical, petrous, cavernous, supraclinoid
  - External carotid artery (ECA)
  - Vertebral artery (VA)
- Use graph neural networks (GNN) for automated vessel type classification
- Reference: EVC framework (perecanals/EVC repo)

### Phase 6: Plaque Quantification
- Extract cross-sectional measurements perpendicular to centerline
- Calculate plaque burden per segment
- Compute calcium scores using Agatston-like methods adapted from coronary literature

## Software Tools & Dependencies

**Core Centerline Pipeline:**
- **VMTK** (Vascular Modelling Toolkit): Centerline extraction, Voronoi diagrams, Eikonal solver
  - Installation: `pip install vmtk` or build from source (https://github.com/vmtk/vmtk)
  - Python bindings for automated workflows

**Supporting Tools:**
- **ITK-SNAP**: Manual segmentation ground truth and QC
- **SimpleITK**: Pre/post-processing (smoothing, island removal)
- **VTK**: Surface model visualization and export
- **NetworkX**: Graph construction and analysis
- **PyTorch**: 3D ResNet vessel segmentation backbone
- **Nibabel/NIfTI**: Medical imaging I/O

**Machine Learning (Graph-Based):**
- **PyTorch Geometric**: Graph neural networks for vessel labeling
- EVC framework (Graph U-Net for node classification)

## Expected Outputs

1. Vessel centerlines in BIDS format (JSON sidecar with anatomical labels)
2. Segmental plaque burden metrics (CSV)
3. Calcium scores per arterial segment
4. Visualization overlays (centerlines + plaque on CTA)

## Next Steps

- [ ] Implement skeletonization pipeline
- [ ] Validate centerline accuracy against manual tracings
- [ ] Develop automated territory labeling
- [ ] Integrate with ResNet segmentation outputs
- [ ] Create visualization tools for QC
