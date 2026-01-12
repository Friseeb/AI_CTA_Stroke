# Quick Start: Antiga 2008 Centerline Extraction Pipeline

## 30-Second Overview

We've built a **modular Python pipeline** that extracts centerlines from vessel segmentations using the Antiga et al. (2008) methodology.

- **7 independent stages** (surface extraction → Eikonal path tracing → graph construction)
- **Unified orchestrator** (`CenterlineExtractionPipeline` class)
- **Multiple output formats** (NetworkX pickle + JSON)
- **Integrated with batch processing** (manifest-based preprocessing)

---

## Installation & Setup

### 1. Verify Dependencies
```bash
pip install nibabel networkx scipy numpy
```

### 2. Confirm Package Structure
```
AI_CTA_Stroke/python/analysis/centerline_antiga_2008/
├── __init__.py
├── pipeline.py
├── stage1_surface_extraction.py
├── stage2_extremal_points.py
├── stage4_eikonal.py
├── stage5_radius.py
├── stage6_bifurcations.py
└── stage7_graph.py
```

---

## Usage: 3 Ways

### 💡 Option 1: One-Liner (Recommended for Most Users)

```python
from python.analysis.centerline_antiga_2008 import CenterlineExtractionPipeline

# Run full pipeline (all stages)
pipeline = CenterlineExtractionPipeline('vessel_mask.nii.gz', 'output/')
results = pipeline.run()
```

**Output:**
- `output/centerline.pkl` — NetworkX graph
- `output/centerline_nodes.json` — Node attributes
- `output/centerline_edges.json` — Edge attributes
- `output/pipeline.log` — Detailed execution log

---

### 💡 Option 2: Batch Processing (Multiple Cases)

```bash
python python/cli/run_centerline_batch.py \
  --manifest data/manifests/cta_inputs.csv \
  --method vmtk \
  --output-root outputs/centerlines/
```

**Manifest format** (CSV):
```csv
case_id,nifti_path
case_001,path/to/vessel_001.nii.gz
case_002,path/to/vessel_002.nii.gz
```

---

### 💡 Option 3: Individual Stages (Advanced)

```python
from python.analysis.centerline_antiga_2008 import (
    extract_surface,
    detect_extremal_points,
    extract_centerlines_via_eikonal,
    compute_radii,
)

# Load your binary vessel mask
import nibabel as nib
img = nib.load('vessel_mask.nii.gz')
mask = (img.get_fdata() > 0).astype('uint8')

# Stage 1: Surface extraction
s1 = extract_surface(mask)
cleaned = s1['cleaned_mask']
distance_map = s1['distance_map']

# Stage 2: Extremal points
s2 = detect_extremal_points(cleaned, distance_map)
extremal_pts = s2['extremal_points']

# Stage 4: Path tracing
s4 = extract_centerlines_via_eikonal(distance_map, extremal_pts)
centerlines = s4['centerlines']

# Stage 5: Radii
s5 = compute_radii(centerlines, distance_map)
```

---

## Reading the Output

### NetworkX Graph (`.pkl`)

```python
import pickle

with open('output/centerline.pkl', 'rb') as f:
    graph = pickle.load(f)

# Get number of centerline points
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")

# Iterate centerline segments
for node_id in graph.nodes():
    data = graph.nodes[node_id]
    pos = data['position']      # [x, y, z]
    radius = data['radius']
    seg_id = data['segment_id']
    print(f"Node {node_id}: pos={pos}, radius={radius:.2f}mm, segment={seg_id}")
```

### Node/Edge JSON

```python
import json

# Load node attributes
with open('output/centerline_nodes.json') as f:
    nodes = json.load(f)

# Load edge attributes
with open('output/centerline_edges.json') as f:
    edges = json.load(f)

print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")

# Print first node
print(json.dumps(nodes[0], indent=2))
```

---

## Summary Statistics

Access pipeline summary after execution:

```python
summary = pipeline.summary()

print(f"Components: {summary['num_components']}")
print(f"Extremal points: {summary['num_extremal_points']}")
print(f"Centerlines: {summary['num_centerlines']}")
print(f"Bifurcations: {summary['num_bifurcations']}")
print(f"Total length: {summary['total_centerline_length_mm']:.1f} mm")
```

---

## Common Parameters

### Stage 1 Parameters
```python
pipeline.run(
    min_component_size=50,      # Min voxels per component (default: 50)
    erosion_iterations=1,       # Morphological erosion (default: 1)
    dilation_iterations=1,      # Morphological dilation (default: 1)
)
```

### Stage 4 Parameters
```python
pipeline.run(
    step_size=0.1,              # Gradient descent step (mm, default: 0.1)
    max_iterations=5000,        # Max gradient descent steps (default: 5000)
)
```

### Stage 6 Parameters
```python
pipeline.run(
    contact_distance_threshold=1.0,  # Bifurcation detection threshold (mm, default: 1.0)
)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **No centerlines detected** | Lower `min_component_size` (try 25) or verify vessel mask quality |
| **Too few centerlines** | Increase `max_iterations` (try 10000) or decrease `step_size` (try 0.05) |
| **Many spurious branches** | Increase `min_component_size` or apply stronger morphological closing |
| **Pipeline crashes** | Check `output/pipeline.log` for detailed error messages |
| **Out of memory** | Process smaller images or reduce volume dimensions |

---

## Performance Tips

### Speed Up
```python
pipeline.run(
    min_component_size=100,     # Skip small components
    step_size=0.2,              # Larger gradient steps
    max_iterations=2000,        # Fewer iterations
)
```

### Higher Quality
```python
pipeline.run(
    min_component_size=25,      # Keep small vessels
    step_size=0.05,             # Finer resolution
    max_iterations=10000,       # More iterations
)
```

---

## Next Steps: Downstream Tools

Once centerlines are extracted:

### Option A: Use with EVC (Vessel Labeling)
```python
# TODO: Create converter
from python.analysis.converters import graph_to_evc_format
evc_data = graph_to_evc_format(graph)
```

### Option B: Use with ArterialGNet (Feature Learning)
```python
# TODO: Create converter
from python.analysis.converters import graph_to_arterial_gnet_format
gnet_data = graph_to_arterial_gnet_format(graph)
```

---

## Testing

Run validation tests:
```bash
python python/analysis/centerline_antiga_2008/test_pipeline.py
```

Output:
```
============================================================
ANTIGA 2008 CENTERLINE EXTRACTION TEST SUITE
============================================================
Testing Stage 1: Surface Extraction
✓ Stage 1 passed. Components: 1
Testing Stage 2: Extremal Points Detection
✓ Stage 2 passed. Extremal points: 8
...
ALL TESTS PASSED ✓
```

---

## Full Documentation

For detailed information, see:
- **Module Documentation:** [README.md](centerline_antiga_2008/README.md)
- **Implementation Summary:** [IMPLEMENTATION_SUMMARY_Antiga2008_Pipeline.md](IMPLEMENTATION_SUMMARY_Antiga2008_Pipeline.md)
- **Example Code:** `centerline_antiga_2008/example_usage.py`

---

## Questions?

Refer to:
1. **Stage-specific docstrings** in module files
2. **Pipeline.log** for execution details
3. **test_pipeline.py** for usage examples
4. **README.md** in centerline_antiga_2008/ for comprehensive docs

---

**Happy centerline extraction! 🚀**
