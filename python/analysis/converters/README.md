# Centerline Graph Converters

## Overview

This module provides converters to transform centerline graphs from the Antiga 2008 pipeline into formats expected by downstream vessel analysis tools:

1. **EVC (Extracranial Vessel Classification)** - PyTorch Geometric node classification
2. **ArterialGNet** - Graph neural network vessel labeling and feature extraction

## Quick Start

### EVC Converter

```python
from python.analysis.converters import centerline_to_evc_graph
import pickle

# Load centerline graph from pipeline
with open('centerline.pkl', 'rb') as f:
    centerline_graph = pickle.load(f)

# Convert to EVC format
evc_graph = centerline_to_evc_graph(
    centerline_graph,
    output_pickle_path='evc_graph.pkl',
    vessel_type_name='RCCA',  # or 'other' if unknown
)

# Result: 16 bifurcation nodes, 28 vessel edges
# Vessels are EDGES with attributes: pos, features, vessel_type_name
```

### ArterialGNet Converter

```python
from python.analysis.converters import centerline_to_arterial_gnet_graph

# Convert to ArterialGNet format
gnet_result = centerline_to_arterial_gnet_graph(
    centerline_graph,
    output_pickle_path='arterial_gnet_graph.pkl',
    vessel_type_name='other',
    include_segment_graph=True,
)

dense_graph = gnet_result['dense_graph']  # All centerline points
segment_graph = gnet_result['segment_graph']  # Aggregated segments

# Result: 872 dense nodes, 55 segments
```

---

## EVC Format Details

### Input Requirements
- **Source**: NetworkX DiGraph from `stage7_graph.py`
- **Node attributes**: position, radius, segment_id, segment_index
- **Edge attributes**: distance, type

### Output Format
- **Graph type**: `nx.Graph` (undirected)
- **Nodes**: Bifurcation points (vessel endpoints/junctions)
- **Edges**: Vessel segments with attributes:
  - `pos`: np.array([x, y, z]) - mean position of segment
  - `features`: np.array([length, mean_radius, min_radius, max_radius, num_points])
  - `vessel_type_name`: str - one of 14 vessel labels (AA, BT, RCCA, etc.)
  - `vessel_type`: int (0-13) - vessel type ID

### Vessel Type Labels
```python
vessel_types = [
    'other',  # 0 - unknown/unlabeled
    'AA',     # 1 - Aortic Arch
    'BT',     # 2 - Brachiocephalic Trunk
    'RCCA',   # 3 - Right Common Carotid Artery
    'LCCA',   # 4 - Left Common Carotid Artery
    'RSA',    # 5 - Right Subclavian Artery
    'LSA',    # 6 - Left Subclavian Artery
    'RVA',    # 7 - Right Vertebral Artery
    'LVA',    # 8 - Left Vertebral Artery
    'RICA',   # 9 - Right Internal Carotid Artery
    'LICA',   # 10 - Left Internal Carotid Artery
    'RECA',   # 11 - Right External Carotid Artery
    'LECA',   # 12 - Left External Carotid Artery
    'BA',     # 13 - Basilar Artery
]
```

### Node Transform for Classification
EVC applies `node_transform()` to convert edges→nodes for node classification:
```python
from python.analysis.converters.evc_converter import apply_evc_node_transform

# Transform: vessel segments (edges) become nodes
transformed_graph = apply_evc_node_transform(evc_graph)

# Now vessels are NODES (not edges) for classification
# Nodes immediately neighboring at bifurcations are connected by edges
```

---

## ArterialGNet Format Details

### Output Structure
```python
{
    'dense_graph': nx.DiGraph,    # All centerline points as nodes
    'segment_graph': nx.DiGraph,  # Aggregated vessel segments (optional)
}
```

### Dense Graph
**Purpose**: Full-resolution centerline with all points

**Node attributes**:
- `pos`: np.array([x, y, z])
- `radius`: float (mm)
- `vessel_type_name`: str
- `vessel_type`: int (0-13)
- `segment_id`: str (from pipeline)
- `segment_index`: int (position within segment)

**Edge attributes**:
- `distance`: float (mm)
- `type`: str ('centerline' or 'bifurcation')

### Segment Graph (Optional)
**Purpose**: Higher-level aggregated representation

**Node attributes**:
- `pos`: np.array([x, y, z]) - mean position
- `radius`: float - mean radius
- `vessel_type_name`: str
- `vessel_type`: int
- `segment_id`: str
- `num_points`: int - points in segment
- `segment_length`: float (mm)

**Edge attributes**:
- `bifurcation_pos`: np.array([x, y, z]) - junction location

### Advanced Features
Add curvature, torsion, direction vectors, HU intensity:
```python
from python.analysis.converters.arterial_gnet_converter import add_arterial_gnet_features

# Add geometric and intensity features
dense_graph_with_features = add_arterial_gnet_features(
    dense_graph,
    hu_intensity_map=cta_volume,  # Optional: CTA volume for HU sampling
)

# Now each node has 'features' dict with:
# - position_x, position_y, position_z
# - diameter (2 * radius)
# - polar_angle, azimuthal_angle (tangent direction)
# - curvature, torsion
# - hu_intensity (if CTA provided)
# - accumulated_length_from_access
# - vessel_type
```

---

## Integration Examples

### Full Pipeline → EVC

```python
from python.analysis.centerline_antiga_2008 import CenterlineExtractionPipeline
from python.analysis.converters import centerline_to_evc_graph

# Extract centerlines
pipeline = CenterlineExtractionPipeline('vessel_mask.nii.gz', 'output/')
results = pipeline.run()

# Convert to EVC format
evc_graph = centerline_to_evc_graph(
    results['stage7']['graph'],
    output_pickle_path='output/evc_graph.pkl',
)

# Use with EVC dataset (PyTorch Geometric)
# Place in: your_evc_dataset/raw/case_001.pickle
```

### Full Pipeline → ArterialGNet

```python
from python.analysis.converters import centerline_to_arterial_gnet_graph
import nibabel as nib

# Load CTA volume for HU intensity
cta = nib.load('cta_scan.nii.gz')
hu_map = cta.get_fdata()

# Convert with advanced features
gnet_result = centerline_to_arterial_gnet_graph(
    results['stage7']['graph'],
    output_pickle_path='output/arterial_gnet_graph.pkl',
    include_segment_graph=True,
)

# Add features
from python.analysis.converters.arterial_gnet_converter import add_arterial_gnet_features
dense_with_features = add_arterial_gnet_features(
    gnet_result['dense_graph'],
    hu_intensity_map=hu_map,
)
```

### Batch Conversion

```python
import pickle
from pathlib import Path

centerline_dir = Path('outputs/centerlines/')
evc_output_dir = Path('evc_dataset/raw/')
evc_output_dir.mkdir(parents=True, exist_ok=True)

for pkl_file in centerline_dir.glob('**/centerline.pkl'):
    case_id = pkl_file.parent.parent.name
    
    with open(pkl_file, 'rb') as f:
        graph = pickle.load(f)
    
    evc_graph = centerline_to_evc_graph(
        graph,
        output_pickle_path=evc_output_dir / f'{case_id}.pickle',
    )
    print(f"✓ Converted {case_id}: {evc_graph.number_of_edges()} vessels")
```

---

## Testing

Run converter tests:
```bash
python test_converters.py
```

Output:
```
✓ Centerline graph: 872 nodes, 1066 edges
✓ EVC graph: 16 bifurcation nodes, 28 vessel edges
✓ Dense graph: 872 nodes, 1066 edges
✓ Segment graph: 55 segments, 249 connections
ALL CONVERTER TESTS PASSED ✓
```

---

## Differences: EVC vs ArterialGNet

| Feature | EVC | ArterialGNet |
|---------|-----|--------------|
| **Graph Type** | Undirected | Directed |
| **Vessel Representation** | Edges (then transformed to nodes) | Nodes (dense) |
| **Resolution** | Aggregated segments | All centerline points |
| **Node Count** | Bifurcations only (~16) | All points (~872) |
| **Task** | Node classification (vessel type) | GNN feature learning |
| **Features** | Basic (length, radius) | Advanced (curvature, HU, direction) |
| **Segment Graph** | Not used | Optional (for hierarchy) |

---

## Troubleshooting

### EVC: Empty graph or few edges
- Check that centerline graph has multiple segments (not just one vessel)
- Verify `segment_id` attributes are present on nodes

### ArterialGNet: Missing features
- Use `add_arterial_gnet_features()` for advanced attributes
- Provide CTA volume for HU intensity sampling

### File size concerns
- EVC graphs: ~4-10 KB (sparse, bifurcations only)
- ArterialGNet dense: ~150-500 KB (all points)
- Compress with `pickle.HIGHEST_PROTOCOL` for smaller files

---

## References

1. **EVC Dataset**
   - Pereañez et al., "Extracranial vessel classification..."
   - GitHub: https://github.com/perecanals/EVC

2. **ArterialGNet**
   - Pereañez et al., "Graph neural networks for arterial labeling..."
   - GitHub: https://github.com/perecanals/arterial_gnet

3. **Antiga 2008 Pipeline**
   - Antiga et al., "An image-based modeling framework..."
   - Medical & Biological Engineering & Computing (2008)
