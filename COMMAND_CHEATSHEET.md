# Command Cheat Sheet

Quick reference for common pipeline operations.

---

## Pipeline Execution

### Single Case (Runner script)
```bash
cd /Users/sebastianfridman/Documents/pwd/AI_CTA_Stroke
/opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output python -u scripts/run_sub547_cta.py > outputs/sub-547_cta_run.log 2>&1 &
tail -f outputs/sub-547_cta_run.log
```

### Batch Processing (CLI)
```bash
python python/cli/run_centerline_batch.py \
    --manifest data/manifest.csv \
    --method vmtk \
    --output-root outputs/batch/
```

### Quick Test (module-only)
```bash
python test_quick_integration.py
```

---

## Format Conversion

### EVC Converter
```python
from python.analysis.converters import centerline_to_evc_graph
import pickle

with open('centerline.pkl', 'rb') as f:
    graph = pickle.load(f)

evc_graph = centerline_to_evc_graph(graph, 'evc_graph.pkl')
```

### ArterialGNet Converter
```python
from python.analysis.converters import centerline_to_arterial_gnet_graph

gnet = centerline_to_arterial_gnet_graph(
    graph, 
    'arterial_gnet_graph.pkl',
    include_segment_graph=True,
)
```

### Test Converters
```bash
python test_converters.py
# Output: ALL CONVERTER TESTS PASSED ✓
```

---

## Parameter Tuning

### Fine Vessels (Small Diameter)
```python
results = pipeline.run(
    min_component_size=50,      # Lower threshold
    min_distance_value=1.0,     # Smaller vessels
    step_size=0.2,              # Finer steps
)
```

### Large Vessels (Aorta, Carotids)
```python
results = pipeline.run(
    min_component_size=500,     # Filter noise
    min_distance_value=3.0,     # Larger vessels
    step_size=0.5,              # Coarser steps
)
```

### More Bifurcations
```python
results = pipeline.run(
    contact_distance_threshold=2.0,  # Larger search radius
)
```

---

## File I/O

### Load Centerline Graph
```python
import pickle
with open('centerline.pkl', 'rb') as f:
    graph = pickle.load(f)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

### Read JSON Output
```python
import json
with open('centerline_nodes.json', 'r') as f:
    nodes = json.load(f)

for node in nodes[:5]:  # First 5 nodes
    print(f"Node {node['id']}: pos={node['position']}, r={node['radius']:.2f}")
```

### Access Results Directly
```python
results = pipeline.run()

# Stage outputs
surface = results['stage1']['cleaned_mask']
extremal_points = results['stage2']['extremal_points']
centerlines = results['stage5']['centerlines_with_radii']
bifurcations = results['stage6']['bifurcations']
graph = results['stage7']['graph']
```

---

## Visualization

### Plot Centerlines (Matplotlib)
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
positions = [data['position'] for _, data in graph.nodes(data=True)]
positions = np.array(positions)
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
           c='red', s=1, alpha=0.5)

# Plot edges
for u, v in graph.edges():
    pos_u = graph.nodes[u]['position']
    pos_v = graph.nodes[v]['position']
    ax.plot([pos_u[0], pos_v[0]], 
            [pos_u[1], pos_v[1]], 
            [pos_u[2], pos_v[2]], 
            'b-', linewidth=0.5, alpha=0.3)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title(f'Centerline Graph ({graph.number_of_nodes()} nodes)')
plt.show()
```

### Export for FSLeyes
```python
import nibabel as nib
import numpy as np

# Create centerline mask
mask = nib.load('vessel_mask.nii.gz')
centerline_volume = np.zeros(mask.shape)

for _, data in graph.nodes(data=True):
    pos = np.round(data['position']).astype(int)
    if all(0 <= pos[i] < mask.shape[i] for i in range(3)):
        centerline_volume[tuple(pos)] = 1

# Save
centerline_img = nib.Nifti1Image(centerline_volume, mask.affine, mask.header)
nib.save(centerline_img, 'centerline_overlay.nii.gz')

# View: fsleyes vessel_mask.nii.gz centerline_overlay.nii.gz
```

---

## Analysis Queries

### Count Segments
```python
segment_ids = set(data['segment_id'] for _, data in graph.nodes(data=True))
print(f"Total segments: {len(segment_ids)}")
```

### Find Longest Path
```python
segments = {}
for node_id, data in graph.nodes(data=True):
    seg_id = data['segment_id']
    if seg_id not in segments:
        segments[seg_id] = []
    segments[seg_id].append(data['position'])

lengths = {seg_id: len(positions) for seg_id, positions in segments.items()}
longest_seg = max(lengths, key=lengths.get)
print(f"Longest segment: {longest_seg} ({lengths[longest_seg]} nodes)")
```

### Compute Total Length
```python
total_length = 0
for u, v, data in graph.edges(data=True):
    total_length += data['distance']
print(f"Total centerline length: {total_length:.1f} mm")
```

### Bifurcation Analysis
```python
bifurcations = results['stage6']['bifurcations']
print(f"Total bifurcations: {len(bifurcations)}")

for bif in bifurcations[:5]:  # First 5
    print(f"  Segments {bif['segment_1']} ↔ {bif['segment_2']} "
          f"at {bif['position']} (dist={bif['contact_distance']:.2f}mm)")
```

---

## Common Workflows

### Extract + Convert + Export
```python
# 1. Extract centerlines
pipeline = CenterlineExtractionPipeline('vessel_mask.nii.gz', 'outputs/')
results = pipeline.run()

# 2. Convert to EVC
evc_graph = centerline_to_evc_graph(
    results['stage7']['graph'], 
    'outputs/evc_graph.pkl',
)

# 3. Save summary
import json
with open('outputs/summary.json', 'w') as f:
    json.dump(pipeline.summary(), f, indent=2)

print("✓ Complete: centerlines extracted, converted, summarized")
```

### Batch Process + Aggregate
```python
from pathlib import Path
import pandas as pd

summaries = []
for case_dir in Path('outputs/batch/').glob('case_*'):
    pkl_path = case_dir / 'centerline.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            graph = pickle.load(f)
        
        summaries.append({
            'case_id': case_dir.name,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_segments': len(set(d['segment_id'] for _, d in graph.nodes(data=True))),
        })

df = pd.DataFrame(summaries)
df.to_csv('outputs/batch_summary.csv', index=False)
print(f"✓ Processed {len(df)} cases")
```

---

## Debugging

### Enable Verbose Logging
```python
pipeline = CenterlineExtractionPipeline(
    'vessel_mask.nii.gz', 
    'outputs/',
    log_level='DEBUG',  # INFO, DEBUG, WARNING, ERROR
)
```

### Check Intermediate Outputs
```python
results = pipeline.run()

# Stage 1: Surface extraction
print(f"Components: {results['stage1']['num_components']}")
print(f"Distance map range: {results['stage1']['distance_map'].min():.2f} - "
      f"{results['stage1']['distance_map'].max():.2f}")

# Stage 2: Extremal points
print(f"Extremal points found: {len(results['stage2']['extremal_points'])}")
for i, pt in enumerate(results['stage2']['extremal_points'][:3]):
    print(f"  Point {i}: {pt}")

# Stage 4: Centerlines
print(f"Paths extracted: {len(results['stage4']['centerlines'])}")
for i, path in enumerate(results['stage4']['centerlines'][:3]):
    print(f"  Path {i}: {len(path)} points")
```

### Validate Graph Structure
```python
import networkx as nx

# Check connectivity
if not nx.is_weakly_connected(graph):
    print("⚠ Graph has disconnected components")
    components = list(nx.weakly_connected_components(graph))
    print(f"  {len(components)} components")

# Check for self-loops
self_loops = list(nx.selfloop_edges(graph))
if self_loops:
    print(f"⚠ Found {len(self_loops)} self-loops")

# Check node attributes
for node_id, data in list(graph.nodes(data=True))[:5]:
    required = ['position', 'radius', 'segment_id', 'segment_index']
    missing = [attr for attr in required if attr not in data]
    if missing:
        print(f"⚠ Node {node_id} missing: {missing}")
```

---

## Quick Fixes

### Mask is Binary but Pipeline Expects Intensity
```python
# Convert binary mask to distance map
from scipy.ndimage import distance_transform_edt
import nibabel as nib

mask = nib.load('binary_mask.nii.gz').get_fdata()
distance_map = distance_transform_edt(mask)

# Save as intensity image
dist_img = nib.Nifti1Image(distance_map, mask.affine)
nib.save(dist_img, 'distance_map.nii.gz')

# Use distance map as input
pipeline = CenterlineExtractionPipeline('distance_map.nii.gz', 'outputs/')
```

### Too Many Small Components
```python
# Increase minimum component size
results = pipeline.run(min_component_size=500)  # Default: 50
```

### Centerlines Stop Prematurely
```python
# Increase max iterations and reduce step size
results = pipeline.run(
    step_size=0.2,        # Default: 0.5
    max_iterations=10000, # Default: 5000
)
```

---

## Performance Tips

- Use binary masks directly (avoid intensity thresholding in pipeline)
- Set appropriate `min_component_size` to filter noise early
- For large volumes (512³), consider downsampling to 256³ first
- Use `log_level='WARNING'` to reduce console output in batch jobs
- Save intermediate results (`results` dict) if re-running stages

---

## Documentation

- Full pipeline guide: `docs/IMPLEMENTATION_SUMMARY_Antiga2008_Pipeline.md`
- Quick start: `QUICKSTART_Antiga2008_Pipeline.md`
- Converter reference: `python/analysis/converters/README.md`
- End-to-end workflow: `docs/END_TO_END_WORKFLOW.md`
- Completion summary: `PIPELINE_COMPLETION_SUMMARY.md`
