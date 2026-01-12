# Centerline Extraction: Antiga et al. 2008 Implementation

**Reference:** Antiga, L., Piccinelli, M., Botti, L., Ene-Iordache, B., Remuzzi, A., & Steinman, D. A. (2008). An image-based modeling framework for patient-specific computational hemodynamics. Medical & Biological Engineering & Computing, 43(3), 252–261.

## Overview

Implementation of the Vascular Modelling Toolkit (VMTK) centerline extraction pipeline using shortest path tracing via Eikonal equation optimization on Voronoi diagrams.

---

## Stage 1: Vessel Segmentation & Surface Model Extraction

**Input:** CTA volume (NIfTI from ResNet)  
**Output:** Closed triangulated surface (VTK)

```python
import vmtk
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening

# Load CTA segmentation (binary mask from ResNet)
cta_img = nib.load('segmentation.nii.gz')
mask = cta_img.get_fdata() > 0

# Clean mask: fill holes and remove small objects
mask = binary_fill_holes(mask)
mask = binary_opening(mask, iterations=2)

# Remove intracranial vessels (Z threshold if needed)
# mask[..., :intracranial_z] = 0

# Surface extraction: Marching Cubes → triangulated mesh
from skimage import measure
vertices, faces, _, _ = measure.marching_cubes(mask, level=0.5)

# Smooth surface (Laplacian, 10 iterations)
# VMTK: vmtkSurfaceSmoothing with NumberOfIterations=10
```

---

## Stage 2: Extremal Point Detection

**Goal:** Identify start/endpoints at vessel terminations  
**Method:** Automatic detection using distance map maxima

```python
from scipy.ndimage import distance_transform_edt, maximum_filter, label

# Compute distance map: distance of each voxel to nearest boundary
distance_map = distance_transform_edt(mask)

# Find local maxima (peaks = vessel centers far from walls)
local_max = (distance_map == maximum_filter(distance_map, size=5)) & mask

# Label connected components of maxima
labeled_array, num_features = label(local_max)

# Extract coordinates of peaks
extremal_points = []
for i in range(1, num_features + 1):
    coords = np.where(labeled_array == i)
    centroid = np.mean(coords, axis=1)
    max_dist = distance_map[tuple(coords)].max()
    extremal_points.append({
        'id': i,
        'position': centroid,
        'distance_value': max_dist
    })

print(f"Detected {len(extremal_points)} extremal points (endpoints)")
```

---

## Stage 3: Voronoi Diagram Computation

**Goal:** Build medial axis skeleton for shortest path routing  
**Method:** Voronoi diagram of binary segmentation

```python
# The Voronoi diagram skeleton = ridge of distance map
# All points satisfying: |∇distance| = 1 lie on the Voronoi surface

# For direct computation, use VMTK's vtkImageToVoronoiDiagram
# or scipy.spatial.Voronoi after discretization

# In practice, the distance map itself encodes the Voronoi skeleton:
# Points with high distance values and low gradient magnitude 
# are interior to the Voronoi skeleton
```

---

## Stage 4: Eikonal Equation & Shortest Path Tracing

**Goal:** Compute geodesic paths connecting extremal points along Voronoi skeleton  
**Method:** Fast Marching Method (FMM) solving Eikonal equation: $|\nabla \phi| = 1$

```python
import skfmm  # scikit-fmm: Fast Marching Method

# Speed function: 1 inside vessel, ∞ outside
speed = np.ones_like(mask, dtype=float)
speed[~mask] = np.inf

# Compute distance map from reference point using FMM
# (Eikonal solver with speed function)
phi = skfmm.distance(speed, self=True)

def extract_path_via_gradient_descent(phi, start_point, end_point):
    """
    Trace shortest path from end to start by following -∇φ.
    phi is the potential function; minimizing φ gives geodesic path.
    """
    path = [np.array(end_point, dtype=float)]
    current = np.array(end_point, dtype=float)
    step_size = 0.5
    
    while np.linalg.norm(current - start_point) > 1.0:
        # Compute gradient at current position (central differences)
        grad = np.array([
            (phi[min(int(current[0])+1, phi.shape[0]-1), int(current[1]), int(current[2])] -
             phi[max(int(current[0])-1, 0), int(current[1]), int(current[2])]) / 2.0,
            (phi[int(current[0]), min(int(current[1])+1, phi.shape[1]-1), int(current[2])] -
             phi[int(current[0]), max(int(current[1])-1, 0), int(current[2])]) / 2.0,
            (phi[int(current[0]), int(current[1]), min(int(current[2])+1, phi.shape[2]-1)] -
             phi[int(current[0]), int(current[1]), max(int(current[2])-1, 0)]) / 2.0
        ])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            break
        grad_unit = grad / grad_norm
        
        # Step in direction of steepest descent
        current = current - step_size * grad_unit
        path.append(current.copy())
    
    return np.array(path)

# Extract centerlines for each pair of extremal points
# In practice, connect endpoints hierarchically (avoid loops)
centerlines = {}
for i in range(len(extremal_points)):
    for j in range(i+1, len(extremal_points)):
        start = extremal_points[i]['position']
        end = extremal_points[j]['position']
        
        path = extract_path_via_gradient_descent(phi, start, end)
        centerlines[f'segment_{i}_{j}'] = path
        
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        print(f"Segment {i}-{j}: {len(path)} points, {length:.1f} mm")
```

---

## Stage 5: Maximal Inscribed Sphere Radius Computation

**Goal:** Calculate vessel radius at each centerline point  
**Method:** Distance from centerline to vessel wall (from distance_map)

```python
def compute_radii_along_centerline(centerline_path, distance_map):
    """
    For each point on centerline, radius = distance to nearest wall.
    The distance_map from Stage 2 already encodes this.
    """
    radii = []
    for point in centerline_path:
        # Trilinear interpolation of distance map
        x, y, z = point
        x_floor, y_floor, z_floor = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        
        # Clamp to bounds
        x_floor = np.clip(x_floor, 0, distance_map.shape[0]-2)
        y_floor = np.clip(y_floor, 0, distance_map.shape[1]-2)
        z_floor = np.clip(z_floor, 0, distance_map.shape[2]-2)
        
        # Trilinear interpolation
        dx = x - x_floor
        dy = y - y_floor
        dz = z - z_floor
        
        v000 = distance_map[x_floor, y_floor, z_floor]
        v001 = distance_map[x_floor, y_floor, z_floor+1]
        v010 = distance_map[x_floor, y_floor+1, z_floor]
        v011 = distance_map[x_floor, y_floor+1, z_floor+1]
        v100 = distance_map[x_floor+1, y_floor, z_floor]
        v101 = distance_map[x_floor+1, y_floor, z_floor+1]
        v110 = distance_map[x_floor+1, y_floor+1, z_floor]
        v111 = distance_map[x_floor+1, y_floor+1, z_floor+1]
        
        v00 = v000*(1-dx) + v100*dx
        v01 = v001*(1-dx) + v101*dx
        v10 = v010*(1-dx) + v110*dx
        v11 = v011*(1-dx) + v111*dx
        
        v0 = v00*(1-dy) + v10*dy
        v1 = v01*(1-dy) + v11*dy
        
        radius = v0*(1-dz) + v1*dz
        radii.append(radius)
    
    return np.array(radii)

# Apply to all centerlines
for seg_id, centerline_path in centerlines.items():
    radii = compute_radii_along_centerline(centerline_path, distance_map)
    centerlines[seg_id] = {
        'path': centerline_path,
        'radii': radii,
        'mean_radius': np.mean(radii),
        'length': np.sum(np.linalg.norm(np.diff(centerline_path, axis=0), axis=1))
    }
```

---

## Stage 6: Branch Splitting & Bifurcation Detection

**Goal:** Identify bifurcation points and segment vessel network into branches  
**Method:** Tube containment relationships via maximal inscribed spheres

```python
def detect_bifurcations(centerlines, distance_map):
    """
    Bifurcations occur where centerline segments spatially intersect.
    Detected via tube overlap (distance of centerline < sum of radii).
    """
    bifurcations = []
    centerline_list = list(centerlines.items())
    
    for i, (seg_id_1, seg_1) in enumerate(centerline_list):
        for j, (seg_id_2, seg_2) in enumerate(centerline_list):
            if i >= j:
                continue
            
            path_1 = seg_1['path']
            path_2 = seg_2['path']
            radii_1 = seg_1['radii']
            radii_2 = seg_2['radii']
            
            # Find closest point pair between segments
            min_dist = np.inf
            best_pair = None
            
            for k, p1 in enumerate(path_1):
                for l, p2 in enumerate(path_2):
                    dist = np.linalg.norm(p1 - p2)
                    
                    # Check tube containment condition
                    if dist < (radii_1[k] + radii_2[l]):
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (k, l, p1, p2)
            
            if best_pair is not None:
                k, l, p1, p2 = best_pair
                bifurcations.append({
                    'segments': [seg_id_1, seg_id_2],
                    'location': (p1 + p2) / 2.0,
                    'point_indices': (k, l),
                    'distance': min_dist
                })
    
    return bifurcations

bifurcations = detect_bifurcations(centerlines, distance_map)
print(f"Detected {len(bifurcations)} bifurcations")
```

---

## Stage 7: Graph Generation

**Goal:** Convert centerlines + bifurcations → NetworkX graph  
**Output:** Nodes = segments, Edges = bifurcation connections

```python
import networkx as nx
import json
import pickle

def build_centerline_graph(centerlines, bifurcations):
    """
    Create graph where:
    - Nodes = vessel segments (with centerline, radius, length, etc.)
    - Edges = bifurcation connections
    """
    G = nx.Graph()
    
    # Add nodes (segments)
    for seg_id, seg_data in centerlines.items():
        G.add_node(seg_id,
                   centerline_path=seg_data['path'].tolist(),
                   radii=seg_data['radii'].tolist(),
                   length=seg_data['length'],
                   mean_radius=seg_data['mean_radius'],
                   start_pos=seg_data['path'][0].tolist(),
                   end_pos=seg_data['path'][-1].tolist(),
                   num_points=len(seg_data['path']))
    
    # Add edges (bifurcations)
    for bifurc in bifurcations:
        seg_1, seg_2 = bifurc['segments']
        if seg_1 in G and seg_2 in G:
            G.add_edge(seg_1, seg_2,
                      bifurcation_location=bifurc['location'].tolist(),
                      contact_distance=bifurc['distance'],
                      point_indices=bifurc['point_indices'])
    
    return G

# Build graph
centerline_graph = build_centerline_graph(centerlines, bifurcations)

print(f"Graph: {centerline_graph.number_of_nodes()} nodes, "
      f"{centerline_graph.number_of_edges()} edges")

# Export to pickle (for EVC/ArterialGNet downstream labeling)
with open('centerline_graph.pickle', 'wb') as f:
    pickle.dump(centerline_graph, f)

# Export to JSON (BIDS-compatible sidecar)
graph_json = {
    'description': 'Centerline graph extracted via Antiga et al. 2008 VMTK pipeline',
    'method': 'Eikonal-based shortest path tracing on Voronoi diagram',
    'nodes': {},
    'edges': []
}

for node in centerline_graph.nodes():
    graph_json['nodes'][node] = dict(centerline_graph.nodes[node])

for u, v, data in centerline_graph.edges(data=True):
    graph_json['edges'].append({
        'source': u,
        'target': v,
        'data': dict(data)
    })

with open('centerline_graph.json', 'w') as f:
    json.dump(graph_json, f, indent=2)

print("Exported: centerline_graph.pickle, centerline_graph.json")
```

---

## QC & Validation

- **Visual inspection:** Overlay centerlines on CTA in ITK-SNAP
- **Topological checks:** No orphaned segments, connected graph
- **Radius plausibility:** 0.5–3 mm typical for carotid/vertebral (4–6 mm for cavernous ICA)
- **Centerline smoothness:** Curvature κ < 0.5 mm⁻¹ (no sharp kinks)
- **Bifurcation angles:** 20–150° (physiologically realistic)
- **Path continuity:** No gaps in centerline, resampling to 0.5 mm uniform spacing

---

## References

1. Antiga, L., & Steinman, D. A. (2004). Robust and objective decomposition of the arterial and venous trees from 3D imaging data. IEEE Transactions on Medical Imaging, 23(11), 1427–1441.

2. Antiga, L., Piccinelli, M., Botti, L., Ene-Iordache, B., Remuzzi, A., & Steinman, D. A. (2008). An image-based modeling framework for patient-specific computational hemodynamics. Medical & Biological Engineering & Computing, 43(3), 252–261.

3. VMTK Documentation: http://www.vmtk.org/
