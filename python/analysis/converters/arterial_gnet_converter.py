"""Converter: Antiga 2008 centerline graph → ArterialGNet format.

ArterialGNet expects:
- Dense graph: all centerline points with node attributes
  - pos: [x, y, z] position
  - radius: vessel radius
  - vessel_type_name: str label
  - (optional) vessel_type, features, hierarchy
- Segment graph: aggregated vessel segments (for higher-level features)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np


def centerline_to_arterial_gnet_graph(
    centerline_graph: nx.DiGraph,
    output_pickle_path: str | Path | None = None,
    vessel_type_name: str = 'other',
    include_segment_graph: bool = False,
) -> dict:
    """Convert Antiga centerline graph to ArterialGNet format.

    ArterialGNet uses dense graphs with all centerline points as nodes.
    Node attributes expected:
    - pos: np.array([x, y, z])
    - radius: float
    - vessel_type_name: str

    Parameters
    ----------
    centerline_graph : nx.DiGraph
        Centerline graph from Stage 7
    output_pickle_path : str | Path, optional
        If provided, saves graph(s) as pickle
    vessel_type_name : str
        Vessel label (default 'other')
    include_segment_graph : bool
        If True, also generate aggregated segment graph

    Returns
    -------
    dict
        'dense_graph': nx.DiGraph with all centerline points
        'segment_graph': nx.DiGraph with segments (if include_segment_graph=True)
    """
    # Vessel type mapping
    vessel_type_dict = {
        'other': 0, 'AA': 1, 'BT': 2, 'RCCA': 3, 'LCCA': 4,
        'RSA': 5, 'LSA': 6, 'RVA': 7, 'LVA': 8, 'RICA': 9,
        'LICA': 10, 'RECA': 11, 'LECA': 12, 'BA': 13
    }

    # Create dense graph (copy centerline graph structure)
    dense_graph = nx.DiGraph()

    for node_id, node_data in centerline_graph.nodes(data=True):
        dense_graph.add_node(
            node_id,
            pos=np.array(node_data['position']),
            radius=float(node_data['radius']),
            vessel_type_name=vessel_type_name,
            vessel_type=vessel_type_dict.get(vessel_type_name, 0),
            segment_id=node_data['segment_id'],
            segment_index=node_data['segment_index'],
        )

    for u, v, edge_data in centerline_graph.edges(data=True):
        dense_graph.add_edge(
            u, v,
            distance=edge_data['distance'],
            type=edge_data['type'],
        )

    result = {'dense_graph': dense_graph}

    # Optionally create segment graph (aggregated by segment_id)
    if include_segment_graph:
        segment_graph = nx.DiGraph()

        # Group nodes by segment_id
        segments = {}
        for node_id, node_data in dense_graph.nodes(data=True):
            seg_id = node_data['segment_id']
            if seg_id not in segments:
                segments[seg_id] = []
            segments[seg_id].append(node_data)

        # Create segment nodes
        for seg_idx, (seg_id, nodes) in enumerate(segments.items()):
            positions = np.array([n['pos'] for n in nodes])
            radii = np.array([n['radius'] for n in nodes])

            segment_graph.add_node(
                seg_idx,
                pos=positions.mean(axis=0),
                radius=float(radii.mean()),
                vessel_type_name=vessel_type_name,
                vessel_type=vessel_type_dict.get(vessel_type_name, 0),
                segment_id=seg_id,
                num_points=len(nodes),
                segment_length=float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
            )

        # Connect segments at bifurcations (find segments with shared endpoints)
        segment_endpoints = {}
        for seg_idx, (seg_id, nodes) in enumerate(segments.items()):
            sorted_nodes = sorted(nodes, key=lambda x: x['segment_index'])
            start_pos = tuple(sorted_nodes[0]['pos'])
            end_pos = tuple(sorted_nodes[-1]['pos'])

            for pos in [start_pos, end_pos]:
                if pos not in segment_endpoints:
                    segment_endpoints[pos] = []
                segment_endpoints[pos].append(seg_idx)

        # Add edges between segments meeting at bifurcations
        for pos, connected_segs in segment_endpoints.items():
            if len(connected_segs) > 1:
                for i, seg_i in enumerate(connected_segs):
                    for seg_j in connected_segs[i + 1:]:
                        segment_graph.add_edge(seg_i, seg_j, bifurcation_pos=np.array(pos))

        result['segment_graph'] = segment_graph

    # Save if output path provided
    if output_pickle_path is not None:
        output_path = Path(output_pickle_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

    return result


def add_arterial_gnet_features(
    dense_graph: nx.DiGraph,
    hu_intensity_map: np.ndarray | None = None,
) -> nx.DiGraph:
    """Add advanced features expected by ArterialGNet.

    Computes:
    - Curvature and torsion along centerlines
    - Direction vectors (polar/azimuthal angles)
    - HU intensity from CTA volume (if provided)
    - Accumulated length from access point

    Parameters
    ----------
    dense_graph : nx.DiGraph
        Dense centerline graph
    hu_intensity_map : np.ndarray, optional
        CTA volume for HU intensity sampling

    Returns
    -------
    nx.DiGraph
        Graph with added 'features' dict per node
    """
    # Group by segment for sequential feature computation
    segments = {}
    for node_id, node_data in dense_graph.nodes(data=True):
        seg_id = node_data['segment_id']
        if seg_id not in segments:
            segments[seg_id] = []
        segments[seg_id].append((node_id, node_data))

    for seg_id, nodes in segments.items():
        # Sort by segment_index
        nodes = sorted(nodes, key=lambda x: x[1]['segment_index'])
        positions = np.array([n[1]['pos'] for _, n in nodes])

        for i, (node_id, node_data) in enumerate(nodes):
            features = {}

            # Position features
            features['position_x'] = float(node_data['pos'][0])
            features['position_y'] = float(node_data['pos'][1])
            features['position_z'] = float(node_data['pos'][2])
            features['diameter'] = float(node_data['radius'] * 2)

            # Direction vectors (tangent at point)
            if i > 0 and i < len(nodes) - 1:
                tangent = positions[i + 1] - positions[i - 1]
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 0:
                    tangent /= tangent_norm
                    polar_angle = float(np.arccos(np.clip(tangent[2], -1, 1)))
                    azimuthal_angle = float(np.arctan2(tangent[1], tangent[0]))
                else:
                    polar_angle = azimuthal_angle = 0.0
            else:
                polar_angle = azimuthal_angle = 0.0

            features['polar_angle'] = polar_angle
            features['azimuthal_angle'] = azimuthal_angle

            # Curvature (simple approximation)
            if 0 < i < len(nodes) - 1:
                v1 = positions[i] - positions[i - 1]
                v2 = positions[i + 1] - positions[i]
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    curvature = float(np.arccos(cos_angle))
                else:
                    curvature = 0.0
            else:
                curvature = 0.0

            features['curvature'] = curvature
            features['torsion'] = 0.0  # Placeholder (requires 4 points)
            features['blanking'] = 0.0  # Placeholder

            # HU intensity sampling
            if hu_intensity_map is not None:
                pos_int = np.clip(node_data['pos'].astype(int), 0, np.array(hu_intensity_map.shape) - 1)
                features['hu_intensity'] = float(hu_intensity_map[tuple(pos_int)])
            else:
                features['hu_intensity'] = 0.0

            # Accumulated length
            if i == 0:
                accum_length = 0.0
            else:
                accum_length = float(np.sum(np.linalg.norm(np.diff(positions[:i + 1], axis=0), axis=1)))
            features['accumulated_length_from_access'] = accum_length

            # Vessel type
            features['vessel_type'] = node_data['vessel_type']

            # Store features dict
            dense_graph.nodes[node_id]['features'] = features

    return dense_graph
