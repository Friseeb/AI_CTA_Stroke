"""Converter: Antiga 2008 centerline graph → EVC format.

EVC expects:
- NetworkX Graph with vessel segments as EDGES (not nodes)
- Edge attributes: 'pos', 'features', 'vessel_type', 'vessel_type_name'
- Node attributes: positions (bifurcation points)
- Then applies node_transform() to convert edges→nodes for classification
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


def centerline_to_evc_graph(
    centerline_graph: nx.DiGraph,
    output_pickle_path: str | Path | None = None,
    vessel_type_name: str = 'other',
    endpoint_merge_tol: float | None = None,
    min_segment_length: float | None = None,
    min_num_points: int | None = None,
    min_mean_radius: float | None = None,
    max_mean_radius: float | None = None,
    max_tortuosity: float | None = None,
    keep_largest_component: bool = False,
) -> nx.Graph:
    """Convert Antiga centerline graph to EVC format.

    EVC expects vessel segments as EDGES with attributes:
    - pos: np.array([x, y, z]) - mean position of segment
    - features: np.array with segment features (length, radius, etc.)
    - vessel_type_name: str - vessel label (default 'other')
    - vessel_type: int - vessel type ID (0-13)

    Parameters
    ----------
    centerline_graph : nx.DiGraph
        Centerline graph from Stage 7 (build_centerline_graph)
        Node attributes: position, radius, segment_id, segment_index
        Edge attributes: distance, type
    output_pickle_path : str | Path, optional
        If provided, saves EVC-formatted graph as pickle
    vessel_type_name : str
        Vessel type label (default 'other' = unknown)
    endpoint_merge_tol : float | None
        If set, merges segment endpoints within this distance (voxel/mm) into
        shared bifurcation nodes.
    min_segment_length : float | None
        Minimum path length for a segment to be kept.
    min_num_points : int | None
        Minimum number of centerline points for a segment to be kept.
    min_mean_radius : float | None
        Minimum mean radius for a segment to be kept.
    max_mean_radius : float | None
        Maximum mean radius for a segment to be kept.
    max_tortuosity : float | None
        Maximum tortuosity (path_length / straight_length) for a segment to be kept.
    keep_largest_component : bool
        If True, keep only the largest connected component after conversion.

    Returns
    -------
    nx.Graph
        EVC-formatted graph with vessel segments as edges
    """
    # Vessel type mapping from EVC dataset
    vessel_type_dict = {
        'other': 0, 'AA': 1, 'BT': 2, 'RCCA': 3, 'LCCA': 4,
        'RSA': 5, 'LSA': 6, 'RVA': 7, 'LVA': 8, 'RICA': 9,
        'LICA': 10, 'RECA': 11, 'LECA': 12, 'BA': 13
    }

    # Create undirected graph (EVC uses nx.Graph)
    evc_graph = nx.Graph()

    def endpoint_key(pos: np.ndarray) -> tuple:
        if endpoint_merge_tol is None or endpoint_merge_tol <= 0:
            return tuple(pos)
        return tuple(np.round(np.asarray(pos) / endpoint_merge_tol).astype(int))

    # Group nodes by segment_id to reconstruct vessel segments
    segments = {}
    for node_id, node_data in centerline_graph.nodes(data=True):
        seg_id = node_data['segment_id']
        if seg_id not in segments:
            segments[seg_id] = []
        segments[seg_id].append({
            'node_id': node_id,
            'position': np.array(node_data['position']),
            'radius': node_data['radius'],
            'segment_index': node_data['segment_index'],
        })

    # Sort each segment by segment_index
    for seg_id in segments:
        segments[seg_id].sort(key=lambda x: x['segment_index'])

    # Collect bifurcation endpoints (segment endpoints)
    endpoint_positions = {}
    for seg_id, nodes in segments.items():
        start_pos = np.asarray(nodes[0]['position'], dtype=float)
        end_pos = np.asarray(nodes[-1]['position'], dtype=float)
        for pos in (start_pos, end_pos):
            key = endpoint_key(pos)
            endpoint_positions.setdefault(key, []).append(pos)

    # Add bifurcation nodes to graph
    bifurcation_node_map = {}
    for i, (key, positions) in enumerate(endpoint_positions.items()):
        mean_pos = np.mean(np.array(positions), axis=0)
        evc_graph.add_node(i, pos=mean_pos)
        bifurcation_node_map[key] = i

    # Add vessel segments as edges
    for seg_id, nodes in segments.items():
        start_pos = np.asarray(nodes[0]['position'], dtype=float)
        end_pos = np.asarray(nodes[-1]['position'], dtype=float)
        start_key = endpoint_key(start_pos)
        end_key = endpoint_key(end_pos)

        start_node = bifurcation_node_map[start_key]
        end_node = bifurcation_node_map[end_key]

        # Compute segment features
        positions = np.array([n['position'] for n in nodes])
        radii = np.array([n['radius'] for n in nodes])

        mean_pos = positions.mean(axis=0)
        segment_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        straight_length = float(np.linalg.norm(end_pos - start_pos))
        mean_radius = float(radii.mean())
        min_radius = float(radii.min())
        max_radius = float(radii.max())
        num_points = len(nodes)

        if min_segment_length is not None and segment_length < min_segment_length:
            continue
        if min_num_points is not None and num_points < min_num_points:
            continue
        if min_mean_radius is not None and mean_radius < min_mean_radius:
            continue
        if max_mean_radius is not None and mean_radius > max_mean_radius:
            continue
        if max_tortuosity is not None and straight_length > 0:
            if segment_length / straight_length > max_tortuosity:
                continue

        # EVC edge attributes
        edge_features = np.array([
            segment_length,
            mean_radius,
            min_radius,
            max_radius,
            num_points,  # number of points
        ])

        evc_graph.add_edge(
            start_node,
            end_node,
            pos=mean_pos,
            features=edge_features,
            vessel_type_name=vessel_type_name,
            vessel_type=vessel_type_dict.get(vessel_type_name, 0),
        )

    if keep_largest_component and evc_graph.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(evc_graph), key=len)
        evc_graph = evc_graph.subgraph(largest_cc).copy()

    # Save if output path provided
    if output_pickle_path is not None:
        output_path = Path(output_pickle_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(evc_graph, f)

    return evc_graph


def apply_evc_node_transform(evc_graph: nx.Graph) -> nx.Graph:
    """Apply EVC's node_transform: edges→nodes for classification.

    This transforms the graph so vessel segments (edges) become nodes,
    enabling node classification instead of edge classification.

    Parameters
    ----------
    evc_graph : nx.Graph
        EVC-formatted graph with vessels as edges

    Returns
    -------
    nx.Graph
        Transformed graph with vessels as nodes
    """
    # Import EVC's node_transform if available, else implement inline
    try:
        import sys
        from pathlib import Path
        evc_path = Path(__file__).parents[2] / 'external' / 'EVC'
        if str(evc_path) not in sys.path:
            sys.path.insert(0, str(evc_path))
        from extracranial_vessel_labelling.data.utils import node_transform
        return node_transform(evc_graph)
    except ImportError:
        # Inline implementation
        new_nodes = []
        edges_to_nodes = {}
        new_nodes_to_old_edges = {}
        new_node = 0

        for edge in evc_graph.edges:
            new_nodes.append(new_node)
            edges_to_nodes[edge] = new_node
            edges_to_nodes[(edge[1], edge[0])] = new_node
            new_nodes_to_old_edges[new_node] = edge
            new_node += 1

        new_edges = []
        for node in evc_graph.nodes:
            if len(list(evc_graph.edges(node))) > 1:
                edge_list_aux = [edges_to_nodes[edge] for edge in evc_graph.edges(node)]
                for idx, src in enumerate(edge_list_aux):
                    for dst in edge_list_aux[idx + 1:]:
                        new_edges.append([src, dst])

        transformed_graph = nx.Graph()
        for node in new_nodes:
            transformed_graph.add_node(node)
            old_edge = new_nodes_to_old_edges[node]
            for attr_key, attr_val in evc_graph[old_edge[0]][old_edge[1]].items():
                transformed_graph.nodes[node][attr_key] = attr_val

        for edge in new_edges:
            transformed_graph.add_edge(edge[0], edge[1])

        return transformed_graph
