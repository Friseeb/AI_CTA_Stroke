"""Stage 7: Centerline graph construction and export."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


def build_centerline_graph(
    centerlines: dict,
    bifurcations: dict,
) -> dict:
    """Build NetworkX graph from centerlines and bifurcations.

    Nodes represent centerline points with attributes (position, radius).
    Edges connect consecutive points along centerlines and bifurcation junctions.

    Parameters
    ----------
    centerlines : dict
        Centerlines with paths and radii (from Stage 5)
    bifurcations : dict
        Bifurcations detected (from Stage 6)

    Returns
    -------
    dict
        'graph': NetworkX DiGraph
        'node_data': list of node attributes
        'edge_data': list of edge attributes
    """
    graph = nx.DiGraph()

    # Map from (segment_id, point_index) -> global node_id
    node_mapping = {}
    node_counter = 0
    node_data = []

    # Add nodes from each centerline
    for seg_id, seg_data in centerlines.items():
        path = np.array(seg_data['path'])
        radii = np.array(seg_data['radii'])

        for i, (point, radius) in enumerate(zip(path, radii)):
            node_id = node_counter
            node_mapping[(seg_id, i)] = node_id
            node_counter += 1

            graph.add_node(
                node_id,
                position=point.tolist(),
                radius=float(radius),
                segment_id=seg_id,
                segment_index=int(i),
            )

            node_data.append({
                'node_id': node_id,
                'position': point.tolist(),
                'radius': float(radius),
                'segment_id': seg_id,
                'segment_index': int(i),
            })

    # Add edges: consecutive points along each centerline
    edge_data = []
    for seg_id, seg_data in centerlines.items():
        path = np.array(seg_data['path'])

        for i in range(len(path) - 1):
            u = node_mapping[(seg_id, i)]
            v = node_mapping[(seg_id, i + 1)]
            distance = float(np.linalg.norm(path[i + 1] - path[i]))

            graph.add_edge(u, v, distance=distance, type='centerline')
            edge_data.append({
                'source': u,
                'target': v,
                'distance': distance,
                'type': 'centerline',
            })

    # Add bifurcation edges
    for bifurc in bifurcations.get('bifurcations', []):
        seg_id_1 = bifurc['segment_1']
        seg_id_2 = bifurc['segment_2']
        i_1, i_2 = bifurc['point_indices']

        u = node_mapping.get((seg_id_1, i_1))
        v = node_mapping.get((seg_id_2, i_2))

        if u is not None and v is not None:
            contact_dist = bifurc['contact_distance']
            graph.add_edge(u, v, distance=contact_dist, type='bifurcation')
            edge_data.append({
                'source': u,
                'target': v,
                'distance': contact_dist,
                'type': 'bifurcation',
            })

    return {
        'graph': graph,
        'node_data': node_data,
        'edge_data': edge_data,
    }


def export_graph(
    graph_result: dict,
    output_dir: str | Path,
    basename: str = 'centerline',
) -> dict:
    """Export centerline graph to pickle and JSON formats.

    Parameters
    ----------
    graph_result : dict
        Result from build_centerline_graph()
    output_dir : str | Path
        Directory to save files
    basename : str
        Base filename (without extension)

    Returns
    -------
    dict
        'pickle_path': str, path to .pkl file
        'json_nodes_path': str, path to nodes.json
        'json_edges_path': str, path to edges.json
        'success': bool
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = graph_result['graph']
    node_data = graph_result['node_data']
    edge_data = graph_result['edge_data']

    paths = {}

    # Save graph as pickle
    pkl_path = output_dir / f'{basename}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(graph, f)
    paths['pickle_path'] = str(pkl_path)

    # Save nodes as JSON
    nodes_json_path = output_dir / f'{basename}_nodes.json'
    with open(nodes_json_path, 'w') as f:
        json.dump(node_data, f, indent=2)
    paths['json_nodes_path'] = str(nodes_json_path)

    # Save edges as JSON
    edges_json_path = output_dir / f'{basename}_edges.json'
    with open(edges_json_path, 'w') as f:
        json.dump(edge_data, f, indent=2)
    paths['json_edges_path'] = str(edges_json_path)

    paths['success'] = True
    paths['num_nodes'] = len(node_data)
    paths['num_edges'] = len(edge_data)

    return paths
