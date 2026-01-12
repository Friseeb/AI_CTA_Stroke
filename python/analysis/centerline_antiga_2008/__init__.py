"""Antiga et al. 2008 centerline extraction pipeline.

Modular implementation of Eikonal-based shortest path tracing via Voronoi diagrams.

Reference:
  Antiga, L., Piccinelli, M., Botti, L., Ene-Iordache, B., Remuzzi, A., & Steinman, D. A. (2008).
  An image-based modeling framework for patient-specific computational hemodynamics.
  Medical & Biological Engineering & Computing, 43(3), 252–261.
"""
from __future__ import annotations

from .stage1_surface_extraction import extract_surface
from .stage2_extremal_points import detect_extremal_points
from .stage4_eikonal import extract_centerlines_via_eikonal
from .stage5_radius import compute_radii
from .stage6_bifurcations import detect_bifurcations
from .stage7_graph import build_centerline_graph, export_graph
from .pipeline import CenterlineExtractionPipeline

__all__ = [
    'extract_surface',
    'detect_extremal_points',
    'extract_centerlines_via_eikonal',
    'compute_radii',
    'detect_bifurcations',
    'build_centerline_graph',
    'export_graph',
    'CenterlineExtractionPipeline',
]
