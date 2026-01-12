"""Converters for centerline graphs to downstream tool formats.

This module provides converters to transform Antiga 2008 centerline extraction
pipeline outputs into formats expected by:
- EVC (Extracranial Vessel Classification): PyTorch Geometric node classification
- ArterialGNet: Graph neural network vessel labeling
"""
from __future__ import annotations

from .evc_converter import centerline_to_evc_graph
from .arterial_gnet_converter import centerline_to_arterial_gnet_graph

__all__ = [
    'centerline_to_evc_graph',
    'centerline_to_arterial_gnet_graph',
]
