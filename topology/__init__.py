"""Topology module - graph structures for the autodidactic experiment spine.

This module provides the canonical Graph object and matrix-to-graph lift
used throughout the pipeline. Unlike analysis/entanglement_graph.py (which
computes metrics), this module defines the structure objects.

Key exports:
- Graph: Frozen dataclass representing a weighted graph
- matrix_to_graph: Canonical M -> Graph lift with configurable modes
- GraphFamily: Samplers for discrete topology hypotheses
"""

from .entanglement_graph import Graph, matrix_to_graph, GraphParams
from .graph_families import GraphFamily, sample_graph_family

__all__ = [
    "Graph",
    "matrix_to_graph",
    "GraphParams",
    "GraphFamily",
    "sample_graph_family",
]
