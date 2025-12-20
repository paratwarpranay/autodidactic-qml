"""Experiments package for autodidactic-qml.

This package contains experiment scripts for:
- Topology-aware quantum kernel benchmarks
- Graph family sweeps
- Phase diagram generation
- SQNT/Loop/UCIP bridge experiments

Key experiments:
- topology_kernel_benchmark: Main experiment spine with graph-conditioned kernels
- graph_family_sweep: Sweep across discrete topology hypotheses
- phase_diagram_visualization: Generate phase diagrams over graph parameters
- sqnt_loop_ucip_bridge: Full bridge from SQNT to Loop to UCIP metrics
"""

__all__ = [
    "topology_kernel_benchmark",
    "graph_family_sweep",
    "phase_diagram_visualization",
    "sqnt_loop_ucip_bridge",
]
