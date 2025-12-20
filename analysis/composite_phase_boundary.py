from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .phase_diagram import compute_transition_line

def score_grid_to_transition_line(
    g_values: np.ndarray,
    axis2_values: np.ndarray,
    score_grid: np.ndarray,
    model_type: str = "continuous",
    max_breaks: int = 2,
    min_seg: int = 3,
    strategy: str = "largest_jump",
):
    """Treat composite score (e.g., p(phase=1)) as a scalar order parameter and map g*(axis2)."""
    return compute_transition_line(
        g_values=np.asarray(g_values, dtype=float),
        axis2_values=np.asarray(axis2_values, dtype=float),
        order_grid=np.asarray(score_grid, dtype=float),
        model_type=model_type,
        max_breaks=max_breaks,
        min_seg=min_seg,
        strategy=strategy,
    )
