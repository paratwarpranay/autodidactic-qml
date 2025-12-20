from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .change_point_detection import (
    fit_piecewise_linear,
    fit_piecewise_linear_continuous,
    fit_piecewise_constant,
    best_model_by_bic,
    breakpoint_x_values,
)

@dataclass(frozen=True)
class TransitionPoint:
    axis2: float
    g_star: float
    bic: float
    sse: float
    n_breaks: int
    breaks_x: List[float]

def _fit_models(xg: np.ndarray, yo: np.ndarray, model_type: str, max_breaks: int, min_seg: int):
    if model_type == "constant":
        models = fit_piecewise_constant(xg, yo, max_breaks=max_breaks, min_seg=min_seg)
    elif model_type == "continuous":
        models = fit_piecewise_linear_continuous(xg, yo, max_breaks=max_breaks, min_seg=min_seg)
    else:
        models = fit_piecewise_linear(xg, yo, max_breaks=max_breaks, min_seg=min_seg)
    return models


def midpoint_threshold_crossing(
    xg: np.ndarray,
    yo: np.ndarray,
    k: int = 0,
    ref_x: float | None = None,
) -> float:
    """Estimate transition as the x where yo crosses the midpoint between low/high plateaus.

    Works well for smooth, monotonic-ish order parameters (e.g., logistic steps).
    If multiple crossings exist, choose the one closest to ref_x (if provided), otherwise the first.
    
    If no crossings found, returns the midpoint of the x range as a fallback.
    """
    xg = np.asarray(xg, dtype=float).reshape(-1)
    yo = np.asarray(yo, dtype=float).reshape(-1)
    n = int(xg.size)
    if n < 3:
        return float(0.5 * (xg[0] + xg[-1])) if n >= 2 else float("nan")
    if k <= 0:
        k = max(3, min(8, n // 5))  # robust default; e.g., n=31 -> 6
    y_low = float(np.median(yo[:k]))
    y_high = float(np.median(yo[-k:]))
    thr = 0.5 * (y_low + y_high)

    crossings: list[float] = []
    for j in range(1, n):
        y0 = float(yo[j - 1] - thr)
        y1 = float(yo[j] - thr)
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            crossings.append(float(xg[j - 1]))
        if y1 == 0.0:
            crossings.append(float(xg[j]))
        if y0 * y1 < 0.0:
            # linear interpolation
            t = abs(y0) / (abs(y0) + abs(y1))
            crossings.append(float(xg[j - 1] + t * (xg[j] - xg[j - 1])))

    if not crossings:
        # Fallback: return midpoint of x range
        return float(0.5 * (xg[0] + xg[-1]))
    if ref_x is None:
        return float(crossings[0])
    crossings = sorted(crossings, key=lambda c: abs(c - float(ref_x)))
    return float(crossings[0])

def pick_transition_from_breaks(
    xg: np.ndarray,
    yo: np.ndarray,
    breaks_x: List[float],
    strategy: str = "largest_jump",
) -> float:
    """Choose a single transition g*.

    For smooth transitions (e.g., logistic), breakpoints from piecewise fits can drift.
    So for strategy='largest_jump' we estimate g* as the **midpoint-threshold crossing**
    between low/high plateaus, optionally anchored near the strongest breakpoint.

    strategy:
      - 'first': first breakpoint
      - 'largest_jump': midpoint crossing (fallback to strongest breakpoint)
      - 'middle': median breakpoint
    """
    if not breaks_x:
        return float("nan")
    breaks_x = [float(b) for b in breaks_x]

    if strategy == "first":
        return float(breaks_x[0])
    if strategy == "middle":
        return float(np.median(breaks_x))

    # default: 'largest_jump'
    xg = np.asarray(xg, dtype=float).reshape(-1)
    yo = np.asarray(yo, dtype=float).reshape(-1)

    # choose a reference breakpoint = one with biggest mean shift
    jumps = []
    for bx in breaks_x:
        left = yo[xg <= bx]
        right = yo[xg > bx]
        if left.size == 0 or right.size == 0:
            jumps.append(-np.inf)
        else:
            jumps.append(float(abs(np.mean(right) - np.mean(left))))
    ref = breaks_x[int(np.argmax(jumps))]

    # attempt plateau-midpoint crossing
    g_star = midpoint_threshold_crossing(xg, yo, ref_x=ref)
    if np.isfinite(g_star):
        return float(g_star)

    # fallback: strongest breakpoint
    return float(ref)

def compute_transition_line(
    g_values: np.ndarray,
    axis2_values: np.ndarray,
    order_grid: np.ndarray,
    model_type: str = "continuous",
    max_breaks: int = 2,
    min_seg: int = 3,
    strategy: str = "largest_jump",
) -> List[TransitionPoint]:
    """Given a 2D order parameter grid order_grid[axis2_idx, g_idx], find g*(axis2).

    Returns a list of TransitionPoint entries, one per axis2 value.
    
    When BIC selects a model with zero breakpoints (smooth transition), we fall
    back to midpoint-threshold crossing to estimate g*.
    """
    g_values = np.asarray(g_values, dtype=float).reshape(-1)
    axis2_values = np.asarray(axis2_values, dtype=float).reshape(-1)
    order_grid = np.asarray(order_grid, dtype=float)
    assert order_grid.shape == (axis2_values.size, g_values.size), "order_grid shape mismatch"

    points: List[TransitionPoint] = []
    for i, a2 in enumerate(axis2_values):
        yo = order_grid[i, :]
        # drop NaNs
        mask = np.isfinite(yo) & np.isfinite(g_values)
        xg = g_values[mask]
        yy = yo[mask]
        if xg.size < max(2*min_seg, 5):
            points.append(TransitionPoint(axis2=float(a2), g_star=float("nan"), bic=float("nan"),
                                          sse=float("nan"), n_breaks=0, breaks_x=[]))
            continue
        models = _fit_models(xg, yy, model_type=model_type, max_breaks=max_breaks, min_seg=min_seg)
        best = best_model_by_bic(models)
        breaks_x = breakpoint_x_values(xg, best)
        
        # Estimate g* from breakpoints if available
        if breaks_x:
            g_star = pick_transition_from_breaks(xg, yy, breaks_x, strategy=strategy)
        else:
            # No breakpoints found (smooth transition) - use midpoint crossing fallback
            g_star = midpoint_threshold_crossing(xg, yy)
            
        points.append(TransitionPoint(axis2=float(a2), g_star=float(g_star),
                                      bic=float(best.bic), sse=float(best.sse),
                                      n_breaks=len(breaks_x), breaks_x=[float(b) for b in breaks_x]))
    return points

def line_to_arrays(points: List[TransitionPoint]) -> Tuple[np.ndarray, np.ndarray]:
    a2 = np.array([p.axis2 for p in points], dtype=float)
    gs = np.array([p.g_star for p in points], dtype=float)
    return a2, gs
