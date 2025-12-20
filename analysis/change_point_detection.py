from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y ≈ a x + b and return (a, b, sse)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2:
        a = 0.0
        b = float(y.mean() if y.size else 0.0)
        sse = float(np.sum((y - b) ** 2))
        return a, b, sse

    xm = float(x.mean())
    ym = float(y.mean())
    vx = float(np.sum((x - xm) ** 2)) + 1e-12
    cov = float(np.sum((x - xm) * (y - ym)))
    a = cov / vx
    b = ym - a * xm
    r = y - (a * x + b)
    sse = float(np.sum(r * r))
    return float(a), float(b), sse

def _bic(n: int, sse: float, n_params: int) -> float:
    # BIC for Gaussian residuals with unknown variance (up to constant)
    # n*log(SSE/n) + k*log(n)
    sse = max(float(sse), 1e-12)
    return float(n * np.log(sse / max(n, 1)) + n_params * np.log(max(n, 2)))

def _segments_from_breaks(n: int, breaks: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """Convert break indices into [start, end) segments."""
    idx = (0,) + breaks + (n,)
    return [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]

@dataclass(frozen=True)
class PiecewiseFitResult:
    breaks: Tuple[int, ...]            # breakpoints as indices into x/y (between points)
    segments: List[Tuple[int, int]]    # [start,end) ranges
    params: List[Tuple[float, float]]  # (a,b) per segment
    sse: float
    bic: float

def fit_piecewise_linear(
    x: np.ndarray,
    y: np.ndarray,
    max_breaks: int = 2,
    min_seg: int = 3,
) -> Dict[str, PiecewiseFitResult]:
    """Fit piecewise-linear models with 0..max_breaks breakpoints and score by BIC.

    Breakpoints are chosen by brute-force enumeration; this is fine for n<=~50.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert x.size == y.size, "x and y must have same length"
    n = int(x.size)

    results: Dict[str, PiecewiseFitResult] = {}

    # Candidate breakpoints are integers in [1, n-1]
    candidates = list(range(1, n))
    for m in range(0, max_breaks + 1):
        best: Optional[PiecewiseFitResult] = None
        for breaks in itertools.combinations(candidates, m):
            segs = _segments_from_breaks(n, breaks)
            # enforce minimum segment length
            if any((j - i) < min_seg for i, j in segs):
                continue
            params = []
            sse_total = 0.0
            for i, j in segs:
                a, b, sse = _fit_linear(x[i:j], y[i:j])
                params.append((a, b))
                sse_total += sse
            # parameters: (a,b) per segment
            k = 2 * len(segs)
            bic = _bic(n, sse_total, k)
            cur = PiecewiseFitResult(breaks=tuple(breaks), segments=segs, params=params, sse=float(sse_total), bic=float(bic))
            if best is None or cur.bic < best.bic:
                best = cur
        if best is not None:
            results[f"{m}_breaks"] = best

    return results

def best_model_by_bic(results: Dict[str, PiecewiseFitResult]) -> PiecewiseFitResult:
    """Select best model by BIC with Occam's razor tie-breaking.
    
    When ΔBIC ≤ 2 between models, prefer the simpler one (fewer breaks).
    This follows the standard BIC interpretation:
        ΔBIC < 2: weak evidence for more complex model
        ΔBIC 2-6: positive evidence
        ΔBIC 6-10: strong evidence
        ΔBIC > 10: very strong evidence
    """
    assert len(results) > 0, "No models were fit (increase n or lower min_seg)"
    
    # Sort by number of breaks (simpler first)
    sorted_models = sorted(results.values(), key=lambda r: len(r.breaks))
    
    # Start with simplest model
    best = sorted_models[0]
    
    # Only accept more complex model if ΔBIC > 2 (positive evidence)
    for model in sorted_models[1:]:
        delta_bic = best.bic - model.bic  # positive if model is better
        if delta_bic > 2.0:
            best = model
    
    return best

def breakpoint_x_values(x: np.ndarray, fit: PiecewiseFitResult) -> List[float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    vals = []
    for b in fit.breaks:
        # breakpoint occurs between b-1 and b; represent as mid-point
        vals.append(float(0.5 * (x[b - 1] + x[b])))
    return vals

def transition_report(x: np.ndarray, y: np.ndarray, max_breaks: int = 2, min_seg: int = 3) -> Dict[str, object]:
    """Convenience: fit models and return a compact report."""
    models = fit_piecewise_linear(x, y, max_breaks=max_breaks, min_seg=min_seg)
    best = best_model_by_bic(models)
    return {
        "n": int(np.asarray(x).size),
        "best": {
            "breaks_idx": list(best.breaks),
            "breaks_x": breakpoint_x_values(x, best),
            "bic": float(best.bic),
            "sse": float(best.sse),
            "segments": [list(s) for s in best.segments],
            "params": [{"a": float(a), "b": float(b)} for (a, b) in best.params],
        },
        "candidates": {k: {"bic": float(v.bic), "breaks_x": breakpoint_x_values(x, v)} for k, v in models.items()},
    }


def _fit_linear_continuous(x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> Tuple[float, float, float]:
    """Fit y ≈ a x + b with constraint that y(x0)=y0. Then b = y0 - a*x0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 1:
        return 0.0, float(y0), 0.0
    # minimize sum (y - (a x + (y0-a x0)))^2 = sum ((y - y0) - a (x - x0))^2
    xd = x - float(x0)
    yd = y - float(y0)
    denom = float(np.sum(xd * xd)) + 1e-12
    a = float(np.sum(xd * yd) / denom)
    b = float(y0 - a * x0)
    r = y - (a * x + b)
    sse = float(np.sum(r * r))
    return a, b, sse

def fit_piecewise_linear_continuous(
    x: np.ndarray,
    y: np.ndarray,
    max_breaks: int = 2,
    min_seg: int = 3,
) -> Dict[str, PiecewiseFitResult]:
    """Piecewise-linear with continuity enforced at breakpoints.

    We enforce that adjacent segments meet at each breakpoint. This reduces degrees of freedom,
    often making detected transitions less 'wiggly' and more physical.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert x.size == y.size
    n = int(x.size)
    results: Dict[str, PiecewiseFitResult] = {}
    candidates = list(range(1, n))

    for m in range(0, max_breaks + 1):
        best: Optional[PiecewiseFitResult] = None
        for breaks in itertools.combinations(candidates, m):
            segs = _segments_from_breaks(n, breaks)
            if any((j - i) < min_seg for i, j in segs):
                continue

            params = []
            sse_total = 0.0

            # anchor at first segment intercept free (a,b unconstrained)
            i0, j0 = segs[0]
            a0, b0, sse0 = _fit_linear(x[i0:j0], y[i0:j0])
            params.append((a0, b0))
            sse_total += sse0

            # propagate continuity: each next segment constrained to meet at breakpoint x_b
            for si in range(1, len(segs)):
                i, j = segs[si]
                bidx = breaks[si-1]  # breakpoint index separating prev and this segment
                # meeting point at x_b = mid between x[b-1] and x[b], y_b predicted from prev segment
                x_b = float(0.5 * (x[bidx - 1] + x[bidx]))
                a_prev, b_prev = params[si - 1]
                y_b = float(a_prev * x_b + b_prev)
                a, b, sse = _fit_linear_continuous(x[i:j], y[i:j], x0=x_b, y0=y_b)
                params.append((a, b))
                sse_total += sse

            # parameter count: first segment 2 params, each additional segment adds 1 slope param
            k = 2 + (len(segs) - 1) * 1
            bic = _bic(n, sse_total, k)
            cur = PiecewiseFitResult(breaks=tuple(breaks), segments=segs, params=params, sse=float(sse_total), bic=float(bic))
            if best is None or cur.bic < best.bic:
                best = cur
        if best is not None:
            results[f"{m}_breaks"] = best
    return results

def fit_piecewise_constant(
    x: np.ndarray,
    y: np.ndarray,
    max_breaks: int = 2,
    min_seg: int = 3,
) -> Dict[str, PiecewiseFitResult]:
    """Piecewise-constant model (order parameter with plateaus)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert x.size == y.size
    n = int(x.size)
    results: Dict[str, PiecewiseFitResult] = {}
    candidates = list(range(1, n))

    for m in range(0, max_breaks + 1):
        best: Optional[PiecewiseFitResult] = None
        for breaks in itertools.combinations(candidates, m):
            segs = _segments_from_breaks(n, breaks)
            if any((j - i) < min_seg for i, j in segs):
                continue
            params = []
            sse_total = 0.0
            for i, j in segs:
                mu = float(np.mean(y[i:j]))
                r = y[i:j] - mu
                sse = float(np.sum(r * r))
                # encode as (a=0,b=mu) to reuse PiecewiseFitResult
                params.append((0.0, mu))
                sse_total += sse
            k = 1 * len(segs)  # one mean per segment
            bic = _bic(n, sse_total, k)
            cur = PiecewiseFitResult(breaks=tuple(breaks), segments=segs, params=params, sse=float(sse_total), bic=float(bic))
            if best is None or cur.bic < best.bic:
                best = cur
        if best is not None:
            results[f"{m}_breaks"] = best
    return results

def bootstrap_breakpoints(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 200,
    max_breaks: int = 2,
    min_seg: int = 3,
    mode: str = "linear",
    seed: int = 0,
) -> Dict[str, object]:
    """Bootstrap uncertainty over breakpoint locations.

    We resample (x,y) pairs with replacement, fit the best model, and collect breaks_x.
    This is a simple way to quantify robustness under finite-N noise.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    breaks_samples = []

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        # sort by xb to preserve sweep order
        order = np.argsort(xb)
        xb = xb[order]
        yb = yb[order]

        if mode == "constant":
            models = fit_piecewise_constant(xb, yb, max_breaks=max_breaks, min_seg=min_seg)
        elif mode == "continuous":
            models = fit_piecewise_linear_continuous(xb, yb, max_breaks=max_breaks, min_seg=min_seg)
        else:
            models = fit_piecewise_linear(xb, yb, max_breaks=max_breaks, min_seg=min_seg)

        best = best_model_by_bic(models)
        breaks_x = breakpoint_x_values(xb, best)
        breaks_samples.append(breaks_x)

    # Align samples by break index (pad with NaN)
    mmax = max((len(b) for b in breaks_samples), default=0)
    A = np.full((len(breaks_samples), mmax), np.nan, dtype=float)
    for i, b in enumerate(breaks_samples):
        for j, v in enumerate(b):
            A[i, j] = float(v)

    ci = []
    for j in range(mmax):
        col = A[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            ci.append({"break": j, "median": float("nan"), "p05": float("nan"), "p95": float("nan")})
        else:
            ci.append({
                "break": j,
                "median": float(np.median(col)),
                "p05": float(np.quantile(col, 0.05)),
                "p95": float(np.quantile(col, 0.95)),
            })

    return {"n_boot": int(n_boot), "mode": mode, "ci": ci}
