import numpy as np
from analysis.change_point_detection import (
    fit_piecewise_linear_continuous,
    fit_piecewise_constant,
    best_model_by_bic,
    breakpoint_x_values,
    bootstrap_breakpoints,
)

def test_continuous_piecewise_prefers_break():
    x = np.linspace(-1.0, 1.0, 21)
    y = np.where(x < 0.0, 0.0 + 0.0 * x, 1.0 + 0.0 * x)  # step
    models = fit_piecewise_constant(x, y, max_breaks=1, min_seg=5)
    best = best_model_by_bic(models)
    bx = breakpoint_x_values(x, best)
    assert len(bx) == 1
    assert abs(bx[0]) < 0.2

def test_bootstrap_returns_ci():
    x = np.linspace(-1.0, 1.0, 21)
    y = np.where(x < 0.0, 0.5 * x, 2.0 * x) + 1e-3*np.sin(9*x)
    boot = bootstrap_breakpoints(x, y, n_boot=50, max_breaks=1, min_seg=5, mode="linear", seed=0)
    assert boot["n_boot"] == 50
    assert isinstance(boot["ci"], list)
