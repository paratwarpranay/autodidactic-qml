import numpy as np
from analysis.change_point_detection import transition_report

def test_change_point_detects_single_break():
    # synthetic: two regimes with different slopes
    x = np.linspace(-1.0, 1.0, 21)
    y = np.where(x < 0.0, 0.5 * x + 0.0, 2.0 * x + 0.0)
    # tiny noise
    y = y + 1e-4 * np.sin(10*x)
    rep = transition_report(x, y, max_breaks=2, min_seg=5)
    breaks = rep["best"]["breaks_x"]
    # should place a break near 0
    assert len(breaks) >= 1
    assert abs(breaks[0]) < 0.2
