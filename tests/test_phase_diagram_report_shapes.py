import numpy as np
from analysis.phase_diagram import compute_transition_line

def test_compute_transition_line_shapes():
    g = np.linspace(-0.5, 0.5, 11)
    a2 = np.linspace(0.1, 0.3, 4)
    Z = np.random.default_rng(0).standard_normal((a2.size, g.size))
    pts = compute_transition_line(g, a2, Z, model_type="linear", max_breaks=2, min_seg=3)
    assert len(pts) == a2.size
