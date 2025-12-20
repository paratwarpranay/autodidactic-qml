import numpy as np
from analysis.phase_diagram import compute_transition_line

def test_transition_line_recovers_known_boundary():
    # Synthetic phase boundary: g_c(T) = a*T + b
    g = np.linspace(-1.0, 1.0, 31)
    T = np.linspace(0.0, 1.0, 9)
    a, b = 0.8, -0.2

    # Order parameter: plateau low then high with a smooth-ish step at boundary
    Z = np.zeros((T.size, g.size), dtype=float)
    for i, t in enumerate(T):
        gc = a * t + b
        # logistic step (continuous)
        Z[i, :] = 1.0 / (1.0 + np.exp(-20.0 * (g - gc)))

    pts = compute_transition_line(g, T, Z, model_type="continuous", max_breaks=1, min_seg=6, strategy="largest_jump")
    for p in pts:
        gc_true = a * p.axis2 + b
        assert np.isfinite(p.g_star)
        assert abs(p.g_star - gc_true) < 0.15  # tolerant under coarse grid
