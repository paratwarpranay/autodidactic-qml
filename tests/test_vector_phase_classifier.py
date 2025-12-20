import numpy as np
from analysis.vector_phase_classifier import classify_phases_from_feature_grids
from analysis.composite_phase_boundary import score_grid_to_transition_line

def test_vector_classifier_recovers_boundary_line_synthetic():
    # synthetic boundary g_c(a2) = 0.5*a2 - 0.1
    g = np.linspace(-1.0, 1.0, 31)
    a2 = np.linspace(0.0, 1.0, 9)
    gc = 0.5*a2 - 0.1

    # build feature grids: left side phase A, right side phase B
    # Use three features (gap, effdim, mi_proxy) with different scales
    gap = np.zeros((a2.size, g.size))
    eff = np.zeros_like(gap)
    mi  = np.zeros_like(gap)

    rng = np.random.default_rng(0)
    for i, t in enumerate(a2):
        for j, gv in enumerate(g):
            phase = 1 if gv > gc[i] else 0
            gap[i, j] = (0.2 + 0.8*phase) + 0.03*rng.standard_normal()
            eff[i, j] = (1.0 + 1.5*phase) + 0.05*rng.standard_normal()
            mi[i, j]  = (0.1 + 0.6*phase) + 0.03*rng.standard_normal()

    feature_grids = {"lap_gap_Mhat": gap, "lap_effdim_Mhat": eff, "train_mi_proxy": mi}
    vp = classify_phases_from_feature_grids(feature_grids, a2, g, ["lap_gap_Mhat","lap_effdim_Mhat","train_mi_proxy"], k=2, seed=1)
    assert "score_grid" in vp
    score = np.array(vp["score_grid"], dtype=float)

    pts = score_grid_to_transition_line(g, a2, score, model_type="continuous", max_breaks=1, min_seg=6, strategy="largest_jump")
    for p in pts:
        true = 0.5*p.axis2 - 0.1
        assert np.isfinite(p.g_star)
        assert abs(p.g_star - true) < 0.2
