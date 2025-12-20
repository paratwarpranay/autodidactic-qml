import numpy as np
from analysis.vector_phase_classifier import classify_phases_from_feature_grids

def test_vector_classifier_handles_nans():
    g = np.linspace(-0.5, 0.5, 11)
    a2 = np.linspace(0.0, 1.0, 5)
    gap = np.random.default_rng(0).standard_normal((a2.size, g.size))
    eff = np.random.default_rng(1).standard_normal((a2.size, g.size))
    mi = np.random.default_rng(2).standard_normal((a2.size, g.size))
    mi[0, :] = np.nan  # missing slice

    feature_grids = {"lap_gap_Mhat": gap, "lap_effdim_Mhat": eff, "train_mi_proxy": mi}
    vp = classify_phases_from_feature_grids(feature_grids, a2, g, ["lap_gap_Mhat","lap_effdim_Mhat","train_mi_proxy"], k=2, seed=0)
    assert "label_grid" in vp
