import numpy as np
from analysis.logistic_boundary import fit_logistic_regression, boundary_gstar_from_scores, bootstrap_logistic_boundary
from analysis.vector_phase_classifier import classify_phases_from_feature_grids

def test_logistic_boundary_band_covers_true_line_synthetic():
    # Synthetic linear boundary in feature space induced by g and a2
    g = np.linspace(-1.0, 1.0, 31)
    a2 = np.linspace(0.0, 1.0, 9)
    rng = np.random.default_rng(0)

    # Features: [gap, effdim, mi] each correlates with phase
    gap = np.zeros((a2.size, g.size))
    eff = np.zeros_like(gap)
    mi = np.zeros_like(gap)
    # true boundary: g_c = 0.6*a2 - 0.15
    for i, t in enumerate(a2):
        gc = 0.6*t - 0.15
        for j, gv in enumerate(g):
            ph = 1 if gv > gc else 0
            gap[i,j] = (0.3 + 0.7*ph) + 0.02*rng.standard_normal()
            eff[i,j] = (1.2 + 1.0*ph) + 0.03*rng.standard_normal()
            mi[i,j]  = (0.2 + 0.5*ph) + 0.02*rng.standard_normal()

    feature_grids = {"lap_gap_Mhat": gap, "lap_effdim_Mhat": eff, "train_mi_proxy": mi}
    vp = classify_phases_from_feature_grids(feature_grids, a2, g, ["lap_gap_Mhat","lap_effdim_Mhat","train_mi_proxy"], k=2, seed=1)
    X = np.array(vp["X_raw"], dtype=float)
    idx_map = np.array(vp["idx_map"], dtype=int)
    y = np.array(vp["labels_flat"], dtype=int)
    y = (y > 0).astype(int)

    model = fit_logistic_regression(X, y, l2=1e-2, max_iter=80, tol=1e-6, seed=0)
    p = model.predict_proba(X)
    # rebuild score grid
    S = np.full((a2.size, g.size), np.nan, dtype=float)
    for pp, (ia2, ig) in zip(p, idx_map):
        S[int(ia2), int(ig)] = float(pp)

    _, g_star = boundary_gstar_from_scores(g, a2, S, threshold=0.5)
    for i, t in enumerate(a2):
        true = 0.6*t - 0.15
        assert np.isfinite(g_star[i])
        assert abs(g_star[i] - true) < 0.25

    band = bootstrap_logistic_boundary(X, y, g, a2, idx_map, l2=1e-2, n_boot=60, seed=0, threshold=0.5)
    p05 = np.array(band["g_star_p05"], dtype=float)
    p95 = np.array(band["g_star_p95"], dtype=float)
    # true line should usually fall within band for most slices (tolerant)
    cover = 0
    for i, t in enumerate(a2):
        true = 0.6*t - 0.15
        if np.isfinite(p05[i]) and np.isfinite(p95[i]) and (p05[i] - 0.35) <= true <= (p95[i] + 0.35):
            cover += 1
    assert cover >= int(0.6 * a2.size)
