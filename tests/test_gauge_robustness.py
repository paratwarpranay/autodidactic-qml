import numpy as np
from analysis.gauge_tools import random_orthogonal, gauge_transform
from analysis.invariant_metrics import InvariantComparator
from matrix_models.hermitian_ensemble import HermitianEnsemble

def test_gauge_invariants_match_under_conjugation():
    rng = np.random.default_rng(0)
    ens = HermitianEnsemble(dim=10, seed=0)
    M = ens.sample(rng=rng)
    U = random_orthogonal(10, rng=rng)
    Mp = gauge_transform(M, U)

    comp = InvariantComparator(max_power=4)
    out = comp.compare(M, Mp)

    # Trace moments + spectrum should be essentially identical up to numerical tolerance.
    assert out["rel_tr_M2"] < 1e-8
    assert out["spec_l2_rel"] < 1e-8
