import numpy as np
from matrix_models.hermitian_ensemble import HermitianEnsemble
from correspondence_maps.roundtrip import RoundTripMapper
from analysis.invariant_metrics import InvariantComparator

def test_roundtrip_metrics_run():
    ens = HermitianEnsemble(dim=8, seed=0)
    M = ens.sample()
    rt = RoundTripMapper(width=32, seed=0)
    core, model = rt.build_model(M)
    Mhat = rt.extract_matrix(core)
    comp = InvariantComparator(max_power=4)
    out = comp.compare(M, Mhat)
    assert "spec_l2_rel" in out
    assert "rel_tr_M2" in out
    assert out["spec_l2_rel"] >= 0.0
