import numpy as np
from analysis.correspondence_diagnostics import CorrespondenceDiagnostics
from matrix_models.hermitian_ensemble import HermitianEnsemble

def test_laplacian_diagnostics_present():
    ens = HermitianEnsemble(dim=8, seed=0)
    M = ens.sample()
    diag = CorrespondenceDiagnostics(max_power=4, laplacian_k=8)
    out = diag.compare(M, M)
    assert "lap_spec_l2_rel" in out
    assert abs(out["lap_spec_l2_rel"]) < 1e-10
    assert "lap_effdim_M" in out and "lap_effdim_Mhat" in out
