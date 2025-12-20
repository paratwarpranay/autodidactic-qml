import numpy as np
from matrix_models.hermitian_ensemble import HermitianEnsemble
from correspondence_maps.roundtrip import RoundTripMapper
from autodidactic_protocols.meta_objective import MetaObjectiveLearner

def test_meta_objective_updates_run():
    ens = HermitianEnsemble(dim=6, seed=0)
    M = ens.sample()
    rt = RoundTripMapper(width=16, seed=0)
    core, model = rt.build_model(M)
    learner = MetaObjectiveLearner(steps_per_update=5, batch_size=16)
    res = learner.update(model)
    assert "lam_mi" in res and "lam_stab" in res and "lam_comp" in res
