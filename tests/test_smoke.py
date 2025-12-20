import numpy as np
import torch

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from correspondence_maps.matrix_to_rnn import MatrixToCyclicRNN
from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from quantum_kernels.quantum_law_encoder import VariationalLawEncoder
from quantum_kernels.kernel_expressivity import QuantumKernel

def test_matrix_langevin_runs():
    ens = HermitianEnsemble(dim=8, seed=0)
    M0 = ens.sample()
    sampler = LangevinSampler(action=CubicAction(g=0.1), seed=0)
    M = sampler.run(M0, steps=50)
    assert M.shape == (8,8)
    assert np.allclose(M, M.T, atol=1e-6)

def test_matrix_to_rnn_and_update():
    ens = HermitianEnsemble(dim=8, seed=0)
    M = ens.sample()
    mapper = MatrixToCyclicRNN(hidden_size=32, input_size=32, seed=0)
    core = mapper.build(M)

    class Wrapped(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
            self.input_size = core.input_size
            self.hidden_size = core.hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size)
        def forward(self, x, steps=3):
            xh = self.fc1(x)
            y, stats = self.core(xh, steps=steps)
            return self.fc2(stats["h_final"])

    model = Wrapped(core)
    learner = SelfConsistencyLearner(steps_per_update=10, batch_size=32)
    res = learner.update(model)
    assert "loss" in res
    assert res["loss"] >= 0.0

def test_quantum_kernel_gram():
    enc = VariationalLawEncoder(n_qubits=4, n_layers=2, seed=0)
    qk = QuantumKernel(encode=enc.encode)
    X = np.random.default_rng(0).normal(size=(10, 8))
    K = qk.gram(X)
    assert K.shape == (10,10)
    assert np.allclose(np.diag(K), 1.0)
