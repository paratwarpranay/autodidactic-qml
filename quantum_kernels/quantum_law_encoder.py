from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

from .variational_ansatz import HardwareEfficientAnsatz
from .statevector_sim import fidelity

@dataclass
class VariationalLawEncoder:
    """Encode vectors into states using a variational ansatz with data reuploading.

    For each data vector x:
      params(x) = W x + b  (linear) then fed into ansatz.
    """
    n_qubits: int
    n_layers: int = 2
    seed: int = 0

    def __post_init__(self):
        self.ansatz = HardwareEfficientAnsatz(self.n_qubits, self.n_layers)
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(scale=0.2, size=(self.ansatz.n_params(),))  # per-parameter scaling
        self.b = rng.normal(scale=0.1, size=(self.ansatz.n_params(),))

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        # compress/expand x deterministically to match param count via hashing
        p = self.ansatz.n_params()
        v = np.zeros((p,), dtype=float)
        for i in range(p):
            v[i] = x[i % x.size] if x.size > 0 else 0.0
        params = self.W * v + self.b
        return self.ansatz.state(params)

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return fidelity(self.encode(x1), self.encode(x2))
