from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .statevector_sim import Circuit

@dataclass
class HardwareEfficientAnsatz:
    """A small, expressive variational circuit family.

    Structure per layer:
    - RY on each qubit
    - RZ on each qubit
    - nearest-neighbor CNOT ring
    """
    n_qubits: int
    n_layers: int = 2

    def n_params(self) -> int:
        return self.n_layers * self.n_qubits * 2

    def ops(self, params: np.ndarray) -> List[Tuple[str, Tuple]]:
        params = np.asarray(params, dtype=float).reshape(self.n_layers, self.n_qubits, 2)
        ops: List[Tuple[str, Tuple]] = []
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                ops.append(("RY", (q, float(params[l,q,0]))))
                ops.append(("RZ", (q, float(params[l,q,1]))))
            for q in range(self.n_qubits):
                ops.append(("CNOT", (q, (q+1) % self.n_qubits)))
        return ops

    def state(self, params: np.ndarray) -> np.ndarray:
        c = Circuit(self.n_qubits)
        return c.run(self.ops(params))
