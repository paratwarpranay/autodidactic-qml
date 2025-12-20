from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .statevector_sim import Circuit

@dataclass
class ConjugationEquivariantFeatureMap:
    """A 'gauge-aware' feature map for matrix inputs.

    We build features from gauge invariants (trace powers) and embed them into
    rotation angles. This yields *invariance to conjugation* by construction.

    Input: invariants dict (tr_Mk, fro_norm, spec_radius)
    Output: circuit ops.
    """
    n_qubits: int
    n_layers: int = 2

    def _angles(self, inv: dict) -> np.ndarray:
        # deterministic ordering
        keys = sorted(inv.keys())
        v = np.array([inv[k] for k in keys], dtype=float)
        v = v / (np.linalg.norm(v) + 1e-12)
        # map to angles
        return 2*np.pi*np.tanh(v)

    def ops_from_invariants(self, inv: dict) -> List[Tuple[str, Tuple]]:
        ang = self._angles(inv)
        ops: List[Tuple[str, Tuple]] = []
        # initialize
        for q in range(self.n_qubits):
            ops.append(("H", (q,)))
        idx = 0
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                a1 = float(ang[idx % len(ang)]); idx += 1
                a2 = float(ang[idx % len(ang)]); idx += 1
                ops.append(("RY", (q, a1)))
                ops.append(("RZ", (q, a2)))
            for q in range(self.n_qubits):
                ops.append(("CNOT", (q, (q+1) % self.n_qubits)))
        return ops

    def state_from_invariants(self, inv: dict) -> np.ndarray:
        c = Circuit(self.n_qubits)
        return c.run(self.ops_from_invariants(inv))
