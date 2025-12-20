from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    A = rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity
    d = np.sign(np.diag(R))
    Q = Q * d
    return Q

def conjugate(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Gauge transform by conjugation: M -> U M U^T."""
    return U @ M @ U.T

def invariants(M: np.ndarray, max_power: int = 6) -> Dict[str, float]:
    """Compute simple gauge-invariant traces Tr(M^k)."""
    inv: Dict[str, float] = {}
    Mk = np.eye(M.shape[0])
    for k in range(1, max_power + 1):
        Mk = Mk @ M
        inv[f"tr_M{k}"] = float(np.trace(Mk))
    inv["fro_norm"] = float(np.linalg.norm(M, ord="fro"))
    inv["spec_radius"] = float(np.max(np.abs(np.linalg.eigvalsh(M))))
    return inv

@dataclass
class GaugeOrbitSampler:
    """Sample points on the gauge orbit by random conjugations."""
    dim: int
    seed: Optional[int] = None

    def sample(self, M: np.ndarray, n: int = 8) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        out = []
        for _ in range(n):
            U = random_orthogonal(self.dim, rng)
            out.append(conjugate(M, U))
        return np.stack(out, axis=0)
