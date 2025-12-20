from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

try:
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    wasserstein_distance = None  # type: ignore

def trace_powers(M: np.ndarray, max_power: int = 6) -> Dict[str, float]:
    """Gauge-invariant trace moments Tr(M^k)."""
    inv: Dict[str, float] = {}
    Mk = np.eye(M.shape[0])
    for k in range(1, max_power + 1):
        Mk = Mk @ M
        inv[f"tr_M{k}"] = float(np.trace(Mk))
    return inv

def spectrum(M: np.ndarray) -> np.ndarray:
    """Sorted real eigenvalues for symmetric matrices."""
    w = np.linalg.eigvalsh((M + M.T) / 2.0)
    return np.sort(w)

def spectral_distances(wA: np.ndarray, wB: np.ndarray) -> Dict[str, float]:
    wA = np.asarray(wA, dtype=float).reshape(-1)
    wB = np.asarray(wB, dtype=float).reshape(-1)
    if wA.shape != wB.shape:
        m = min(wA.size, wB.size)
        wA = wA[:m]; wB = wB[:m]
    l2 = float(np.linalg.norm(wA - wB) / (np.linalg.norm(wA) + 1e-12))
    linf = float(np.max(np.abs(wA - wB)) / (np.max(np.abs(wA)) + 1e-12))
    if wasserstein_distance is not None:
        w1 = float(wasserstein_distance(wA, wB) / (np.std(wA) + 1e-12))
    else:
        w1 = float(np.mean(np.abs(wA - wB)) / (np.std(wA) + 1e-12))
    return {"spec_l2_rel": l2, "spec_linf_rel": linf, "spec_w1_norm": w1}

def matrix_norms(M: np.ndarray) -> Dict[str, float]:
    M = np.asarray(M, dtype=float)
    return {
        "fro_norm": float(np.linalg.norm(M, ord="fro")),
        "spec_radius": float(np.max(np.abs(np.linalg.eigvalsh((M + M.T) / 2.0)))),
        "mean_abs": float(np.mean(np.abs(M))),
    }

def relative_error(a: float, b: float, eps: float = 1e-12) -> float:
    return float(abs(a - b) / (abs(a) + eps))

@dataclass(frozen=True)
class InvariantComparator:
    """Compute and compare invariants of two matrices A and B."""
    max_power: int = 6

    def compute(self, M: np.ndarray) -> Dict[str, float]:
        inv = {}
        inv.update(trace_powers(M, self.max_power))
        inv.update(matrix_norms(M))
        return inv

    def compare(self, A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
        invA = self.compute(A)
        invB = self.compute(B)
        out: Dict[str, float] = {}

        for k in range(1, self.max_power + 1):
            key = f"tr_M{k}"
            out[f"rel_{key}"] = relative_error(invA[key], invB[key])

        for key in ["fro_norm", "spec_radius", "mean_abs"]:
            out[f"rel_{key}"] = relative_error(invA[key], invB[key])

        wA = spectrum(A)
        wB = spectrum(B)
        out.update(spectral_distances(wA, wB))
        out["A_spec_std"] = float(np.std(wA))
        out["B_spec_std"] = float(np.std(wB))
        return out


def fro_rel(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """Relative Frobenius error ||A-B||_F / (||A||_F + eps)."""
    return float(np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(A, ord="fro") + eps))
