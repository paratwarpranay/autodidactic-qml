from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict
from .statevector_sim import fidelity

def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    return H @ K @ H

def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Alignment of kernel with labels y in {-1,+1}."""
    y = y.reshape(-1,1)
    Y = y @ y.T
    Kc = center_kernel(K)
    Yc = center_kernel(Y)
    num = np.sum(Kc * Yc)
    den = np.sqrt(np.sum(Kc*Kc) * np.sum(Yc*Yc) + 1e-12)
    return float(num / den)

@dataclass
class QuantumKernel:
    """Kernel defined by fidelity between encoded quantum states."""
    encode: Callable[[np.ndarray], np.ndarray]

    def gram(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        states = [self.encode(X[i]) for i in range(n)]
        K = np.zeros((n,n), dtype=float)
        for i in range(n):
            K[i,i] = 1.0
            for j in range(i+1, n):
                fij = fidelity(states[i], states[j])
                K[i,j] = K[j,i] = fij
        return K

def random_fourier_features_kernel(X: np.ndarray, gamma: float = 1.0, n_features: int = 256, seed: int = 0) -> np.ndarray:
    """Classical baseline kernel approximation (RBF via random Fourier features)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    W = rng.normal(scale=np.sqrt(2*gamma), size=(d, n_features))
    b = rng.uniform(0, 2*np.pi, size=(n_features,))
    Z = np.sqrt(2/n_features) * np.cos(X @ W + b)
    return Z @ Z.T
