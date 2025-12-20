from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .action_functionals import ActionFunctional

@dataclass
class HMCSampler:
    """Hamiltonian Monte Carlo for symmetric matrices.

    We treat the matrix entries as coordinates in R^{N(N+1)/2}, but implement
    updates directly on full matrices with symmetry projection.

    H(M, P) = S(M) + 0.5 * ||P||_F^2
    """
    action: ActionFunctional
    step_size: float = 1e-2
    n_leapfrog: int = 20
    seed: Optional[int] = None

    def _sym(self, X: np.ndarray) -> np.ndarray:
        return (X + X.T) / 2.0

    def propose(self, M: np.ndarray, *, rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
        P = self._sym(rng.normal(size=M.shape))
        current_H = float(self.action.action(M) + 0.5 * np.sum(P * P))

        M_new = M.copy()
        P_new = P.copy()

        # Leapfrog integration (velocity Verlet)
        # Initial half step for momentum
        P_new = self._sym(P_new - 0.5 * self.step_size * self.action.grad(M_new))

        for i in range(self.n_leapfrog):
            # Full step for position
            M_new = self._sym(M_new + self.step_size * P_new)
            # Full step for momentum, except at end
            if i < self.n_leapfrog - 1:
                P_new = self._sym(P_new - self.step_size * self.action.grad(M_new))

        # Final half step for momentum
        P_new = self._sym(P_new - 0.5 * self.step_size * self.action.grad(M_new))

        proposed_H = float(self.action.action(M_new) + 0.5 * np.sum(P_new * P_new))
        return M_new, current_H, proposed_H

    def step(self, M: np.ndarray, *, rng: Optional[np.random.Generator]=None) -> Tuple[np.ndarray, bool]:
        r = rng if rng is not None else np.random.default_rng(self.seed)
        M_prop, Hc, Hp = self.propose(M, rng=r)
        accept_prob = float(np.exp(min(0.0, Hc - Hp)))
        if r.random() < accept_prob:
            return M_prop, True
        return M, False

    def run(self, M0: np.ndarray, steps: int = 200, *, rng: Optional[np.random.Generator]=None) -> np.ndarray:
        M = M0.copy()
        r = rng if rng is not None else np.random.default_rng(self.seed)
        for _ in range(steps):
            M, _ = self.step(M, rng=r)
        return M
