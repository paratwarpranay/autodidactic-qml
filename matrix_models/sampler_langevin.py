from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .action_functionals import ActionFunctional

@dataclass
class LangevinSampler:
    """Overdamped Langevin sampler for matrix models.

    dM = -∇S(M) dt + sqrt(2 T dt) dW

    We keep symmetry by re-symmetrizing each step.
    """
    action: ActionFunctional
    dt: float = 1e-3
    temperature: float = 1e-2
    seed: Optional[int] = None

    def step(self, M: np.ndarray, *, rng: Optional[np.random.Generator]=None) -> np.ndarray:
        r = rng if rng is not None else np.random.default_rng(self.seed)
        grad = self.action.grad(M)
        noise = r.normal(size=M.shape) * np.sqrt(2.0 * self.temperature * self.dt)
        M_new = M - grad * self.dt + noise
        M_new = (M_new + M_new.T) / 2.0
        return M_new

    def run(self, M0: np.ndarray, steps: int = 1000, *, rng: Optional[np.random.Generator]=None) -> np.ndarray:
        M = M0.copy()
        r = rng if rng is not None else np.random.default_rng(self.seed)
        for _ in range(steps):
            M = self.step(M, rng=r)
        return M
