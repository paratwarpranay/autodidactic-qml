"""Matrix model sampling and dynamics.

This module provides:
- HermitianEnsemble: Random symmetric matrix generation
- Action functionals: QuadraticAction, CubicAction, QuarticAction
- Samplers: LangevinSampler, HMCSampler
- Gauge tools: random_orthogonal, conjugate, invariants, GaugeOrbitSampler
"""

from .hermitian_ensemble import HermitianEnsemble
from .action_functionals import (
    ActionFunctional,
    QuadraticAction,
    CubicAction,
    QuarticAction,
    make_action,
)
from .sampler_langevin import LangevinSampler
from .sampler_hmc import HMCSampler
from .gauge_symmetries import (
    random_orthogonal,
    conjugate,
    invariants,
    GaugeOrbitSampler,
)

__all__ = [
    "HermitianEnsemble",
    "ActionFunctional",
    "QuadraticAction",
    "CubicAction",
    "QuarticAction",
    "make_action",
    "LangevinSampler",
    "HMCSampler",
    "random_orthogonal",
    "conjugate",
    "invariants",
    "GaugeOrbitSampler",
]
