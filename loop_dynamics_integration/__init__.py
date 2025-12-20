"""Loop dynamics integration - time-coupled memory and reflexive updates.

This module provides mechanisms for Loop Theory-style recursive reasoning:
- TimeCoupledMemory: FIFO buffer with time-symmetric smoothing
- ReflexiveUpdater: Updates scaled by parameter state and gradient statistics
- TimeSymmetricGradientSmoother: Retrocausal-style gradient smoothing
- FixedPointFinder: Find approximate fixed points of learned mappings
- EndogenousUpdate: Topology-driven parameter updates
- SpectralEndogenousUpdate: Curvature-based endogenous updates
- TopologyGuidedOptimizer: Optimizer wrapper with topology modulation

Note: All components require PyTorch. Use lazy imports.
"""

def __getattr__(name):
    if name == "TimeCoupledMemory":
        from .time_coupled_memory import TimeCoupledMemory
        return TimeCoupledMemory
    elif name == "ReflexiveUpdater":
        from .reflexive_update_rules import ReflexiveUpdater
        return ReflexiveUpdater
    elif name == "TimeSymmetricGradientSmoother":
        from .retrocausal_feedback import TimeSymmetricGradientSmoother
        return TimeSymmetricGradientSmoother
    elif name == "FixedPointFinder":
        from .operator_fixed_points import FixedPointFinder
        return FixedPointFinder
    elif name == "EndogenousUpdate":
        from .endogenous_update import EndogenousUpdate
        return EndogenousUpdate
    elif name == "SpectralEndogenousUpdate":
        from .endogenous_update import SpectralEndogenousUpdate
        return SpectralEndogenousUpdate
    elif name == "TopologyGuidedOptimizer":
        from .endogenous_update import TopologyGuidedOptimizer
        return TopologyGuidedOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TimeCoupledMemory",
    "ReflexiveUpdater",
    "TimeSymmetricGradientSmoother",
    "FixedPointFinder",
    "EndogenousUpdate",
    "SpectralEndogenousUpdate",
    "TopologyGuidedOptimizer",
]
