"""Autodidactic learning protocols - systems that learn their own objectives.

This module provides learning rules without external supervision:
- SelfConsistencyLearner: Minimize ||f(x) - x||^2 for fixed-point attractors
- MutualInformationLearner: Maximize MI between input and internal state
- MetaObjectiveLearner: Evolving Lagrangian multipliers for objective self-construction
- RecursiveSelfModifier: Meta-optimizer that learns its own learning rules
- RGLearner: Renormalization-group inspired scale-invariant learning

Note: All components require PyTorch. Use lazy imports.
"""

def __getattr__(name):
    if name == "SelfConsistencyLearner":
        from .unsupervised_update import SelfConsistencyLearner
        return SelfConsistencyLearner
    elif name == "MutualInformationLearner":
        from .mutual_information_drive import MutualInformationLearner
        return MutualInformationLearner
    elif name == "gaussian_mi_proxy":
        from .mutual_information_drive import gaussian_mi_proxy
        return gaussian_mi_proxy
    elif name == "MetaObjectiveLearner":
        from .meta_objective import MetaObjectiveLearner
        return MetaObjectiveLearner
    elif name == "RecursiveSelfModifier":
        from .recursive_self_modification import RecursiveSelfModifier
        return RecursiveSelfModifier
    elif name == "RGLearner":
        from .renormalization_learning import RGLearner
        return RGLearner
    elif name == "coarse_grain":
        from .renormalization_learning import coarse_grain
        return coarse_grain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SelfConsistencyLearner",
    "MutualInformationLearner",
    "gaussian_mi_proxy",
    "MetaObjectiveLearner",
    "RecursiveSelfModifier",
    "RGLearner",
    "coarse_grain",
]
