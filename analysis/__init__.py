"""Analysis tools for spectral diagnostics, phase transitions, and invariants.

This module provides:
- SpectralDiagnostics: Eigenvalue statistics for symmetric matrices
- PhaseTransitionMetrics: Jump detection and kurtosis heuristics
- InvariantComparator: Compare gauge-invariant properties of matrices
- LawStabilityTracker: Track learning dynamics stability
- LogisticModel, fit_logistic_regression: Phase boundary fitting
- SpectralComplexityAnalyzer: Information-theoretic complexity measures
- EntanglementGraphAnalyzer: Graph-theoretic structure from matrices
- PhaseTransitionAnalyzer: Comprehensive phase detection
- PhysicsHygiene: State validation and invariant assertions
- AdaptiveThreshold: Prevent graph degeneracy with adaptive thresholds
- TimeReversalProbe: Bidirectional evolution tests for irreversibility
"""

from .metrics import running_stats
from .spectral_diagnostics import SpectralDiagnostics
from .phase_transition_metrics import PhaseTransitionMetrics
from .invariant_metrics import (
    InvariantComparator,
    trace_powers,
    spectrum,
    spectral_distances,
    matrix_norms,
    relative_error,
    fro_rel,
)
from .law_stability_tracker import LawStabilityTracker
from .logistic_boundary import (
    LogisticModel,
    fit_logistic_regression,
    sigmoid,
    standardize,
    boundary_gstar_from_scores,
    bootstrap_logistic_boundary,
)
from .spectral_complexity import (
    SpectralComplexityAnalyzer,
    spectral_entropy,
    level_spacing_ratio,
    participation_ratio,
    spectral_form_factor,
    complexity_distance,
)
from .entanglement_graph import (
    EntanglementGraphAnalyzer,
    adjacency_from_matrix,
    graph_laplacian,
    laplacian_spectrum,
    algebraic_connectivity,
    spectral_gap,
    effective_resistance,
    von_neumann_entropy,
    graph_distance,
)
from .autodidactic_phase_detector import (
    PhaseTransitionAnalyzer,
    OnlinePhaseDetector,
    detect_learning_phases,
    susceptibility,
    binder_cumulant,
)
from .spectral_observables import (
    Observable,
    ObservableHistory,
    SpectralObservables,
    compute_thermodynamic_observables,
)
from .physics_hygiene import (
    PhysicsHygiene,
    HygieneConfig,
    HygieneReport,
    PhysicsViolation,
    assert_physics_invariants,
    atomic_step,
    AtomicUpdateResult,
    create_atomic_updater,
)
from .adaptive_threshold import (
    compute_adaptive_threshold,
    adaptive_adjacency,
    smooth_threshold_adjacency,
)
from .time_reversal_probe import (
    TimeReversalProbe,
    TimeReversalResult,
    ReversibilityClass,
    matrix_fidelity,
    spectral_fidelity,
    topology_fidelity,
    quick_reversibility_check,
    create_reversible_dynamics_pair,
)

__all__ = [
    # Basic metrics
    "running_stats",
    "SpectralDiagnostics",
    "PhaseTransitionMetrics",
    # Invariants
    "InvariantComparator",
    "trace_powers",
    "spectrum",
    "spectral_distances",
    "matrix_norms",
    "relative_error",
    "fro_rel",
    # Stability
    "LawStabilityTracker",
    # Phase boundaries
    "LogisticModel",
    "fit_logistic_regression",
    "sigmoid",
    "standardize",
    "boundary_gstar_from_scores",
    "bootstrap_logistic_boundary",
    # Spectral complexity (NEW)
    "SpectralComplexityAnalyzer",
    "spectral_entropy",
    "level_spacing_ratio",
    "participation_ratio",
    "spectral_form_factor",
    "complexity_distance",
    # Entanglement graph (NEW)
    "EntanglementGraphAnalyzer",
    "adjacency_from_matrix",
    "graph_laplacian",
    "laplacian_spectrum",
    "algebraic_connectivity",
    "spectral_gap",
    "effective_resistance",
    "von_neumann_entropy",
    "graph_distance",
    # Phase detection (NEW)
    "PhaseTransitionAnalyzer",
    "OnlinePhaseDetector",
    "detect_learning_phases",
    "susceptibility",
    "binder_cumulant",
    # Spectral observables
    "Observable",
    "ObservableHistory",
    "SpectralObservables",
    "compute_thermodynamic_observables",
    # Physics hygiene (NEW)
    "PhysicsHygiene",
    "HygieneConfig",
    "HygieneReport",
    "PhysicsViolation",
    "assert_physics_invariants",
    "atomic_step",
    "AtomicUpdateResult",
    "create_atomic_updater",
    # Adaptive thresholding (NEW)
    "compute_adaptive_threshold",
    "adaptive_adjacency",
    "smooth_threshold_adjacency",
    # Time-reversal probe (NEW)
    "TimeReversalProbe",
    "TimeReversalResult",
    "ReversibilityClass",
    "matrix_fidelity",
    "spectral_fidelity",
    "topology_fidelity",
    "quick_reversibility_check",
    "create_reversible_dynamics_pair",
]
