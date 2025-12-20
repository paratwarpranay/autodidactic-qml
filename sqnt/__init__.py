"""SQNT: Superpositional Quantum Network Topologies.

This module implements a classical testbed inspired by the SQNT program for
studying topology-adaptive learning dynamics under spectral constraints.

**Theoretical Lineage:**

SQNT (Superpositional Quantum Network Topologies) was introduced in:

    C. Altman, J. Pykacz & R. Zapatrin, "Superpositional Quantum Network
    Topologies," International Journal of Theoretical Physics 43, 2029–2041
    (2004). https://arxiv.org/abs/q-bio/0311016

The original SQNT framework proposed:
    - Quantum superposition of distinct network topologies via Rota algebras
    - Spatialization procedure associating subspaces with graph structures
    - Training on superpositions where topology itself becomes trainable
    - Collapse to classical network via measurement/observation

The follow-up work extended this with gradient-based learning:

    C. Altman & R. Zapatrin, "Backpropagation in Adaptive Quantum Networks,"
    International Journal of Theoretical Physics 49, 2991–2997 (2010).
    https://arxiv.org/abs/0903.4416

**This Implementation:**

This module provides a *classical* instantiation of SQNT-inspired principles:
    1. Soft spectral mass invariant (finite budget constraint)
    2. Participation-based topology plasticity
    3. Bidirectional topology-matrix coupling
    4. Falsification protocol (KT-1)

The implementation does NOT use Rota algebras or true quantum superposition.
Instead, it captures the *spirit* of SQNT—that topology should be adaptive
and trainable alongside dynamics—in a classical, testable framework.

**Scientific Claim (Falsifiable):**

    "Adaptive topology under a conserved spectral budget yields perturbation
    responses that cannot be replicated by fixed or random topologies."

Components:
    - SpectralMassInvariant: Soft budget constraint
    - TopologyPlasticity: Participation-based edge updates
    - TopologyFeedback: Bidirectional coupling
    - SQNTSystem: Integrated evolution system
    - KT1Protocol: Falsification test

Example:
    from sqnt import SQNTSystem, SQNTConfig, SystemMode, KT1Protocol
    
    # Create system
    config = SQNTConfig(mode=SystemMode.SQNT)
    system = SQNTSystem(config)
    system.initialize(M0, A0)
    
    # Define dynamics
    def dynamics(M, A):
        return -0.1 * M  # Example: decay dynamics
    
    # Run evolution
    for t in range(100):
        state = system.step(dynamics)
    
    # Run falsification test
    protocol = KT1Protocol()
    result = protocol.run(M0, A0, dynamics)

References:
    [1] C. Altman, J. Pykacz & R. Zapatrin (2004). "Superpositional Quantum
        Network Topologies." Int. J. Theor. Phys. 43, 2029–2041.
        https://doi.org/10.1023/B:IJTP.0000049008.51567.ec
        
    [2] C. Altman & R. Zapatrin (2010). "Backpropagation in Adaptive Quantum
        Networks." Int. J. Theor. Phys. 49, 2991–2997.
        https://doi.org/10.1007/s10773-009-0103-1

Author: Christopher Altman
Date: 2025
"""

from .invariant import (
    SpectralMassInvariant,
    apply_invariant_correction,
)

from .topology_plasticity import (
    TopologyPlasticity,
    TopologyFeedback,
    random_rewire,
    apply_perturbation,
)

from .sqnt_update import (
    SystemMode,
    SQNTConfig,
    SQNTState,
    SQNTSystem,
    run_comparison_experiment,
)

from .falsification_protocol import (
    FalsificationResult,
    KT1Protocol,
    quick_falsification_test,
)

__all__ = [
    # Invariant
    'SpectralMassInvariant',
    'apply_invariant_correction',
    # Topology
    'TopologyPlasticity',
    'TopologyFeedback',
    'random_rewire',
    'apply_perturbation',
    # SQNT System
    'SystemMode',
    'SQNTConfig',
    'SQNTState',
    'SQNTSystem',
    'run_comparison_experiment',
    # Falsification
    'FalsificationResult',
    'KT1Protocol',
    'quick_falsification_test',
]

__version__ = '1.0.0'
