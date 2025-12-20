"""Pure NumPy quantum circuit simulation and variational kernels.

This module provides:
- Circuit: Minimal statevector simulator (gates: H, RX, RY, RZ, CNOT)
- AmplitudeEncoder: Amplitude encoding of classical vectors
- HardwareEfficientAnsatz: Variational circuit ansatz
- VariationalLawEncoder: Data reuploading encoder for quantum kernels
- QuantumKernel: Fidelity-based quantum kernel
- ConjugationEquivariantFeatureMap: Gauge-invariant feature map
- GraphConditionedEncoder: Encoder with graph-defined entanglement (NEW)
- GraphConditionedKernel: Kernel using graph topology (NEW)
"""

from .statevector_sim import (
    Circuit,
    I2, X, Y, Z, H,
    RX, RY, RZ,
    kron_n,
    apply_single,
    apply_cnot,
    measure_probs,
    state_inner,
    fidelity,
)
from .amplitude_encoding import AmplitudeEncoder, pad_to_pow2
from .variational_ansatz import HardwareEfficientAnsatz
from .quantum_law_encoder import VariationalLawEncoder
from .kernel_expressivity import (
    QuantumKernel,
    center_kernel,
    kernel_alignment,
    random_fourier_features_kernel,
)
from .gauge_equivariant_layers import ConjugationEquivariantFeatureMap
from .graph_conditioned_encoder import (
    GraphConditionedAnsatz,
    GraphConditionedEncoder,
    GraphConditionedKernel,
    weight_to_angle,
)

__all__ = [
    # Simulator
    "Circuit",
    "I2", "X", "Y", "Z", "H",
    "RX", "RY", "RZ",
    "kron_n",
    "apply_single",
    "apply_cnot",
    "measure_probs",
    "state_inner",
    "fidelity",
    # Encoders
    "AmplitudeEncoder",
    "pad_to_pow2",
    "HardwareEfficientAnsatz",
    "VariationalLawEncoder",
    # Kernels
    "QuantumKernel",
    "center_kernel",
    "kernel_alignment",
    "random_fourier_features_kernel",
    # Gauge-aware
    "ConjugationEquivariantFeatureMap",
    # Graph-conditioned (NEW)
    "GraphConditionedAnsatz",
    "GraphConditionedEncoder",
    "GraphConditionedKernel",
    "weight_to_angle",
]
