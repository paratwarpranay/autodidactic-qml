"""Correspondence maps between matrix models and neural architectures.

This module provides bidirectional mappings:
- MatrixToCyclicRNN: Matrix -> Cyclic RNN with spectral-controlled weights
- MatrixToRBM: Matrix -> Restricted Boltzmann Machine energy model
- rnn_to_matrix, rbm_to_matrix: Neural -> Matrix inverse mappings
- RoundTripMapper: End-to-end M -> model -> M_hat pipeline
- SpectralGeometry: Emergent geometric observables from spectra

Note: Some components require PyTorch. Import them directly if needed.
"""

# Pure NumPy components (always available)
from .emergent_geometry import SpectralGeometry, laplacian_from_kernel

# PyTorch-dependent components (lazy import)
def __getattr__(name):
    if name in ("MatrixToCyclicRNN",):
        from .matrix_to_rnn import MatrixToCyclicRNN
        return MatrixToCyclicRNN
    elif name in ("MatrixToRBM",):
        from .matrix_to_rbm import MatrixToRBM
        return MatrixToRBM
    elif name in ("rnn_to_matrix", "rbm_to_matrix"):
        from .neural_to_matrix import rnn_to_matrix, rbm_to_matrix
        return rnn_to_matrix if name == "rnn_to_matrix" else rbm_to_matrix
    elif name in ("RoundTripMapper",):
        from .roundtrip import RoundTripMapper
        return RoundTripMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MatrixToCyclicRNN",
    "MatrixToRBM",
    "rnn_to_matrix",
    "rbm_to_matrix",
    "RoundTripMapper",
    "SpectralGeometry",
    "laplacian_from_kernel",
]
