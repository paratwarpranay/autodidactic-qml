from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

def rnn_to_matrix(model: nn.Module) -> np.ndarray:
    """Extract a symmetric 'effective law matrix' from a cyclic RNN.

    We interpret the recurrent weights as a generator of dynamics; symmetrize
    to obtain a Hermitian proxy that can be compared with matrix-model spectra.
    """
    if not hasattr(model, "W_rec"):
        raise ValueError("Model does not have W_rec parameter.")
    W = model.W_rec.detach().cpu().numpy()
    M = (W + W.T) / 2.0
    # normalize
    w = np.linalg.eigvalsh(M)
    s = float(np.std(w) + 1e-12)
    return M / s

def rbm_to_matrix(params: Dict[str, torch.Tensor]) -> np.ndarray:
    """Map RBM weight matrix to a symmetric proxy."""
    W = params["W"].detach().cpu().numpy()
    M = W @ W.T
    M = (M + M.T) / 2.0
    w = np.linalg.eigvalsh(M)
    s = float(np.std(w) + 1e-12)
    return M / s
