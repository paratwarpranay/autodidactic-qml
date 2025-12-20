from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class MatrixToRBM:
    """Map a matrix M to an RBM-like energy model.

    This is a practical proxy for the 'energy landscape' analogy:
    - visible layer corresponds to observed degrees of freedom
    - hidden layer corresponds to latent structure induced by M

    We do NOT claim this equals the paper's formal mapping; it is an operational sandbox.
    """
    n_visible: int = 64
    n_hidden: int = 64
    seed: Optional[int] = 0

    def build(self, M: np.ndarray) -> Dict[str, torch.Tensor]:
        torch.manual_seed(int(self.seed or 0))
        w, V = np.linalg.eigh(M)
        # Use spectrum to set coupling scale
        scale = float(np.std(w) + 1e-6)
        W = torch.randn(self.n_visible, self.n_hidden) * (0.2 / scale)
        b_v = torch.zeros(self.n_visible)
        b_h = torch.zeros(self.n_hidden)
        return {"W": W, "b_v": b_v, "b_h": b_h}

    @staticmethod
    def energy(v: torch.Tensor, h: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        W, b_v, b_h = params["W"], params["b_v"], params["b_h"]
        return -(v @ b_v) - (h @ b_h) - torch.sum((v @ W) * h, dim=-1)

    @staticmethod
    def gibbs_step(v: torch.Tensor, params: Dict[str, torch.Tensor], k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        W, b_v, b_h = params["W"], params["b_v"], params["b_h"]
        for _ in range(k):
            p_h = torch.sigmoid(v @ W + b_h)
            h = torch.bernoulli(p_h)
            p_v = torch.sigmoid(h @ W.T + b_v)
            v = torch.bernoulli(p_v)
        return v, h

    @staticmethod
    def cd_loss(v_data: torch.Tensor, params: Dict[str, torch.Tensor], k: int = 1) -> torch.Tensor:
        v_model, h_model = MatrixToRBM.gibbs_step(v_data.clone(), params, k=k)
        # Contrastive divergence proxy: energy(data) - energy(model)
        # Use expected hidden states for data term
        W, b_v, b_h = params["W"], params["b_v"], params["b_h"]
        p_h_data = torch.sigmoid(v_data @ W + b_h)
        E_data = MatrixToRBM.energy(v_data, p_h_data, params)
        E_model = MatrixToRBM.energy(v_model, h_model, params)
        return torch.mean(E_data - E_model)
