from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class TimeCoupledMemory:
    """A simple time-coupled memory buffer.

    Stores a sequence of vectors and allows time-symmetric smoothing,
    which is useful for Loop-style recurrent reasoning experiments.
    """
    maxlen: int = 128
    device: str = "cpu"

    def __post_init__(self):
        self._buf: List[torch.Tensor] = []

    def push(self, x: torch.Tensor) -> None:
        x = x.detach().to(self.device)
        self._buf.append(x)
        if len(self._buf) > self.maxlen:
            self._buf.pop(0)

    def as_tensor(self) -> torch.Tensor:
        if not self._buf:
            return torch.empty((0,), device=self.device)
        return torch.stack(self._buf, dim=0)

    def smooth(self, sigma: float = 2.0) -> torch.Tensor:
        """Time-symmetric Gaussian smoothing across stored timesteps."""
        T = self.as_tensor()
        if T.ndim == 0 or T.shape[0] == 0:
            return T
        n = T.shape[0]
        idx = torch.arange(n, device=self.device).float()
        W = torch.exp(-0.5 * (idx[:,None]-idx[None,:])**2 / (sigma**2 + 1e-12))
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
        return W @ T
