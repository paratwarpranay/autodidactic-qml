from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple

def coarse_grain(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Simple RG-like coarse graining for 1D features."""
    if factor <= 1:
        return x
    n = x.shape[-1]
    m = (n // factor) * factor
    x = x[..., :m]
    x = x.reshape(*x.shape[:-1], m // factor, factor).mean(dim=-1)
    return x

@dataclass
class RGLearner:
    """RG-inspired learning: stabilize structure across scales.

    Objective: encourage f(x) and f(coarse(x)) to be consistent when both
    are compared on the same coarse scale.
    """
    lr: float = 1e-3
    steps: int = 80
    batch_size: int = 64
    factor: int = 2
    device: str = "cpu"

    def update(self, model: nn.Module) -> Dict[str, float]:
        model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        losses = []
        for _ in range(self.steps):
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            y = model(x)
            x_c = coarse_grain(x, self.factor)
            y_c = coarse_grain(y, self.factor)
            loss = torch.mean((y_c - x_c) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        return {"rg_loss": float(np.mean(losses))}
