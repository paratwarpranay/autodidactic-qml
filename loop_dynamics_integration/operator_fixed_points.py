from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class FixedPointFinder:
    """Find approximate fixed points of a model map x -> f(x)."""
    steps: int = 50
    lr: float = 0.1
    device: str = "cpu"

    def find(self, model: nn.Module, dim: int, n: int = 8) -> Dict[str, float]:
        model = model.to(self.device)
        model.eval()
        x = torch.randn(n, dim, device=self.device, requires_grad=True)
        opt = torch.optim.SGD([x], lr=self.lr)
        losses = []
        for _ in range(self.steps):
            y = model(x)
            loss = torch.mean((y - x)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return {"fixed_point_loss": float(np.mean(losses[-10:])), "fixed_point_loss_start": float(losses[0])}
