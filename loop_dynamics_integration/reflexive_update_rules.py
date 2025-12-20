from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Callable

@dataclass
class ReflexiveUpdater:
    """Reflexive update: update parameters using a function of their own state.

    Mechanism:
    - compute gradients of a stability objective
    - scale gradient by a function of parameter magnitude and running gradient stats
    """
    lr: float = 1e-3
    beta: float = 0.95
    device: str = "cpu"

    def __post_init__(self):
        self.running_var: Dict[int, torch.Tensor] = {}

    def step(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, float]:
        model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, grads)):
                if i not in self.running_var:
                    self.running_var[i] = torch.zeros_like(p)
                rv = self.running_var[i]
                rv.mul_(self.beta).add_((1-self.beta) * (g*g))
                # Reflexive scaling: larger params update a bit less, noisy grads update less
                scale = 1.0 / (1.0 + p.abs() + torch.sqrt(rv + 1e-8))
                p.add_(-self.lr * scale * g)

        return {"reflexive_update_norm": float(sum(g.norm().item() for g in grads))}
