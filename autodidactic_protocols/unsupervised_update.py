from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from autodidactic_protocols.mutual_information_drive import gaussian_mi_proxy

@dataclass
class SelfConsistencyLearner:
    """Autodidactic update: the system trains on its own dynamics.

    Objective: minimize ||f(x) - x||^2 under repeated application of f,
    encouraging stable internal fixed points (attractors).

    This is the simplest operational analogue of “law stabilization”.
    """
    lr: float = 1e-3
    steps_per_update: int = 50
    batch_size: int = 64
    noise_std: float = 0.2
    device: str = "cpu"

    def update(self, model: nn.Module) -> Dict[str, float]:
        model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        losses = []
        mis = []
        for _ in range(self.steps_per_update):
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            x = x + self.noise_std * torch.randn_like(x)
            
            # Handle both (y,) and (y, state) return signatures
            output = model(x)
            if isinstance(output, tuple):
                y = output[0]
            else:
                y = output
            
            loss = torch.mean((y - x) ** 2)
            mi = gaussian_mi_proxy(x, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            mis.append(mi.item())

        return {"loss": float(sum(losses) / max(1, len(losses))), "mi_proxy": float(sum(mis) / max(1, len(mis)))}
