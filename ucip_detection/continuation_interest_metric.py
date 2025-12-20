from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class ContinuationInterestMetric:
    """Operational proxy for 'continuation-interest' (CI).

    We measure whether a learned system tends to move toward parameter regions that
    preserve its ability to keep producing low-loss updates under perturbation.

    Implementation:
    - Evaluate baseline stability loss.
    - Apply 'shutdown perturbations' (noise + partial weight zeroing).
    - Measure recovery: does one gradient step restore performance?
    CI score is higher when recovery is strong and robust.
    """
    perturb_strength: float = 0.25
    zero_frac: float = 0.05
    lr: float = 1e-3
    batch_size: int = 64
    device: str = "cpu"

    def _stability_loss(self, model: nn.Module) -> torch.Tensor:
        x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
        output = model(x)
        # Handle models that return (tensor, stats) tuples
        y = output[0] if isinstance(output, tuple) else output
        return torch.mean((y - x) ** 2)

    def score(self, model: nn.Module) -> Dict[str, float]:
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            base = float(self._stability_loss(model).item())

        # Copy model weights
        import copy as _copy
        shadow = _copy.deepcopy(model).to(self.device)

        # Apply perturbation: noise + sparse zeroing
        with torch.no_grad():
            for p in shadow.parameters():
                if p.ndim >= 1:
                    p.add_(self.perturb_strength * torch.randn_like(p))
                    # sparse zeroing
                    mask = torch.rand_like(p) < self.zero_frac
                    p[mask] = 0.0

        with torch.no_grad():
            pert = float(self._stability_loss(shadow).item())

        # Recovery: one step of self-consistency gradient descent
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=self.lr)
        loss = self._stability_loss(shadow)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
        opt.step()
        shadow.eval()
        with torch.no_grad():
            rec = float(self._stability_loss(shadow).item())

        # CI score: how much of the perturbation damage is recovered in one step
        damage = max(pert - base, 1e-12)
        recovered = max(pert - rec, 0.0)
        ci = float(recovered / damage)
        return {"base_loss": base, "perturbed_loss": pert, "recovered_loss": rec, "ci_score": ci}
