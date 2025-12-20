from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class SelfModelingEvaluator:
    """Evaluate whether the system forms a compact predictive model of itself.

    We fit a linear probe that predicts intermediate hidden state from parameters
    and input statistics. If prediction is good, the system is compressible/self-modelable.

    Output:
    - probe R^2 on held-out batch
    """
    batch_size: int = 256
    device: str = "cpu"
    ridge: float = 1e-3

    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        model = model.to(self.device)
        model.eval()

        # Collect data: x -> h_final if available
        x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
        with torch.no_grad():
            out = model(x, steps=4) if "steps" in model.forward.__code__.co_varnames else model(x)
        if isinstance(out, tuple):
            y, stats = out
            h = stats.get("h_final", y)
        else:
            y = out
            h = y

        # Build features: input mean/var + parameter norm (global)
        with torch.no_grad():
            pvec = torch.cat([p.reshape(-1) for p in model.parameters() if p.requires_grad])
            pnorm = torch.norm(pvec).item()
        feat = torch.stack([x.mean(dim=1), x.var(dim=1), torch.full((x.shape[0],), pnorm, device=self.device)], dim=1)
        # Ridge regression closed form: W = (X^T X + λI)^-1 X^T H
        X = feat
        H = h
        XtX = X.T @ X + self.ridge * torch.eye(X.shape[1], device=self.device)
        W = torch.linalg.solve(XtX, X.T @ H)
        Hhat = X @ W
        # R^2
        ss_res = torch.sum((H - Hhat)**2).item()
        ss_tot = torch.sum((H - H.mean(dim=0, keepdim=True))**2).item() + 1e-12
        r2 = float(1.0 - ss_res/ss_tot)
        return {"self_model_r2": r2, "param_norm": float(pnorm)}
