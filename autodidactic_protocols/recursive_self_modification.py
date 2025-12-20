from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple

@dataclass
class RecursiveSelfModifier:
    """Loop-style recursion: the system updates its own update rule.

    Mechanism:
    - Maintain a small 'meta-optimizer' network that produces per-parameter learning-rate multipliers.
    - The base model performs an inner update step; the meta network is trained to reduce a stability loss.

    This is a compact, runnable proxy for 'systems learning how to learn' under recursion.
    """
    meta_hidden: int = 64
    inner_lr: float = 1e-3
    meta_lr: float = 1e-3
    inner_steps: int = 10
    outer_steps: int = 60
    batch_size: int = 64
    device: str = "cpu"

    def _build_meta(self, n_params: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(3, self.meta_hidden),
            nn.Tanh(),
            nn.Linear(self.meta_hidden, 1),
            nn.Sigmoid(),  # output in (0,1)
        ).to(self.device)

    def update(self, model: nn.Module) -> Dict[str, float]:
        model.to(self.device)
        model.train()

        # Flatten params
        params = [p for p in model.parameters() if p.requires_grad]
        n = sum(p.numel() for p in params)
        meta = self._build_meta(n)
        meta_opt = torch.optim.Adam(meta.parameters(), lr=self.meta_lr)

        # We create a simple feature vector per parameter element:
        # [value, grad, running_grad_var] -> lr_multiplier
        running_var = torch.zeros(n, device=self.device)

        def flatten(tensors):
            return torch.cat([t.reshape(-1) for t in tensors], dim=0)

        def unflatten(vec, like_tensors):
            out = []
            idx = 0
            for t in like_tensors:
                k = t.numel()
                out.append(vec[idx:idx+k].reshape_as(t))
                idx += k
            return out

        stability_losses = []
        for _ in range(self.outer_steps):
            # inner loop: do self-consistency update with meta-scaled LR
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            target = x.detach()

            for _ in range(self.inner_steps):
                y = model(x)
                loss = torch.mean((y - target) ** 2)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                gflat = flatten(grads)
                pflat = flatten(params)

                running_var = 0.95 * running_var + 0.05 * (gflat.detach() ** 2)
                feat = torch.stack([pflat, gflat, torch.sqrt(running_var + 1e-8)], dim=1)
                lr_mult = meta(feat).reshape(-1)  # (n,)
                step = -self.inner_lr * lr_mult * gflat

                new_params = unflatten(pflat + step, params)
                with torch.no_grad():
                    for p, npv in zip(params, new_params):
                        p.copy_(npv)

            # outer objective: after inner updates, measure stability on fresh sample
            x2 = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            y2 = model(x2)
            stability = torch.mean((y2 - x2) ** 2)

            meta_opt.zero_grad()
            stability.backward()
            torch.nn.utils.clip_grad_norm_(meta.parameters(), 5.0)
            meta_opt.step()
            stability_losses.append(float(stability.item()))

        return {"recursive_stability": float(np.mean(stability_losses))}
