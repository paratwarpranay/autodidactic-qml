from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional

from .mutual_information_drive import gaussian_mi_proxy

@dataclass
class MetaObjectiveLearner:
    """Objective self-construction via evolving constraint multipliers.

    We treat the learning problem as:
      minimize  recon + λ_stab * stab + λ_comp * comp  - λ_mi * MI

    and update the λ's (nonnegative) online based on constraint violation
    and performance.

    This is *not* a full second-order meta-learning method; it's a principled,
    testable first step: a Lagrangian-style controller that evolves its objective.
    """
    lr_model: float = 1e-3
    lr_lambda: float = 5e-2
    steps_per_update: int = 80
    batch_size: int = 128
    noise_std: float = 0.15

    # constraint setpoints
    recon_max: float = 0.25
    stab_max: float = 1.10      # target max spectral radius of W_hh
    comp_max: float = 5.0       # target parameter norm (scaled)
    mi_min: float = 0.15        # MI proxy minimum

    # evolving multipliers (initialized small)
    lam_stab: float = 0.10
    lam_comp: float = 0.05
    lam_mi: float = 0.10

    device: str = "cpu"

    def _recurrent_weight(self, model: nn.Module) -> Optional[torch.Tensor]:
        core = getattr(model, "core", None)
        if core is None:
            return None
        if hasattr(core, "W_hh") and hasattr(core.W_hh, "weight"):
            return core.W_hh.weight
        if hasattr(core, "weight_hh_l0"):
            return core.weight_hh_l0
        return None

    def _spectral_radius_proxy(self, W: torch.Tensor) -> torch.Tensor:
        # use largest singular value as robust proxy
        s = torch.linalg.svdvals(W)
        return torch.max(s)

    def update(self, model: nn.Module) -> Dict[str, float]:
        model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr_model)

        recon_hist, mi_hist, stab_hist, comp_hist, loss_hist = [], [], [], [], []
        for _ in range(self.steps_per_update):
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            x = x + self.noise_std * torch.randn_like(x)

            y = model(x)
            recon = torch.mean((y - x) ** 2)

            # representation for MI proxy
            h = None
            if hasattr(model, "core"):
                # try to expose hidden via calling core directly (best effort)
                try:
                    _, stats = model.core(model.fc1(x), steps=4)
                    h = stats.get("h_final", None)
                except Exception:
                    h = None
            if h is None:
                h = y
            mi = gaussian_mi_proxy(x, h)

            # stability: penalize spectral radius beyond stab_max
            W = self._recurrent_weight(model)
            if W is None:
                stab = torch.tensor(0.0, device=self.device)
                sr = torch.tensor(0.0, device=self.device)
            else:
                sr = self._spectral_radius_proxy(W)
                stab = torch.relu(sr - self.stab_max) ** 2

            # compression: scaled L2 norm of parameters
            comp = torch.tensor(0.0, device=self.device)
            n_params = 0
            for p in model.parameters():
                comp = comp + torch.sum(p * p)
                n_params += p.numel()
            comp = comp / max(1, n_params) * 1e3  # scale into human range

            # evolving objective weights
            lam_stab = torch.tensor(self.lam_stab, device=self.device)
            lam_comp = torch.tensor(self.lam_comp, device=self.device)
            lam_mi = torch.tensor(self.lam_mi, device=self.device)

            loss = recon + lam_stab * stab + lam_comp * torch.relu(comp - self.comp_max) - lam_mi * mi

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            recon_hist.append(float(recon.detach().cpu().item()))
            mi_hist.append(float(mi.detach().cpu().item()))
            stab_hist.append(float(sr.detach().cpu().item()))
            comp_hist.append(float(comp.detach().cpu().item()))
            loss_hist.append(float(loss.detach().cpu().item()))

        # Controller updates for lambdas (projected to >=0)
        recon_mean = float(np.mean(recon_hist))
        mi_mean = float(np.mean(mi_hist))
        sr_mean = float(np.mean(stab_hist))
        comp_mean = float(np.mean(comp_hist))

        # if stability proxy too high, increase lam_stab; else decay slightly
        self.lam_stab = max(0.0, self.lam_stab + self.lr_lambda * (sr_mean - self.stab_max))
        # if comp above setpoint, increase lam_comp
        self.lam_comp = max(0.0, self.lam_comp + self.lr_lambda * (comp_mean - self.comp_max))
        # if MI below minimum, increase lam_mi; else decay slightly
        self.lam_mi = max(0.0, self.lam_mi + self.lr_lambda * (self.mi_min - mi_mean))

        return {
            "meta_loss": float(np.mean(loss_hist)),
            "recon": recon_mean,
            "mi_proxy": mi_mean,
            "W_hh_spectral_radius": sr_mean,
            "comp_scaled": comp_mean,
            "lam_stab": float(self.lam_stab),
            "lam_comp": float(self.lam_comp),
            "lam_mi": float(self.lam_mi),
        }
