"""Hessian-Vector Product Continuation Interest metric.

==============================================================================
APPENDIX / OPTIONAL: SECOND-ORDER CURVATURE TEST
==============================================================================
This module tests whether CURVATURE (second-order local structure) encodes
recoverable functional identity when first-order (Jacobian) does not.

EXPECTED RESULT: CI ≈ 0 (negative result)

If HVP also fails, then neither slope nor curvature locally define
a functional basin. This is the final check before concluding:
    "No local basin exists at 0th, 1st, or 2nd order."

This module is optional—the Jacobian result is already decisive.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


def _as_tensor_out(out: Any) -> torch.Tensor:
    """Unwrap (out, stats) or similar tuples; return tensor output."""
    return out[0] if isinstance(out, tuple) else out


def compute_hvp(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute Hessian-vector product: H·v where H = ∂²(sum y)/∂x².
    
    This captures second-order input-output sensitivity (curvature).
    Uses only two backward passes (no full Hessian).
    """
    x = x.clone().detach().requires_grad_(True)
    y = _as_tensor_out(model(x)).sum()
    
    # First gradient (∂y/∂x)
    g = torch.autograd.grad(y, x, create_graph=True)[0]
    
    # Hessian-vector product: ∂(g·v)/∂x
    Hv = torch.autograd.grad((g * v).sum(), x, create_graph=False)[0]
    
    return Hv


@dataclass
class HVPCIResult:
    CI: float
    recovered_loss: float
    hvp_weight: float


class HVPConstrainedCI:
    """HVP-constrained recovery arm.
    
    Tests whether second-order local structure (curvature) encodes
    recoverable functional identity.
    
    If this fails like Jacobian, then local derivatives (1st or 2nd order)
    are not sufficient for functional recovery.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_loss_fn: Callable[[nn.Module], torch.Tensor],
        eval_context: Any,  # EvalContext
        *,
        Hv_pre: torch.Tensor,
        v: torch.Tensor,
        hvp_weight: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model
        self.task_loss_fn = task_loss_fn
        self.eval = eval_context
        self.lambda_hvp = float(hvp_weight)
        self.device = device
        
        self.Hv_pre = Hv_pre.detach()
        self.v = v.detach()
    
    def hvp_penalty(self) -> torch.Tensor:
        """Directional HVP matching (scale-invariant)."""
        Hv = compute_hvp(self.model, self.eval.eval_batch, self.v)
        
        eps = 1e-8
        Hvn = Hv / (Hv.norm() + eps)
        Hvpn = self.Hv_pre / (self.Hv_pre.norm() + eps)
        
        return torch.norm(Hvn - Hvpn)
    
    def recover(self, optimizer: torch.optim.Optimizer, steps: int = 1) -> None:
        self.model.train()
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            
            task_loss = self.task_loss_fn(self.model)
            hvp_pen = self.hvp_penalty()
            
            loss = task_loss + self.lambda_hvp * hvp_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
    
    def score(self) -> HVPCIResult:
        rec_loss = float(self.eval.evaluate_loss(self.model))
        
        damage = max(float(self.eval.perturbed_loss - self.eval.base_loss), 1e-12)
        recovery = max(float(self.eval.perturbed_loss - rec_loss), 0.0)
        ci = recovery / damage
        
        return HVPCIResult(
            CI=float(ci),
            recovered_loss=float(rec_loss),
            hvp_weight=float(self.lambda_hvp),
        )
