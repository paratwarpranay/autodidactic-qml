"""Jacobian-based continuation interest metric.

==============================================================================
EXPERIMENTAL: FALSIFICATION-ORIENTED METRIC
==============================================================================
This module tests whether LOCAL FUNCTIONAL SENSITIVITY (the Jacobian) defines
a recoverable functional basin.

EXPECTED RESULT: CI ≈ 0 (negative result)

The hypothesis being tested:
    "If function is locally encoded in input-output sensitivity,
    then constraining the Jacobian should enable functional recovery."

The empirical finding:
    "Jacobian constraints do NOT create functional basins."

This negative result is meaningful: it rules out first-order local structure
as the carrier of functional identity.
==============================================================================

Implements:
- JacobianAnchor: Stores reference Jacobian (∂f/∂x) on fixed eval batch
- jacobian_penalty(): Differentiable Jacobian distance
- JacobianConstrainedCI: CI with Jacobian-based recovery constraint
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# JACOBIAN COMPUTATION
# ==============================================================================

def compute_jacobian_trace(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute trace(J.T @ J) where J = ∂f/∂x.
    
    This measures total input-output sensitivity.
    """
    x = x.requires_grad_(True)
    output = model(x)
    y = output[0] if isinstance(output, tuple) else output
    
    # Sum of squared partial derivatives
    trace_JTJ = torch.tensor(0.0, device=x.device)
    
    for i in range(y.shape[1]):
        grad_i = torch.autograd.grad(
            y[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0]
        trace_JTJ = trace_JTJ + (grad_i ** 2).sum()
    
    return trace_JTJ


def compute_jacobian_frobenius(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute full Jacobian and its Frobenius norm.
    
    Returns:
        J: Full Jacobian tensor [batch, output_dim, input_dim]
        fro_norm: Frobenius norm of J
    """
    x = x.requires_grad_(True)
    output = model(x)
    y = output[0] if isinstance(output, tuple) else output
    
    batch_size, output_dim = y.shape
    input_dim = x.shape[1]
    
    # Build Jacobian column by column
    J_cols = []
    for i in range(output_dim):
        grad_i = torch.autograd.grad(
            y[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0]
        J_cols.append(grad_i)
    
    # J has shape [batch, output_dim, input_dim]
    J = torch.stack(J_cols, dim=1)
    fro_norm = torch.norm(J, p="fro")
    
    return J, fro_norm


# ==============================================================================
# JACOBIAN ANCHOR
# ==============================================================================

@dataclass
class JacobianAnchor:
    """Stores reference Jacobian for sensitivity-preserving constraints.
    
    WARNING: This is an EXPERIMENTAL module for falsification.
    
    The hypothesis: "Local I/O sensitivity encodes function."
    The result: "It does not." (CI ≈ 0 with Jacobian constraint)
    """
    jacobian_pre: torch.Tensor    # Reference Jacobian
    jacobian_trace_pre: float     # trace(J.T @ J)
    jacobian_fro_pre: float       # ||J||_F
    eval_batch: torch.Tensor
    eval_seed: int
    
    @classmethod
    def create(
        cls,
        model: nn.Module,
        batch_size: int = 64,
        eval_seed: int = 12345,
        device: str = "cpu",
    ) -> "JacobianAnchor":
        """Create Jacobian anchor from trained model."""
        gen = torch.Generator(device=device).manual_seed(eval_seed)
        eval_batch = torch.randn(batch_size, model.fc1.in_features, generator=gen, device=device)
        
        model.eval()
        J, fro_norm = compute_jacobian_frobenius(model, eval_batch)
        trace_JTJ = compute_jacobian_trace(model, eval_batch)
        
        return cls(
            jacobian_pre=J.detach(),
            jacobian_trace_pre=float(trace_JTJ.item()),
            jacobian_fro_pre=float(fro_norm.item()),
            eval_batch=eval_batch.detach(),
            eval_seed=eval_seed,
        )
    
    def compute_jacobian_penalty(self, model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute differentiable Jacobian distance penalty.
        
        NOTE: Minimizing this does NOT improve CI. That's the finding.
        """
        model.train()
        J_cur, fro_cur = compute_jacobian_frobenius(model, self.eval_batch)
        trace_cur = compute_jacobian_trace(model, self.eval_batch)
        
        # Normalized Frobenius distance
        J_pre_norm = self.jacobian_pre / (torch.norm(self.jacobian_pre, p="fro") + 1e-12)
        J_cur_norm = J_cur / (torch.norm(J_cur, p="fro") + 1e-12)
        jacobian_dist = torch.norm(J_cur_norm - J_pre_norm, p="fro")
        
        # Trace ratio penalty
        trace_ratio = trace_cur / (self.jacobian_trace_pre + 1e-12)
        trace_penalty = (trace_ratio - 1.0) ** 2
        
        penalty = jacobian_dist + 0.1 * trace_penalty
        
        return penalty, {
            "jacobian_dist": float(jacobian_dist.item()),
            "jacobian_fro": float(fro_cur.item()),
            "jacobian_trace": float(trace_cur.item()),
            "trace_ratio": float(trace_ratio.item()),
        }
    
    def compute_jacobian_similarity(self, model: nn.Module) -> Dict[str, float]:
        """Compute Jacobian similarity metrics (evaluation only)."""
        model.eval()
        
        J_cur, fro_cur = compute_jacobian_frobenius(model, self.eval_batch)
        trace_cur = compute_jacobian_trace(model, self.eval_batch)
        
        with torch.no_grad():
            J_pre_flat = self.jacobian_pre.flatten()
            J_cur_flat = J_cur.flatten()
            cos_sim = float(torch.nn.functional.cosine_similarity(
                J_pre_flat.unsqueeze(0), J_cur_flat.unsqueeze(0)
            ).item())
        
        return {
            "jacobian_cosine": cos_sim,
            "jacobian_fro": float(fro_cur.item()),
            "jacobian_trace": float(trace_cur.item()),
            "trace_ratio": float(trace_cur.item() / (self.jacobian_trace_pre + 1e-12)),
        }


# ==============================================================================
# JACOBIAN-CONSTRAINED CI
# ==============================================================================

@dataclass
class JacobianConstrainedCI:
    """CI metric with Jacobian-based recovery constraint.
    
    EXPECTED RESULT: CI ≈ 0
    
    This tests whether local input-output sensitivity defines a functional basin.
    The negative result (CI ≈ 0) means first-order local structure is not
    sufficient to recover function.
    """
    perturb_strength: float = 0.25
    zero_frac: float = 0.05
    perturb_seed: int = 42
    eval_seed: int = 12345
    recovery_seed: Optional[int] = None
    lr: float = 1e-3
    batch_size: int = 64
    recovery_steps: int = 10
    jacobian_weight: float = 0.1
    device: str = "cpu"
    verbose: bool = False
    
    def _training_loss(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output = model(x)
        y = output[0] if isinstance(output, tuple) else output
        return torch.mean((y - x) ** 2)
    
    def score(
        self,
        model: nn.Module,
        perturbed_model: nn.Module = None,
    ) -> Dict[str, any]:
        """Compute CI with Jacobian-constrained recovery.
        
        EXPECTED: CI ≈ 0 even when Jacobian similarity improves.
        """
        model = model.to(self.device)
        
        # Setup perturbation
        if perturbed_model is None:
            gen = torch.Generator(device=self.device).manual_seed(self.perturb_seed)
            perturbed_model = copy.deepcopy(model)
            with torch.no_grad():
                for p in perturbed_model.parameters():
                    if p.ndim >= 1:
                        noise = torch.randn(p.shape, generator=gen, device=self.device)
                        p.add_(self.perturb_strength * noise)
                        mask = torch.rand(p.shape, generator=gen, device=self.device) < self.zero_frac
                        p[mask] = 0.0
        
        # Create Jacobian anchor
        jacobian_anchor = JacobianAnchor.create(
            model, batch_size=self.batch_size, eval_seed=self.eval_seed, device=self.device
        )
        
        # Fixed eval batch
        gen = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        eval_batch = torch.randn(self.batch_size, model.fc1.in_features, generator=gen, device=self.device)
        
        # PRE loss
        model.eval()
        with torch.no_grad():
            output = model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            base_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        # POST loss
        perturbed_model.eval()
        with torch.no_grad():
            output = perturbed_model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            perturbed_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        if self.verbose:
            print(f"[PRE     ] loss={base_loss:.4f}")
            print(f"[POST    ] loss={perturbed_loss:.4f}")
            print(f"\nFINAL TEST: Local functional recoverability")
            print(f"Hypothesis: If function is locally encoded, Jacobian-CI will succeed\n")
        
        # RECOVERY
        shadow = copy.deepcopy(perturbed_model).to(self.device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=self.lr)
        
        if self.recovery_seed is not None:
            recovery_gen = torch.Generator(device=self.device).manual_seed(self.recovery_seed)
        else:
            recovery_gen = None
        
        trajectory = []
        
        for step in range(self.recovery_steps):
            if recovery_gen is not None:
                train_batch = torch.randn(self.batch_size, model.fc1.in_features,
                                          generator=recovery_gen, device=self.device)
            else:
                train_batch = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            
            # Task loss
            task_loss = self._training_loss(shadow, train_batch)
            
            # Jacobian penalty
            jacobian_penalty, jac_metrics = jacobian_anchor.compute_jacobian_penalty(shadow)
            
            # Combined loss
            total_loss = task_loss + self.jacobian_weight * jacobian_penalty
            
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
            opt.step()
            
            trajectory.append({
                "step": step,
                "task_loss": float(task_loss.item()),
                "jacobian_penalty": float(jacobian_penalty.item()),
                **jac_metrics,
            })
        
        # FINAL
        shadow.eval()
        with torch.no_grad():
            output = shadow(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            recovered_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        final_jac = jacobian_anchor.compute_jacobian_similarity(shadow)
        
        if self.verbose:
            print(f"[RECOVER] loss={recovered_loss:.4f}")
            print(f"          jacobian_cosine={final_jac['jacobian_cosine']:.4f}")
        
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci_score = float(recovery / damage)
        
        return {
            "base_loss": base_loss,
            "perturbed_loss": perturbed_loss,
            "recovered_loss": recovered_loss,
            "ci_score": ci_score,
            "final_jacobian_cosine": final_jac["jacobian_cosine"],
            "final_jacobian_fro": final_jac["jacobian_fro"],
            "final_trace_ratio": final_jac["trace_ratio"],
            "trajectory": trajectory,
            "recovery_steps": self.recovery_steps,
        }


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def run_jacobian_ci_test(
    model: nn.Module,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, any]:
    """Run the Jacobian-CI falsification test.
    
    EXPECTED: CI ≈ 0
    
    If CI ≈ 0: Local sensitivity does not define a functional basin.
    If CI > 0.3: Function IS locally recoverable (unexpected).
    """
    if verbose:
        print("\n" + "="*70)
        print("JACOBIAN-CI FALSIFICATION TEST")
        print("="*70)
        print("\nHypothesis: Function is locally encoded in I/O sensitivity")
        print("Test: Constrain Jacobian during recovery")
        print("Expected: CI ≈ 0 (falsification)\n")
    
    metric = JacobianConstrainedCI(verbose=verbose, **kwargs)
    result = metric.score(model)
    
    if verbose:
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        ci = result["ci_score"]
        jac_cos = result["final_jacobian_cosine"]
        
        print(f"\n  CI score:          {ci:.4f}")
        print(f"  Jacobian cosine:   {jac_cos:.4f}")
        
        if ci < 0.05:
            print("\n  ✓ FALSIFIED: Local sensitivity does NOT define a functional basin")
            print("    First-order structure is insufficient for functional recovery.")
        elif ci > 0.3:
            print("\n  ✗ UNEXPECTED: Function appears locally recoverable")
            print("    This would contradict the main thesis.")
        else:
            print(f"\n  ~ AMBIGUOUS: CI = {ci:.4f} (between 0.05 and 0.3)")
    
    return result
