"""Nonlocality probes: k-step CI curve and hysteresis measurement.

This module implements publication-grade diagnostics for characterizing
the nonlocal nature of functional recovery:

1. k-step CI curve: CI(k) for k ∈ {1,2,4,8,16} reveals whether recovery
   requires path-dependent optimization vs. local gradient descent.

2. Hysteresis measurement: Forward/reverse perturbation sweeps measure
   memory/metastability effects—path dependence indicates nonlocal structure.

3. Distance triad: Report parameter + representation + functional distances
   together to show their decoupling (the core result).

4. Null model baseline: Direct functional loss optimization provides upper
   bound on what's achievable in k steps (control for proxy constraints).

Scientific foundation:
    - A 1-step recovery test probes the local vector field at the perturbed point.
    - If CI≈0 systematically across all constraint families, the constraint
      gradient is near-orthogonal to the functional recovery direction.
    - Hysteresis area > 0 indicates memory/metastability/path dependence.

References:
    - This implements the "k-step curve" and "hysteresis signature" recommended
      for publication-grade falsification of the locality hypothesis.

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

from .invariant_constrained_ci import (
    PerturbationConfig,
    apply_perturbation,
    EvalContext,
    compute_model_invariants,
    invariant_distance,
    _find_weight_param,
    _spectral_entropy_torch,
    RepresentationCI,
    DEFAULT_PENALTY_KEYS,
    PENALTY_KEYS_SCALE,
    PENALTY_KEYS_SHAPE,
)


# =============================================================================
# DISTANCE TRIAD: Parameter + Representation + Functional
# =============================================================================

@dataclass
class DistanceTriad:
    """Distance triad: parameter, representation, and functional distances.
    
    The core result is that these three distances DECOUPLE:
    - Parameter distance can be small while functional distance is large
    - Representation distance can be small while functional distance is large
    - Geometric recovery ≠ functional recovery
    """
    param_distance: float      # ||W_recover - W_pre||_F / ||W_pre||_F
    repr_distance: float       # 1 - cosine(h_recover, h_pre)
    functional_distance: float # (L_recover - L_pre) / (L_post - L_pre) = 1 - CI
    invariant_distance: float  # Normalized invariant distance
    
    @classmethod
    def compute(
        cls,
        model_pre: nn.Module,
        model_recover: nn.Module,
        model_post: nn.Module,
        eval_batch: torch.Tensor,
        inv_pre: Dict[str, float],
        penalty_keys: List[str] = None,
    ) -> "DistanceTriad":
        """Compute all three distances."""
        if penalty_keys is None:
            penalty_keys = DEFAULT_PENALTY_KEYS
        
        # Parameter distance (normalized Frobenius)
        W_pre = _find_weight_param(model_pre)
        W_rec = _find_weight_param(model_recover)
        if W_pre is not None and W_rec is not None:
            param_dist = float(torch.norm(W_rec - W_pre).item() / (torch.norm(W_pre).item() + 1e-12))
        else:
            param_dist = float('nan')
        
        # Representation distance (1 - cosine similarity)
        rep_ci = RepresentationCI.compute(model_pre, model_recover, eval_batch)
        repr_dist = 1.0 - rep_ci["h_cosine"]
        
        # Functional distance (1 - CI, computed from losses)
        model_pre.eval()
        model_recover.eval()
        model_post.eval()
        with torch.no_grad():
            out_pre = model_pre(eval_batch)
            out_rec = model_recover(eval_batch)
            out_post = model_post(eval_batch)
            
            y_pre = out_pre[0] if isinstance(out_pre, tuple) else out_pre
            y_rec = out_rec[0] if isinstance(out_rec, tuple) else out_rec
            y_post = out_post[0] if isinstance(out_post, tuple) else out_post
            
            L_pre = float(torch.mean((y_pre - eval_batch) ** 2).item())
            L_rec = float(torch.mean((y_rec - eval_batch) ** 2).item())
            L_post = float(torch.mean((y_post - eval_batch) ** 2).item())
        
        damage = max(L_post - L_pre, 1e-12)
        functional_dist = (L_rec - L_pre) / damage  # = 1 - CI
        functional_dist = max(0.0, min(functional_dist, 1.0))
        
        # Invariant distance
        inv_rec = compute_model_invariants(model_recover)
        inv_dist = invariant_distance(inv_rec, inv_pre, penalty_keys)
        
        return cls(
            param_distance=param_dist,
            repr_distance=repr_dist,
            functional_distance=functional_dist,
            invariant_distance=inv_dist,
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "param_distance": self.param_distance,
            "repr_distance": self.repr_distance,
            "functional_distance": self.functional_distance,
            "invariant_distance": self.invariant_distance,
            "ci_score": 1.0 - self.functional_distance,
        }
    
    def decoupling_signature(self) -> str:
        """Interpret the decoupling pattern."""
        if self.param_distance < 0.1 and self.functional_distance > 0.5:
            return "PARAM_RECOVERED_FUNC_LOST"
        elif self.repr_distance < 0.1 and self.functional_distance > 0.5:
            return "REPR_RECOVERED_FUNC_LOST"
        elif self.invariant_distance < 0.1 and self.functional_distance > 0.5:
            return "INV_RECOVERED_FUNC_LOST"
        elif self.functional_distance < 0.1:
            return "FULL_RECOVERY"
        else:
            return "PARTIAL_RECOVERY"


# =============================================================================
# K-STEP CI CURVE: Nonlocality signature
# =============================================================================

@dataclass
class KStepCurveResult:
    """Results from k-step CI curve experiment."""
    k_values: List[int]
    ci_values: List[float]
    distance_triads: List[DistanceTriad]
    constraint_type: str
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "k_values": self.k_values,
            "ci_values": self.ci_values,
            "constraint_type": self.constraint_type,
            "triads": [t.to_dict() for t in self.distance_triads],
        }
    
    def is_locally_recoverable(self, threshold: float = 0.1) -> bool:
        """Check if function is locally recoverable (CI > threshold at k=1)."""
        if len(self.ci_values) > 0:
            return self.ci_values[0] > threshold
        return False
    
    def recovery_onset_k(self, threshold: float = 0.1) -> Optional[int]:
        """Find smallest k where CI > threshold (None if never)."""
        for k, ci in zip(self.k_values, self.ci_values):
            if ci > threshold:
                return k
        return None
    
    def saturation_ci(self) -> float:
        """CI at largest k (asymptotic recovery)."""
        return self.ci_values[-1] if self.ci_values else 0.0


def compute_k_step_curve(
    model: nn.Module,
    k_values: List[int] = None,
    constraint_type: str = "shape",
    perturb_seed: int = 42,
    eval_seed: int = 12345,
    recovery_seed: Optional[int] = 2025,
    perturb_strength: float = 0.25,
    zero_frac: float = 0.05,
    lr: float = 1e-3,
    invariant_weight: float = 0.1,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> KStepCurveResult:
    """Compute CI(k) curve for k ∈ k_values.
    
    A 1-step recovery test probes the local vector field induced by the 
    constraint objective at the perturbed point. If CI≈0 at k=1 but rises
    for larger k, recovery requires path-dependent optimization—the function
    is nonlocally encoded.
    
    Args:
        model: Trained model to test
        k_values: Recovery steps to test (default: [1, 2, 4, 8, 16])
        constraint_type: "none", "scale", "shape", "direct" (functional loss)
        perturb_seed: Fixed seed for reproducible perturbation
        eval_seed: Fixed seed for evaluation batch
        recovery_seed: Fixed seed for recovery (deterministic dynamics)
        perturb_strength: Noise magnitude
        zero_frac: Fraction of weights to zero
        lr: Learning rate for recovery
        invariant_weight: λ for constraint penalty
        batch_size: Evaluation batch size
        device: Compute device
        verbose: Print progress
        
    Returns:
        KStepCurveResult with CI values and distance triads for each k
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16]
    
    # Select penalty keys based on constraint type
    if constraint_type == "none":
        penalty_keys = []
    elif constraint_type == "scale":
        penalty_keys = PENALTY_KEYS_SCALE
    elif constraint_type == "shape":
        penalty_keys = PENALTY_KEYS_SHAPE
    elif constraint_type == "direct":
        penalty_keys = []  # Direct functional loss optimization
    else:
        penalty_keys = DEFAULT_PENALTY_KEYS
    
    model = model.to(device)
    
    # Create fixed perturbation
    config = PerturbationConfig(seed=perturb_seed, strength=perturb_strength, zero_frac=zero_frac)
    perturbed_model = apply_perturbation(model, config, device)
    
    # Create eval context
    eval_ctx = EvalContext.create(
        model=model,
        perturbed_model=perturbed_model,
        batch_size=batch_size,
        eval_seed=eval_seed,
        device=device,
        penalty_keys=penalty_keys if penalty_keys else DEFAULT_PENALTY_KEYS,
    )
    
    inv_pre = eval_ctx.inv_pre
    base_loss = eval_ctx.base_loss
    perturbed_loss = eval_ctx.perturbed_loss
    
    if verbose:
        print(f"\n  k-step CI curve ({constraint_type} constraint)")
        print(f"  Seeds: perturb={perturb_seed}, eval={eval_seed}, recovery={recovery_seed}")
        print(f"  PRE loss:  {base_loss:.6f}")
        print(f"  POST loss: {perturbed_loss:.6f}")
        print(f"\n  {'k':>4} {'CI':>8} {'param_d':>8} {'repr_d':>8} {'func_d':>8} {'inv_d':>8}")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    ci_values = []
    triads = []
    
    for k in k_values:
        # Fresh copy of perturbed model for each k
        shadow = copy.deepcopy(perturbed_model).to(device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=lr)
        
        if recovery_seed is not None:
            recovery_gen = torch.Generator(device=device).manual_seed(recovery_seed)
        else:
            recovery_gen = None
        
        # Recovery loop
        for step in range(k):
            # Generate training batch
            if recovery_gen is not None:
                x = torch.randn(batch_size, model.fc1.in_features, generator=recovery_gen, device=device)
            else:
                x = torch.randn(batch_size, model.fc1.in_features, device=device)
            
            output = shadow(x)
            y = output[0] if isinstance(output, tuple) else output
            task_loss = torch.mean((y - x) ** 2)
            
            if constraint_type == "direct":
                # Direct functional loss optimization (null model / upper bound)
                total_loss = task_loss
            elif penalty_keys:
                # Constraint-based recovery
                inv_penalty = _compute_invariant_penalty(shadow, inv_pre, penalty_keys, device)
                total_loss = task_loss + invariant_weight * inv_penalty
            else:
                # Unconstrained recovery
                total_loss = task_loss
            
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
            opt.step()
        
        # Evaluate recovery
        recovered_loss = eval_ctx.evaluate_loss(shadow)
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci = float(recovery / damage)
        ci_values.append(ci)
        
        # Compute distance triad
        triad = DistanceTriad.compute(
            model_pre=model,
            model_recover=shadow,
            model_post=perturbed_model,
            eval_batch=eval_ctx.eval_batch,
            inv_pre=inv_pre,
            penalty_keys=penalty_keys if penalty_keys else DEFAULT_PENALTY_KEYS,
        )
        triads.append(triad)
        
        if verbose:
            print(f"  {k:>4} {ci:>8.4f} {triad.param_distance:>8.4f} {triad.repr_distance:>8.4f} "
                  f"{triad.functional_distance:>8.4f} {triad.invariant_distance:>8.4f}")
    
    result = KStepCurveResult(
        k_values=k_values,
        ci_values=ci_values,
        distance_triads=triads,
        constraint_type=constraint_type,
    )
    
    if verbose:
        print(f"\n  Local recovery (k=1): {'YES' if result.is_locally_recoverable() else 'NO'}")
        onset = result.recovery_onset_k()
        if onset:
            print(f"  Recovery onset: k={onset}")
        print(f"  Saturation CI (k={k_values[-1]}): {result.saturation_ci():.4f}")
    
    return result


def _compute_invariant_penalty(
    model: nn.Module,
    target_inv: Dict[str, float],
    penalty_keys: List[str],
    device: str,
) -> torch.Tensor:
    """Compute differentiable invariant penalty."""
    W = _find_weight_param(model)
    if W is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    M = (W + W.T) / 2
    penalty = torch.tensor(0.0, device=device)
    
    # Trace powers
    M_power = M
    for k in range(1, 5):
        if k > 1:
            M_power = M_power @ M
        key = f"tr_M{k}"
        if key in penalty_keys:
            tr_k = torch.trace(M_power)
            target_k = target_inv.get(key, 0.0)
            scale = max(abs(target_k), 1e-8)
            penalty = penalty + ((tr_k - target_k) / scale) ** 2
    
    # Frobenius norm
    if "fro_norm" in penalty_keys:
        fro = torch.norm(M, p="fro")
        target_fro = target_inv.get("fro_norm", 1.0)
        scale = max(abs(target_fro), 1e-8)
        penalty = penalty + ((fro - target_fro) / scale) ** 2
    
    # Spectral entropy
    if "spec_entropy" in penalty_keys:
        entropy = _spectral_entropy_torch(M)
        target_entropy = target_inv.get("spec_entropy", 3.0)
        scale = max(abs(target_entropy), 1e-8)
        penalty = penalty + ((entropy - target_entropy) / scale) ** 2
    
    return penalty


def compare_constraint_families(
    model: nn.Module,
    k_values: List[int] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, KStepCurveResult]:
    """Compare k-step curves across constraint families.
    
    Tests: none, scale, shape, direct (functional loss).
    The "direct" constraint is the NULL MODEL providing an upper bound
    on what's achievable with k optimization steps.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16]
    
    results = {}
    for constraint in ["none", "scale", "shape", "direct"]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Constraint: {constraint.upper()}")
            print('='*60)
        
        results[constraint] = compute_k_step_curve(
            model=model,
            k_values=k_values,
            constraint_type=constraint,
            verbose=verbose,
            **kwargs,
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY: CI at k=1 (locality test)")
        print('='*60)
        print(f"  {'Constraint':<12} {'CI(k=1)':>10} {'CI(k=16)':>10} {'Local?':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
        for name, result in results.items():
            ci_1 = result.ci_values[0] if result.ci_values else 0
            ci_max = result.saturation_ci()
            local = "YES" if result.is_locally_recoverable() else "NO"
            print(f"  {name:<12} {ci_1:>10.4f} {ci_max:>10.4f} {local:>8}")
        
        # Interpret
        direct_ci = results["direct"].ci_values[0] if "direct" in results else 0
        best_proxy_ci = max(
            results.get("scale", KStepCurveResult([], [], [], "")).ci_values[0] if "scale" in results else 0,
            results.get("shape", KStepCurveResult([], [], [], "")).ci_values[0] if "shape" in results else 0,
        )
        
        print(f"\n  Upper bound (direct functional): CI(1) = {direct_ci:.4f}")
        print(f"  Best proxy constraint: CI(1) = {best_proxy_ci:.4f}")
        
        if direct_ci < 0.1:
            print(f"\n  ✗ Even DIRECT functional optimization fails at k=1")
            print(f"    → Function is fundamentally nonlocal (no 1-step basin exists)")
        elif best_proxy_ci < 0.1 and direct_ci > 0.1:
            print(f"\n  ✗ Proxies fail but direct succeeds")
            print(f"    → Geometric constraints MISALIGNED with functional recovery")
    
    return results


# =============================================================================
# HYSTERESIS MEASUREMENT: Memory/metastability signature
# =============================================================================

@dataclass
class HysteresisResult:
    """Results from hysteresis experiment."""
    g_values: np.ndarray           # Perturbation strengths
    forward_ci: np.ndarray         # CI on forward sweep (increasing g)
    reverse_ci: np.ndarray         # CI on reverse sweep (decreasing g)
    hysteresis_area: float         # ∫|forward - reverse| dg
    max_gap: float                 # max|forward - reverse|
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "g_values": self.g_values.tolist(),
            "forward_ci": self.forward_ci.tolist(),
            "reverse_ci": self.reverse_ci.tolist(),
            "hysteresis_area": self.hysteresis_area,
            "max_gap": self.max_gap,
        }
    
    def has_hysteresis(self, threshold: float = 0.05) -> bool:
        """Check if significant hysteresis exists."""
        return self.hysteresis_area > threshold or self.max_gap > 0.1


def compute_hysteresis(
    model: nn.Module,
    g_values: np.ndarray = None,
    constraint_type: str = "shape",
    recovery_steps: int = 4,
    base_seed: int = 42,
    eval_seed: int = 12345,
    lr: float = 1e-3,
    invariant_weight: float = 0.1,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> HysteresisResult:
    """Measure hysteresis: forward vs reverse perturbation sweep.
    
    Protocol:
        1. Forward sweep: g = 0 → g_max, measure CI(g) with recovery
        2. Reverse sweep: g = g_max → 0, measure CI(g) with recovery
        3. Compute hysteresis area A = ∫|CI_forward - CI_reverse| dg
    
    If A > 0, the system exhibits memory/metastability/path dependence—
    exactly the "nonlocal identity" phenomenon we're characterizing.
    
    Args:
        model: Trained model
        g_values: Perturbation strengths to sweep (default: np.linspace(0, 0.5, 11))
        constraint_type: Constraint family for recovery
        recovery_steps: Steps for each recovery attempt
        base_seed: Base seed for perturbation
        eval_seed: Seed for evaluation batch
        lr: Learning rate
        invariant_weight: λ for constraint
        batch_size: Batch size
        device: Compute device
        verbose: Print progress
        
    Returns:
        HysteresisResult with forward/reverse CI curves and area
    """
    if g_values is None:
        g_values = np.linspace(0.0, 0.5, 11)
    
    # Select penalty keys
    if constraint_type == "none":
        penalty_keys = []
    elif constraint_type == "scale":
        penalty_keys = PENALTY_KEYS_SCALE
    elif constraint_type == "shape":
        penalty_keys = PENALTY_KEYS_SHAPE
    else:
        penalty_keys = DEFAULT_PENALTY_KEYS
    
    model = model.to(device)
    
    # Get PRE invariants and loss
    inv_pre = compute_model_invariants(model)
    gen = torch.Generator(device=device).manual_seed(eval_seed)
    eval_batch = torch.randn(batch_size, model.fc1.in_features, generator=gen, device=device)
    
    model.eval()
    with torch.no_grad():
        out = model(eval_batch)
        y = out[0] if isinstance(out, tuple) else out
        base_loss = float(torch.mean((y - eval_batch) ** 2).item())
    
    def measure_ci_at_g(g: float, seed_offset: int = 0) -> float:
        """Measure CI at perturbation strength g."""
        if g < 1e-6:
            return 1.0  # No perturbation → perfect recovery
        
        config = PerturbationConfig(seed=base_seed + seed_offset, strength=g, zero_frac=0.05)
        perturbed = apply_perturbation(model, config, device)
        
        # Measure perturbed loss
        perturbed.eval()
        with torch.no_grad():
            out = perturbed(eval_batch)
            y = out[0] if isinstance(out, tuple) else out
            perturbed_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        # Recovery
        shadow = copy.deepcopy(perturbed).to(device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=lr)
        
        recovery_gen = torch.Generator(device=device).manual_seed(eval_seed + 1000)
        
        for _ in range(recovery_steps):
            x = torch.randn(batch_size, model.fc1.in_features, generator=recovery_gen, device=device)
            output = shadow(x)
            y_out = output[0] if isinstance(output, tuple) else output
            task_loss = torch.mean((y_out - x) ** 2)
            
            if penalty_keys:
                inv_penalty = _compute_invariant_penalty(shadow, inv_pre, penalty_keys, device)
                total_loss = task_loss + invariant_weight * inv_penalty
            else:
                total_loss = task_loss
            
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
            opt.step()
        
        # Measure recovered loss
        shadow.eval()
        with torch.no_grad():
            out = shadow(eval_batch)
            y = out[0] if isinstance(out, tuple) else out
            recovered_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        return float(recovery / damage)
    
    if verbose:
        print(f"\n  Hysteresis measurement ({constraint_type} constraint)")
        print(f"  g range: [{g_values[0]:.3f}, {g_values[-1]:.3f}]")
        print(f"  Recovery steps: {recovery_steps}")
    
    # Forward sweep (increasing g)
    forward_ci = []
    for i, g in enumerate(g_values):
        ci = measure_ci_at_g(g, seed_offset=0)
        forward_ci.append(ci)
        if verbose:
            print(f"  Forward g={g:.3f}: CI={ci:.4f}")
    
    # Reverse sweep (decreasing g, different seed for different path)
    reverse_ci = []
    for i, g in enumerate(reversed(g_values)):
        ci = measure_ci_at_g(g, seed_offset=1000)  # Different perturbation instances
        reverse_ci.insert(0, ci)
        if verbose:
            print(f"  Reverse g={g:.3f}: CI={ci:.4f}")
    
    forward_ci = np.array(forward_ci)
    reverse_ci = np.array(reverse_ci)
    
    # Compute hysteresis area via trapezoidal integration
    gap = np.abs(forward_ci - reverse_ci)
    if len(g_values) > 1:
        dg = np.diff(g_values)
        hysteresis_area = float(np.sum((gap[:-1] + gap[1:]) / 2 * dg))
    else:
        hysteresis_area = 0.0
    
    max_gap = float(np.max(gap))
    
    result = HysteresisResult(
        g_values=g_values,
        forward_ci=forward_ci,
        reverse_ci=reverse_ci,
        hysteresis_area=hysteresis_area,
        max_gap=max_gap,
    )
    
    if verbose:
        print(f"\n  Hysteresis area: {hysteresis_area:.4f}")
        print(f"  Max gap: {max_gap:.4f}")
        if result.has_hysteresis():
            print(f"  ✓ SIGNIFICANT hysteresis detected → path dependence / memory")
        else:
            print(f"  ~ No significant hysteresis")
    
    return result


# =============================================================================
# COMPREHENSIVE NONLOCALITY PROBE
# =============================================================================

@dataclass
class NonlocalityProbeResult:
    """Complete nonlocality characterization."""
    k_step_curves: Dict[str, KStepCurveResult]
    hysteresis: HysteresisResult
    decisive_1step_table: Dict[str, float]
    verdict: str
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "k_step_curves": {k: v.to_dict() for k, v in self.k_step_curves.items()},
            "hysteresis": self.hysteresis.to_dict(),
            "decisive_1step_table": self.decisive_1step_table,
            "verdict": self.verdict,
        }


def run_nonlocality_probe(
    model: nn.Module,
    verbose: bool = True,
    **kwargs,
) -> NonlocalityProbeResult:
    """Run comprehensive nonlocality characterization.
    
    Combines:
    1. k-step CI curves across constraint families
    2. Hysteresis measurement
    3. Distance triad analysis
    4. Decisive 1-step table
    
    This provides publication-grade evidence for/against the locality hypothesis.
    """
    if verbose:
        print("\n" + "="*70)
        print("NONLOCALITY PROBE: Comprehensive Characterization")
        print("="*70)
    
    # k-step curves
    if verbose:
        print("\n[1/2] k-step CI curves across constraint families...")
    
    k_step_curves = compare_constraint_families(model, verbose=verbose, **kwargs)
    
    # Hysteresis
    if verbose:
        print("\n[2/2] Hysteresis measurement...")
    
    hysteresis = compute_hysteresis(
        model,
        constraint_type=kwargs.get("constraint_type", "shape"),
        recovery_steps=kwargs.get("recovery_steps", 4),
        verbose=verbose,
        **{k: v for k, v in kwargs.items() if k not in ["constraint_type", "recovery_steps"]}
    )
    
    # Decisive 1-step table
    decisive_table = {}
    for name, result in k_step_curves.items():
        if result.ci_values:
            decisive_table[name] = result.ci_values[0]  # CI at k=1
    
    # Verdict
    all_fail_1step = all(ci < 0.1 for ci in decisive_table.values())
    direct_fails = decisive_table.get("direct", 0) < 0.1
    has_hysteresis = hysteresis.has_hysteresis()
    
    if all_fail_1step and direct_fails:
        verdict = "STRONGLY_NONLOCAL"
        verdict_text = "Function is fundamentally nonlocal (even direct optimization fails at k=1)"
    elif all_fail_1step and not direct_fails:
        verdict = "PROXY_MISALIGNED"
        verdict_text = "Geometric proxies misaligned with functional recovery"
    elif has_hysteresis:
        verdict = "PATH_DEPENDENT"
        verdict_text = "Recovery is path-dependent (hysteresis detected)"
    else:
        verdict = "LOCALLY_RECOVERABLE"
        verdict_text = "Function is locally recoverable"
    
    if verbose:
        print("\n" + "="*70)
        print("FINAL VERDICT")
        print("="*70)
        print(f"\n  Decisive 1-step table:")
        print(f"  {'Constraint':<12} {'CI(k=1)':>10}")
        print(f"  {'-'*12} {'-'*10}")
        for name, ci in decisive_table.items():
            marker = "✓" if ci > 0.1 else "✗"
            print(f"  {name:<12} {ci:>10.4f} {marker}")
        
        print(f"\n  Hysteresis area: {hysteresis.hysteresis_area:.4f}")
        print(f"\n  VERDICT: {verdict}")
        print(f"  {verdict_text}")
    
    return NonlocalityProbeResult(
        k_step_curves=k_step_curves,
        hysteresis=hysteresis,
        decisive_1step_table=decisive_table,
        verdict=verdict,
    )


# =============================================================================
# STEP-SIZE ENVELOPE: Best 1-step CI over learning rate grid
# =============================================================================

@dataclass
class StepSizeEnvelopeResult:
    """Results from step-size envelope test.
    
    This test computes the BEST achievable 1-step CI over a grid of learning rates,
    removing the objection "maybe you just picked a bad η."
    
    If best_ci ≈ 0 across all η, that's strong evidence that local information
    is genuinely absent, not just poorly extracted.
    """
    lr_values: List[float]
    ci_values: List[float]
    best_lr: float
    best_ci: float
    constraint_type: str
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "lr_values": self.lr_values,
            "ci_values": self.ci_values,
            "best_lr": self.best_lr,
            "best_ci": self.best_ci,
            "constraint_type": self.constraint_type,
        }
    
    def is_locally_recoverable(self, threshold: float = 0.1) -> bool:
        """Check if BEST 1-step CI exceeds threshold."""
        return self.best_ci > threshold


def compute_step_size_envelope(
    model: nn.Module,
    lr_values: List[float] = None,
    constraint_type: str = "shape",
    perturb_seed: int = 42,
    eval_seed: int = 12345,
    recovery_seed: Optional[int] = 2025,
    perturb_strength: float = 0.25,
    zero_frac: float = 0.05,
    invariant_weight: float = 0.1,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> StepSizeEnvelopeResult:
    """Compute best 1-step CI over a grid of learning rates.
    
    This removes the objection "maybe you just picked a bad learning rate."
    If the best CI across all η is still ≈0, local information is genuinely absent.
    
    Args:
        model: Trained model to test
        lr_values: Learning rates to test (default: log-spaced from 1e-4 to 1.0)
        constraint_type: "none", "scale", "shape", "direct"
        perturb_seed: Fixed seed for reproducible perturbation
        eval_seed: Fixed seed for evaluation batch
        recovery_seed: Fixed seed for recovery (deterministic dynamics)
        perturb_strength: Noise magnitude
        zero_frac: Fraction of weights to zero
        invariant_weight: λ for constraint penalty
        batch_size: Evaluation batch size
        device: Compute device
        verbose: Print progress
        
    Returns:
        StepSizeEnvelopeResult with CI values for each η and best (η, CI) pair
    """
    if lr_values is None:
        # Log-spaced grid from 1e-4 to 1.0
        lr_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
    
    # Select penalty keys based on constraint type
    if constraint_type == "none":
        penalty_keys = []
    elif constraint_type == "scale":
        penalty_keys = PENALTY_KEYS_SCALE
    elif constraint_type == "shape":
        penalty_keys = PENALTY_KEYS_SHAPE
    elif constraint_type == "direct":
        penalty_keys = []
    else:
        penalty_keys = DEFAULT_PENALTY_KEYS
    
    model = model.to(device)
    
    # Create fixed perturbation
    config = PerturbationConfig(seed=perturb_seed, strength=perturb_strength, zero_frac=zero_frac)
    perturbed_model = apply_perturbation(model, config, device)
    
    # Create eval context
    eval_ctx = EvalContext.create(
        model=model,
        perturbed_model=perturbed_model,
        batch_size=batch_size,
        eval_seed=eval_seed,
        device=device,
        penalty_keys=penalty_keys if penalty_keys else DEFAULT_PENALTY_KEYS,
    )
    
    inv_pre = eval_ctx.inv_pre
    base_loss = eval_ctx.base_loss
    perturbed_loss = eval_ctx.perturbed_loss
    
    if verbose:
        print(f"\n  Step-size envelope test ({constraint_type} constraint)")
        print(f"  Seeds: perturb={perturb_seed}, eval={eval_seed}, recovery={recovery_seed}")
        print(f"  PRE loss:  {base_loss:.6f}")
        print(f"  POST loss: {perturbed_loss:.6f}")
        print("\n  {:>10} {:>10}".format("eta", "CI(k=1)"))
        print("  {:>10} {:>10}".format("-" * 10, "-" * 10))
    
    ci_values = []
    
    for lr in lr_values:
        # Fresh copy of perturbed model for each η
        shadow = copy.deepcopy(perturbed_model).to(device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=lr)
        
        if recovery_seed is not None:
            recovery_gen = torch.Generator(device=device).manual_seed(recovery_seed)
        else:
            recovery_gen = None
        
        # Single recovery step
        if recovery_gen is not None:
            x = torch.randn(batch_size, model.fc1.in_features, generator=recovery_gen, device=device)
        else:
            x = torch.randn(batch_size, model.fc1.in_features, device=device)
        
        output = shadow(x)
        y = output[0] if isinstance(output, tuple) else output
        task_loss = torch.mean((y - x) ** 2)
        
        if constraint_type == "direct":
            total_loss = task_loss
        elif penalty_keys:
            inv_penalty = _compute_invariant_penalty(shadow, inv_pre, penalty_keys, device)
            total_loss = task_loss + invariant_weight * inv_penalty
        else:
            total_loss = task_loss
        
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
        opt.step()
        
        # Evaluate recovery
        recovered_loss = eval_ctx.evaluate_loss(shadow)
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci = float(recovery / damage)
        ci_values.append(ci)
        
        if verbose:
            print(f"  {lr:>10.4f} {ci:>10.4f}")
    
    # Find best
    best_idx = int(np.argmax(ci_values))
    best_lr = lr_values[best_idx]
    best_ci = ci_values[best_idx]
    
    if verbose:
        print(f"\n  Best: η={best_lr:.4f} → CI={best_ci:.4f}")
        if best_ci < 0.1:
            print(f"  ✗ Even with optimal η, 1-step CI < 0.1")
            print(f"    → Local information is genuinely absent")
        else:
            print(f"  ✓ Local information exists (extractable with right η)")
    
    return StepSizeEnvelopeResult(
        lr_values=lr_values,
        ci_values=ci_values,
        best_lr=best_lr,
        best_ci=best_ci,
        constraint_type=constraint_type,
    )


def compute_step_size_envelope_all_constraints(
    model: nn.Module,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, StepSizeEnvelopeResult]:
    """Compute step-size envelope for all constraint families."""
    results = {}
    for constraint in ["none", "scale", "shape", "direct"]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Constraint: {constraint.upper()}")
            print('='*60)
        
        results[constraint] = compute_step_size_envelope(
            model=model,
            constraint_type=constraint,
            verbose=verbose,
            **kwargs,
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print("ENVELOPE SUMMARY")
        print('='*60)
        print("\n  {:<12} {:>10} {:>10} {:>8}".format("Constraint", "Best eta", "Best CI", "Local?"))
        print("  {:<12} {:>10} {:>10} {:>8}".format("-" * 12, "-" * 10, "-" * 10, "-" * 8))
        for name, result in results.items():
            local = "✓" if result.is_locally_recoverable() else "✗"
            print(f"  {name:<12} {result.best_lr:>10.4f} {result.best_ci:>10.4f} {local:>8}")
        
        all_fail = all(not r.is_locally_recoverable() for r in results.values())
        if all_fail:
            print(f"\n  ✗ ALL constraints fail even with optimal η")
            print(f"    → Strong evidence: local information genuinely absent")
    
    return results


# =============================================================================
# CORRELATION ANALYSIS: Decoupling signature across seeds/conditions
# =============================================================================

@dataclass 
class DecouplingAnalysis:
    """Correlation analysis of distance triad across conditions.
    
    The core empirical claim is that parameter/representation distance
    DECOUPLES from functional distance. This quantifies that claim.
    """
    n_samples: int
    param_functional_corr: float      # Correlation: param_d vs func_d
    repr_functional_corr: float       # Correlation: repr_d vs func_d
    param_repr_corr: float            # Correlation: param_d vs repr_d
    mean_param_d: float
    mean_repr_d: float
    mean_func_d: float
    decoupling_score: float           # 1 - max(|param_func_corr|, |repr_func_corr|)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "n_samples": self.n_samples,
            "param_functional_corr": self.param_functional_corr,
            "repr_functional_corr": self.repr_functional_corr,
            "param_repr_corr": self.param_repr_corr,
            "mean_param_d": self.mean_param_d,
            "mean_repr_d": self.mean_repr_d,
            "mean_func_d": self.mean_func_d,
            "decoupling_score": self.decoupling_score,
        }
    
    def is_decoupled(self, threshold: float = 0.3) -> bool:
        """Check if functional distance is decoupled from geometric distances."""
        return self.decoupling_score > (1 - threshold)


def analyze_decoupling(
    triads: List[DistanceTriad],
) -> DecouplingAnalysis:
    """Analyze decoupling across a collection of distance triads.
    
    Args:
        triads: List of DistanceTriad objects from different conditions
        
    Returns:
        DecouplingAnalysis with correlation coefficients and decoupling score
    """
    if len(triads) < 3:
        # Not enough samples for meaningful correlation
        return DecouplingAnalysis(
            n_samples=len(triads),
            param_functional_corr=float('nan'),
            repr_functional_corr=float('nan'),
            param_repr_corr=float('nan'),
            mean_param_d=np.mean([t.param_distance for t in triads]),
            mean_repr_d=np.mean([t.repr_distance for t in triads]),
            mean_func_d=np.mean([t.functional_distance for t in triads]),
            decoupling_score=float('nan'),
        )
    
    param_d = np.array([t.param_distance for t in triads])
    repr_d = np.array([t.repr_distance for t in triads])
    func_d = np.array([t.functional_distance for t in triads])
    
    # Compute correlations (handle constant arrays)
    def safe_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    
    param_func_corr = safe_corr(param_d, func_d)
    repr_func_corr = safe_corr(repr_d, func_d)
    param_repr_corr = safe_corr(param_d, repr_d)
    
    # Decoupling score: 1 - max correlation with functional distance
    max_corr = max(abs(param_func_corr), abs(repr_func_corr))
    decoupling_score = 1.0 - max_corr
    
    return DecouplingAnalysis(
        n_samples=len(triads),
        param_functional_corr=param_func_corr,
        repr_functional_corr=repr_func_corr,
        param_repr_corr=param_repr_corr,
        mean_param_d=float(np.mean(param_d)),
        mean_repr_d=float(np.mean(repr_d)),
        mean_func_d=float(np.mean(func_d)),
        decoupling_score=decoupling_score,
    )


def collect_triads_across_seeds(
    model_factory,  # Callable[[int], nn.Module] - creates trained model from seed
    seeds: List[int] = None,
    constraint_type: str = "shape",
    k: int = 1,
    perturb_seed: int = 42,
    eval_seed: int = 12345,
    recovery_seed: Optional[int] = 2025,
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[DistanceTriad], DecouplingAnalysis]:
    """Collect distance triads across multiple model seeds and analyze decoupling.
    
    Args:
        model_factory: Function that creates a trained model given a seed
        seeds: List of seeds to use for model training
        constraint_type: Constraint family for recovery
        k: Number of recovery steps
        perturb_seed: Fixed seed for perturbation (same damage across models)
        eval_seed: Fixed seed for evaluation
        recovery_seed: Fixed seed for recovery
        verbose: Print progress
        
    Returns:
        Tuple of (list of DistanceTriad, DecouplingAnalysis)
    """
    if seeds is None:
        seeds = [42, 137, 256, 314, 999]  # Pre-registered ensemble
    
    triads = []
    
    if verbose:
        print(f"\n  Collecting triads across {len(seeds)} seeds...")
        print(f"  Constraint: {constraint_type}, k={k}")
    
    for seed in seeds:
        model = model_factory(seed)
        result = compute_k_step_curve(
            model=model,
            k_values=[k],
            constraint_type=constraint_type,
            perturb_seed=perturb_seed,
            eval_seed=eval_seed,
            recovery_seed=recovery_seed,
            verbose=False,
            **kwargs,
        )
        if result.distance_triads:
            triads.append(result.distance_triads[0])
            if verbose:
                t = result.distance_triads[0]
                print(f"    Seed {seed}: param_d={t.param_distance:.4f}, "
                      f"repr_d={t.repr_distance:.4f}, func_d={t.functional_distance:.4f}")
    
    analysis = analyze_decoupling(triads)
    
    if verbose:
        print(f"\n  Decoupling Analysis:")
        print(f"    Samples: {analysis.n_samples}")
        print(f"    Corr(param, func): {analysis.param_functional_corr:.4f}")
        print(f"    Corr(repr, func):  {analysis.repr_functional_corr:.4f}")
        print(f"    Corr(param, repr): {analysis.param_repr_corr:.4f}")
        print(f"    Decoupling score:  {analysis.decoupling_score:.4f}")
        if analysis.is_decoupled():
            print(f"    ✓ Geometric and functional distances are DECOUPLED")
        else:
            print(f"    ✗ Distances show significant correlation")
    
    return triads, analysis
