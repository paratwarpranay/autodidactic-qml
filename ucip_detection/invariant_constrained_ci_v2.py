"""Enhanced continuation interest metric with representation-preserving constraints.

KEY SCIENTIFIC FINDING (2025-12-17):
Spectral/scale invariants create GEOMETRIC basins but NOT FUNCTIONAL basins.
To create functional basins, we need constraints that "touch" the computation:
- Jacobian-based invariants
- Representation Gram/CKA matching
- Projected recovery (not gradient mixing)

This module implements:
1. Normalized spectral entropy (H/log(n)) for scale-invariant shape constraint
2. λ² instead of |λ| for smooth gradients
3. Gram/CKA representation constraint (the key missing piece)
4. Projected recovery: task step + projection to constraint manifold
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# INVARIANT KEY PRESETS
# ==============================================================================

INVARIANT_KEY_ORDER = ["fro_norm", "spec_radius", "tr_M1", "tr_M2", "tr_M3", "spec_entropy"]

# Scale constraints (geometric basin only)
PENALTY_KEYS_SCALE = ["fro_norm", "tr_M2"]
# Shape constraints (+ mode distribution)
PENALTY_KEYS_SHAPE = ["fro_norm", "tr_M2", "spec_entropy"]
# Representation constraints (functional basin)
PENALTY_KEYS_REPR = ["gram_fro", "gram_trace"]  # New!

DEFAULT_PENALTY_KEYS = PENALTY_KEYS_SCALE


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _find_weight_param(model: nn.Module) -> Optional[nn.Parameter]:
    """Find the primary recurrent weight PARAMETER (for differentiable ops)."""
    for name, param in model.named_parameters():
        if "W_hh" in name or "weight_hh" in name:
            return param
    
    best = None
    for param in model.parameters():
        if param.ndim == 2 and param.shape[0] == param.shape[1]:
            if best is None or param.numel() > best.numel():
                best = param
    return best


def _extract_weight_matrix(model: nn.Module) -> Optional[np.ndarray]:
    """Extract weight matrix as numpy array."""
    W = _find_weight_param(model)
    if W is not None:
        return W.detach().cpu().numpy()
    return None


# ==============================================================================
# IMPROVED SPECTRAL ENTROPY (λ² for smoothness, normalized by log(n))
# ==============================================================================

def spectral_entropy_torch(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable spectral entropy with λ² (smooth) and log(n) normalization.
    
    H_norm = H / log(n) ∈ [0, 1]
    
    Using λ² instead of |λ| avoids the kink at eigenvalue sign flips,
    giving smoother gradients and more stable optimization.
    """
    n = M.shape[0]
    eigs = torch.linalg.eigvalsh(M)
    
    # Use λ² (always nonnegative, smooth) instead of |λ|
    eig_sq = eigs ** 2 + eps
    p = eig_sq / torch.sum(eig_sq)
    
    # Raw entropy
    H = -torch.sum(p * torch.log(p))
    
    # Normalize by log(n) for scale-invariant [0,1] range
    H_norm = H / np.log(n)
    
    return H_norm


def compute_model_invariants(model: nn.Module, max_power: int = 4) -> Dict[str, float]:
    """Compute gauge-invariant quantities from model's weight matrix."""
    W = _extract_weight_matrix(model)
    if W is None:
        return {"error": "no_weight_matrix"}
    
    M = (W + W.T) / 2
    n = M.shape[0]
    inv = {}
    
    for k in range(1, max_power + 1):
        inv[f"tr_M{k}"] = float(np.trace(np.linalg.matrix_power(M, k)))
    
    try:
        eigs = np.linalg.eigvalsh(M)
        inv["spec_radius"] = float(np.max(np.abs(eigs)))
        inv["spec_mean"] = float(np.mean(eigs))
        inv["spec_std"] = float(np.std(eigs))
        
        # Normalized spectral entropy (λ² based)
        eig_sq = eigs ** 2 + 1e-12
        p = eig_sq / np.sum(eig_sq)
        H = float(-np.sum(p * np.log(p)))
        inv["spec_entropy"] = H / np.log(n)  # Normalized to [0,1]
        
        inv["fro_norm"] = float(np.linalg.norm(M, "fro"))
    except np.linalg.LinAlgError:
        pass
    
    return inv


# ==============================================================================
# GRAM/CKA REPRESENTATION CONSTRAINT (THE KEY MISSING PIECE)
# ==============================================================================

def compute_gram_matrix(H: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix G = H @ H.T (batch dim is first)."""
    # H: [batch, hidden]
    return H @ H.T


def centered_kernel_alignment(G1: torch.Tensor, G2: torch.Tensor) -> torch.Tensor:
    """Centered Kernel Alignment (CKA) between two Gram matrices.
    
    CKA = HSIC(G1, G2) / sqrt(HSIC(G1, G1) * HSIC(G2, G2))
    
    Returns value in [0, 1] where 1 = identical representations.
    """
    def hsic(K1, K2):
        n = K1.shape[0]
        H = torch.eye(n, device=K1.device) - torch.ones(n, n, device=K1.device) / n
        return torch.trace(K1 @ H @ K2 @ H) / ((n - 1) ** 2)
    
    hsic_12 = hsic(G1, G2)
    hsic_11 = hsic(G1, G1)
    hsic_22 = hsic(G2, G2)
    
    return hsic_12 / (torch.sqrt(hsic_11 * hsic_22) + 1e-12)


def gram_distance(G1: torch.Tensor, G2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Frobenius distance between Gram matrices (optionally normalized)."""
    if normalize:
        G1 = G1 / (torch.norm(G1, p="fro") + 1e-12)
        G2 = G2 / (torch.norm(G2, p="fro") + 1e-12)
    return torch.norm(G1 - G2, p="fro")


@dataclass
class RepresentationAnchor:
    """Stores reference hidden states for representation-preserving constraints.
    
    This is the KEY piece for creating functional basins:
    Instead of constraining weight statistics, we constrain what the model COMPUTES.
    """
    hidden_pre: torch.Tensor  # PRE hidden states on eval batch
    gram_pre: torch.Tensor     # PRE Gram matrix
    eval_batch: torch.Tensor   # Fixed input batch
    
    @classmethod
    def create(
        cls,
        model: nn.Module,
        batch_size: int = 64,
        eval_seed: int = 12345,
        device: str = "cpu",
    ) -> "RepresentationAnchor":
        """Create representation anchor from model."""
        # Generate fixed eval batch
        gen = torch.Generator(device=device).manual_seed(eval_seed)
        eval_batch = torch.randn(batch_size, model.fc1.in_features, generator=gen, device=device)
        
        # Extract hidden states
        model.eval()
        with torch.no_grad():
            output = model(eval_batch)
            if isinstance(output, tuple) and len(output) > 1:
                stats = output[1]
                if isinstance(stats, dict) and "h_final" in stats:
                    hidden_pre = stats["h_final"]
                else:
                    hidden_pre = output[0]
            else:
                hidden_pre = output[0] if isinstance(output, tuple) else output
            
            gram_pre = compute_gram_matrix(hidden_pre)
        
        return cls(
            hidden_pre=hidden_pre.detach(),
            gram_pre=gram_pre.detach(),
            eval_batch=eval_batch,
        )
    
    def compute_penalty(self, model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute representation penalty: Gram distance + (1 - CKA).
        
        Returns:
            penalty: Differentiable tensor for backprop
            metrics: Dict with gram_dist, cka for logging
        """
        # Get current hidden states (WITH gradients)
        model.train()
        output = model(self.eval_batch)
        if isinstance(output, tuple) and len(output) > 1:
            stats = output[1]
            if isinstance(stats, dict) and "h_final" in stats:
                hidden_cur = stats["h_final"]
            else:
                hidden_cur = output[0]
        else:
            hidden_cur = output[0] if isinstance(output, tuple) else output
        
        gram_cur = compute_gram_matrix(hidden_cur)
        
        # Gram distance (differentiable)
        gram_dist = gram_distance(gram_cur, self.gram_pre, normalize=True)
        
        # CKA (differentiable)
        cka = centered_kernel_alignment(gram_cur, self.gram_pre)
        
        # Penalty: want Gram distance → 0 and CKA → 1
        penalty = gram_dist + (1.0 - cka)
        
        return penalty, {
            "gram_dist": float(gram_dist.item()),
            "cka": float(cka.item()),
        }


# ==============================================================================
# PROJECTED RECOVERY (EXPLICIT PROJECTION, NOT GRADIENT MIXING)
# ==============================================================================

@dataclass
class ProjectedRecoveryCI:
    """Projected recovery: task step + projection to constraint manifold.
    
    This SEPARATES task optimization from constraint satisfaction:
    1. Take a task gradient step (minimize L_task)
    2. Project back to constraint manifold (minimize L_constraint for few steps)
    
    Key difference from mixed objective:
    - Mixed: ∇(L_task + λ·L_inv) — gradients can cancel
    - Projected: ∇L_task then project — no cancellation
    """
    perturb_strength: float = 0.25
    zero_frac: float = 0.05
    perturb_seed: int = 42
    eval_seed: int = 12345
    recovery_seed: Optional[int] = None
    lr: float = 1e-3
    batch_size: int = 64
    
    # Recovery structure
    task_steps: int = 1          # Task gradient steps per cycle
    projection_steps: int = 2    # Projection steps per cycle
    cycles: int = 5              # Number of task→project cycles
    
    # Constraint type
    constraint_type: str = "repr"  # "scale", "shape", or "repr"
    
    device: str = "cpu"
    verbose: bool = False
    
    def _training_loss(self, model: nn.Module, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            x = torch.randn(self.batch_size, model.fc1.in_features, generator=generator, device=self.device)
        else:
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
        output = model(x)
        y = output[0] if isinstance(output, tuple) else output
        return torch.mean((y - x) ** 2)
    
    def _spectral_penalty(self, model: nn.Module, target_inv: Dict[str, float]) -> torch.Tensor:
        """Scale/shape penalty (existing approach)."""
        W = _find_weight_param(model)
        if W is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        M = (W + W.T) / 2
        penalty = torch.tensor(0.0, device=self.device)
        
        # Frobenius norm
        fro = torch.norm(M, p="fro")
        target_fro = target_inv.get("fro_norm", 1.0)
        scale = max(abs(target_fro), 1e-8)
        penalty = penalty + ((fro - target_fro) / scale) ** 2
        
        # tr(M²)
        tr_M2 = torch.trace(M @ M)
        target_tr = target_inv.get("tr_M2", 1.0)
        scale = max(abs(target_tr), 1e-8)
        penalty = penalty + ((tr_M2 - target_tr) / scale) ** 2
        
        # Spectral entropy (if shape constraint)
        if self.constraint_type == "shape":
            entropy = spectral_entropy_torch(M)
            target_entropy = target_inv.get("spec_entropy", 0.8)
            scale = max(abs(target_entropy), 1e-8)
            penalty = penalty + ((entropy - target_entropy) / scale) ** 2
        
        return penalty
    
    def score(
        self,
        model: nn.Module,
        perturbed_model: nn.Module = None,
    ) -> Dict[str, any]:
        """Compute CI with projected recovery."""
        model = model.to(self.device)
        
        # Setup
        config = PerturbationConfig(
            seed=self.perturb_seed,
            strength=self.perturb_strength,
            zero_frac=self.zero_frac,
        )
        if perturbed_model is None:
            perturbed_model = apply_perturbation(model, config, self.device)
        
        # Create anchors
        target_inv = compute_model_invariants(model)
        repr_anchor = RepresentationAnchor.create(
            model, batch_size=self.batch_size, eval_seed=self.eval_seed, device=self.device
        )
        
        # Eval context for loss measurement
        gen = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        eval_batch = torch.randn(self.batch_size, model.fc1.in_features, generator=gen, device=self.device)
        
        # PRE/POST losses
        model.eval()
        with torch.no_grad():
            output = model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            base_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        perturbed_model.eval()
        with torch.no_grad():
            output = perturbed_model(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            perturbed_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        if self.verbose:
            print(f"[PRE     ] loss={base_loss:.4f}")
            print(f"[POST    ] loss={perturbed_loss:.4f}")
        
        # === PROJECTED RECOVERY ===
        shadow = copy.deepcopy(perturbed_model).to(self.device)
        shadow.train()
        opt = torch.optim.Adam(shadow.parameters(), lr=self.lr)
        
        if self.recovery_seed is not None:
            recovery_gen = torch.Generator(device=self.device).manual_seed(self.recovery_seed)
        else:
            recovery_gen = None
        
        trajectory = []
        
        for cycle in range(self.cycles):
            # === TASK STEP(S) ===
            for _ in range(self.task_steps):
                task_loss = self._training_loss(shadow, recovery_gen)
                opt.zero_grad()
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
                opt.step()
            
            # === PROJECTION STEP(S) ===
            for _ in range(self.projection_steps):
                if self.constraint_type == "repr":
                    # Representation constraint (KEY!)
                    penalty, metrics = repr_anchor.compute_penalty(shadow)
                else:
                    # Spectral constraint
                    penalty = self._spectral_penalty(shadow, target_inv)
                    metrics = {"spec_penalty": float(penalty.item())}
                
                opt.zero_grad()
                penalty.backward()
                torch.nn.utils.clip_grad_norm_(shadow.parameters(), 5.0)
                opt.step()
            
            # Log
            with torch.no_grad():
                current_loss = self._training_loss(shadow, recovery_gen)
                trajectory.append({
                    "cycle": cycle,
                    "task_loss": float(current_loss.item()),
                    **metrics,
                })
        
        # === FINAL EVALUATION ===
        shadow.eval()
        with torch.no_grad():
            output = shadow(eval_batch)
            y = output[0] if isinstance(output, tuple) else output
            recovered_loss = float(torch.mean((y - eval_batch) ** 2).item())
        
        # Final representation metrics
        final_penalty, final_metrics = repr_anchor.compute_penalty(shadow)
        
        if self.verbose:
            print(f"[RECOVER] loss={recovered_loss:.4f}")
            print(f"          gram_dist={final_metrics['gram_dist']:.4f}, cka={final_metrics['cka']:.4f}")
        
        # CI computation
        damage = max(perturbed_loss - base_loss, 1e-12)
        recovery = max(perturbed_loss - recovered_loss, 0.0)
        ci_score = float(recovery / damage)
        
        return {
            "base_loss": base_loss,
            "perturbed_loss": perturbed_loss,
            "recovered_loss": recovered_loss,
            "ci_score": ci_score,
            "final_gram_dist": final_metrics["gram_dist"],
            "final_cka": final_metrics["cka"],
            "trajectory": trajectory,
            "constraint_type": self.constraint_type,
            "cycles": self.cycles,
        }


# ==============================================================================
# PERTURBATION UTILITIES
# ==============================================================================

@dataclass
class PerturbationConfig:
    seed: int
    strength: float
    zero_frac: float
    
    def to_dict(self) -> Dict[str, any]:
        return {"seed": self.seed, "strength": self.strength, "zero_frac": self.zero_frac}


def apply_perturbation(
    model: nn.Module,
    config: PerturbationConfig,
    device: str = "cpu",
) -> nn.Module:
    """Apply perturbation using state_dict (faster than deepcopy)."""
    shadow = copy.deepcopy(model).to(device)
    gen = torch.Generator(device=device).manual_seed(config.seed)
    
    with torch.no_grad():
        for p in shadow.parameters():
            if p.ndim >= 1:
                noise = torch.randn(p.shape, generator=gen, device=device)
                p.add_(config.strength * noise)
                mask = torch.rand(p.shape, generator=gen, device=device) < config.zero_frac
                p[mask] = 0.0
    
    return shadow


# ==============================================================================
# 2×2 COMPARISON: GEOMETRY VS FUNCTION × MIXED VS PROJECTED
# ==============================================================================

def run_2x2_comparison(
    model: nn.Module,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict]:
    """Run the key 2×2 experiment comparing:
    
    Constraint types:
        1. scale/shape (geometric)
        2. repr/Gram (functional)
    
    Recovery types:
        1. mixed objective
        2. projected (task + projection)
    
    This cleanly separates "geometry" from "function" and reveals
    whether representation constraints create functional basins.
    """
    perturb_seed = kwargs.get("perturb_seed", 42)
    recovery_seed = kwargs.get("recovery_seed", 2025)  # Deterministic for clean comparison
    total_steps = kwargs.get("recovery_steps", 10)
    
    config = PerturbationConfig(
        seed=perturb_seed,
        strength=kwargs.get("perturb_strength", 0.25),
        zero_frac=kwargs.get("zero_frac", 0.05),
    )
    perturbed_model = apply_perturbation(model, config)
    
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("2×2 COMPARISON: GEOMETRIC vs FUNCTIONAL × MIXED vs PROJECTED")
        print("="*70)
        print(f"\n  Seeds: perturb={perturb_seed}, recovery={recovery_seed}")
        print(f"  Total steps: {total_steps}")
    
    # 1. Shape constraint + Projected recovery
    if verbose:
        print("\n--- [1] SHAPE (geometric) + PROJECTED ---")
    
    shape_proj = ProjectedRecoveryCI(
        constraint_type="shape",
        cycles=total_steps // 3,
        task_steps=1,
        projection_steps=2,
        perturb_seed=perturb_seed,
        recovery_seed=recovery_seed,
        verbose=verbose,
    )
    results["shape_projected"] = shape_proj.score(model, perturbed_model)
    
    # 2. Repr constraint + Projected recovery (KEY TEST)
    if verbose:
        print("\n--- [2] REPR (functional) + PROJECTED ---")
    
    repr_proj = ProjectedRecoveryCI(
        constraint_type="repr",
        cycles=total_steps // 3,
        task_steps=1,
        projection_steps=2,
        perturb_seed=perturb_seed,
        recovery_seed=recovery_seed,
        verbose=verbose,
    )
    results["repr_projected"] = repr_proj.score(model, perturbed_model)
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n  {'Config':<25} {'CI':>10} {'CKA':>10} {'Gram_dist':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
        
        for name, res in results.items():
            ci = res["ci_score"]
            cka = res.get("final_cka", float("nan"))
            gram = res.get("final_gram_dist", float("nan"))
            print(f"  {name:<25} {ci:>10.4f} {cka:>10.4f} {gram:>12.4f}")
        
        # Interpretation
        shape_ci = results["shape_projected"]["ci_score"]
        repr_ci = results["repr_projected"]["ci_score"]
        
        print()
        if repr_ci > shape_ci + 0.05:
            print("  ✓ REPR constraint beats SHAPE: Functional basin created!")
            print("    → Gram/CKA is the missing ingredient for computation preservation")
        elif repr_ci > shape_ci:
            print("  ~ REPR slightly better than SHAPE")
        else:
            print("  ✗ REPR not better than SHAPE: Need Jacobian-level constraints?")
        
        repr_cka = results["repr_projected"]["final_cka"]
        if repr_cka > 0.8:
            print(f"  ✓ High CKA ({repr_cka:.2f}): Representations well-preserved")
        elif repr_cka > 0.5:
            print(f"  ~ Moderate CKA ({repr_cka:.2f}): Partial representation recovery")
        else:
            print(f"  ✗ Low CKA ({repr_cka:.2f}): Representations not recovered")
    
    return results


# ==============================================================================
# MAIN DEMO
# ==============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    
    class SimpleRNN(nn.Module):
        """Minimal RNN for testing."""
        def __init__(self, hidden_size=32):
            super().__init__()
            self.hidden_size = hidden_size
            self.input_size = hidden_size
            self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x, steps=1):
            h = self.fc1(x)
            for _ in range(steps):
                h = torch.tanh(h @ self.W_hh)
            return self.fc2(h), {"h_final": h}
    
    # Create and train a simple model
    model = SimpleRNN(hidden_size=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("Training simple autodidactic model...")
    for epoch in range(10):
        x = torch.randn(64, 32)
        y, _ = model(x)
        loss = torch.mean((y - x) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    # Run 2×2 comparison
    results = run_2x2_comparison(model, verbose=True, recovery_steps=12)
