"""Eigenvalue repulsion dynamics for autodidactic learning.

This module implements a novel learning signal based on Random Matrix Theory:
eigenvalue repulsion as an organizing principle for learned dynamics.

Scientific hypotheses:
1. Systems that maintain eigenvalue repulsion are more expressive
2. Eigenvalue clustering indicates "law collapse" (loss of complexity)
3. The transition from Poisson to GOE statistics indicates learning

The key insight from RMT: in chaotic/complex systems, eigenvalues repel
each other. This repulsion prevents spectral collapse and maintains
the system's capacity for rich dynamics.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def eigenvalue_repulsion_energy(eigenvalues: np.ndarray) -> float:
    """Compute Coulomb-like repulsion energy between eigenvalues.

    E = -sum_{i<j} log|λ_i - λ_j|

    This is the log-gas potential from RMT. Lower energy = more repulsion.
    """
    w = np.sort(np.real(eigenvalues))
    n = len(w)
    if n < 2:
        return 0.0

    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(w[i] - w[j])
            if diff > 1e-12:
                energy -= np.log(diff)
            else:
                energy += 100.0  # Penalty for collision

    return energy


def wigner_semicircle_distance(eigenvalues: np.ndarray) -> float:
    """Compute distance from Wigner semicircle distribution.

    The Wigner semicircle is the limiting eigenvalue distribution for
    large random matrices (GOE). Proximity to this distribution indicates
    "maximal randomness" or "maximal complexity".
    """
    w = np.real(eigenvalues)
    n = len(w)

    # Normalize eigenvalues to have std 1
    std = np.std(w)
    if std < 1e-12:
        return float('inf')
    w_norm = w / std

    # Wigner semicircle: ρ(x) = (2/π) sqrt(1 - x^2/4) for |x| ≤ 2
    # Empirical CDF vs semicircle CDF
    def semicircle_cdf(x):
        x = np.clip(x, -2, 2)
        return 0.5 + x * np.sqrt(4 - x*x) / (2*np.pi) + np.arcsin(x/2) / np.pi

    w_sorted = np.sort(w_norm)
    empirical_cdf = np.arange(1, n+1) / n
    theoretical_cdf = semicircle_cdf(w_sorted)

    # Kolmogorov-Smirnov distance
    return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


def torch_eigenvalue_repulsion_loss(W: torch.Tensor) -> torch.Tensor:
    """Differentiable eigenvalue repulsion loss for PyTorch.

    Encourages eigenvalue spreading to maintain expressivity.
    Uses soft minimum to penalize small gaps.
    """
    # Symmetrize
    W_sym = (W + W.T) / 2.0

    # Eigenvalues (real for symmetric)
    eigs = torch.linalg.eigvalsh(W_sym)

    # Pairwise differences
    n = eigs.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=W.device)

    # Sorted eigenvalues
    eigs_sorted, _ = torch.sort(eigs)

    # Gaps between consecutive eigenvalues
    gaps = eigs_sorted[1:] - eigs_sorted[:-1]

    # Penalize small gaps (encourage repulsion)
    # Using log for scale invariance
    repulsion_loss = -torch.mean(torch.log(torch.abs(gaps) + 1e-8))

    return repulsion_loss


def torch_spectral_spread_loss(W: torch.Tensor, target_spread: float = 1.0) -> torch.Tensor:
    """Encourage eigenvalues to spread to a target range.

    This prevents both collapse (all eigenvalues same) and explosion.
    """
    W_sym = (W + W.T) / 2.0
    eigs = torch.linalg.eigvalsh(W_sym)

    spread = torch.max(eigs) - torch.min(eigs)
    return (spread - target_spread) ** 2


@dataclass
class EigenvalueRepulsionLearner:
    """Autodidactic learner that uses eigenvalue repulsion as a learning signal.

    This learner encourages the system to maintain spectral complexity by:
    1. Preventing eigenvalue collapse (clustering)
    2. Maintaining GOE-like level repulsion
    3. Balancing spread against stability

    Scientific motivation: Complex systems in physics exhibit eigenvalue
    repulsion. By encouraging this property, we may help neural systems
    maintain their capacity for rich, non-trivial dynamics.
    """
    lr: float = 1e-3
    repulsion_weight: float = 0.1
    spread_weight: float = 0.05
    target_spread: float = 2.0
    steps: int = 50
    batch_size: int = 64
    device: str = "cpu"

    def _get_weight_matrix(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Extract the main weight matrix from a model."""
        # Try common patterns
        if hasattr(model, 'core') and hasattr(model.core, 'W_rec'):
            return model.core.W_rec
        if hasattr(model, 'W_rec'):
            return model.W_rec
        # Fall back to largest square weight matrix
        for p in model.parameters():
            if p.dim() == 2 and p.shape[0] == p.shape[1]:
                return p
        return None

    def update(self, model: nn.Module) -> Dict[str, float]:
        """Perform learning updates with eigenvalue repulsion regularization."""
        model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        losses = []
        repulsion_losses = []
        spread_losses = []

        for _ in range(self.steps):
            x = torch.randn(self.batch_size, model.fc1.in_features, device=self.device)
            y = model(x)

            # Self-consistency loss
            recon_loss = torch.mean((y - x) ** 2)

            # Eigenvalue repulsion on weight matrix
            W = self._get_weight_matrix(model)
            if W is not None:
                rep_loss = torch_eigenvalue_repulsion_loss(W)
                spread_loss = torch_spectral_spread_loss(W, self.target_spread)
            else:
                rep_loss = torch.tensor(0.0, device=self.device)
                spread_loss = torch.tensor(0.0, device=self.device)

            total_loss = (
                recon_loss
                + self.repulsion_weight * rep_loss
                + self.spread_weight * spread_loss
            )

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(recon_loss.item())
            repulsion_losses.append(rep_loss.item())
            spread_losses.append(spread_loss.item())

        return {
            "recon_loss": float(np.mean(losses)),
            "repulsion_loss": float(np.mean(repulsion_losses)),
            "spread_loss": float(np.mean(spread_losses)),
            "total_loss": float(np.mean(losses) +
                               self.repulsion_weight * np.mean(repulsion_losses) +
                               self.spread_weight * np.mean(spread_losses)),
        }


def diagnose_spectral_health(model: nn.Module) -> Dict[str, float]:
    """Diagnose the spectral health of a model's weight matrices.

    Returns metrics indicating whether the model is at risk of
    spectral collapse or maintains healthy eigenvalue repulsion.
    """
    metrics = {}

    # Find weight matrices
    weight_matrices = []
    for name, p in model.named_parameters():
        if p.dim() == 2 and p.shape[0] == p.shape[1]:
            weight_matrices.append((name, p.detach().cpu().numpy()))

    for name, W in weight_matrices:
        W_sym = (W + W.T) / 2.0
        eigs = np.linalg.eigvalsh(W_sym)

        prefix = name.replace('.', '_')
        metrics[f"{prefix}_repulsion_energy"] = eigenvalue_repulsion_energy(eigs)
        metrics[f"{prefix}_wigner_distance"] = wigner_semicircle_distance(eigs)
        metrics[f"{prefix}_spectral_spread"] = float(np.max(eigs) - np.min(eigs))
        metrics[f"{prefix}_min_gap"] = float(np.min(np.diff(np.sort(eigs))))

    return metrics
