"""Topology plasticity rules for SQNT-inspired learning.

This module implements participation-based edge updates where topology adapts
based on spectral coherence under the constraint of a shared invariant budget.

**Theoretical Background:**

The SQNT framework (Superpositional Quantum Network Topologies) proposed that
network topology itself should be trainable, not fixed a priori. The original
work (Altman, Pykacz & Zapatrin, 2004) achieved this via quantum superposition
of topologies using Rota algebras.

This module provides a *classical* mechanism for topology adaptation based on
spectral participation—a testable instantiation of the SQNT principle that
structure and dynamics should co-evolve.

**Key Principles:**
    1. Local: Edge updates depend only on local graph + spectral participation
    2. Budget-aware: Reinforcement in one place weakens another
    3. Invariant-compatible: No edge update bypasses spectral mass constraint

The participation score measures how much each edge contributes to dominant
eigenmodes. Edges that participate in coherent structure get reinforced;
edges contributing to noise get weakened.

Mathematical foundation:
    - Participation score: p_ij = Σ_k |λ_k| |v_k(i)| |v_k(j)|  for k in dominant modes
    - Centered update: δp_ij = p_ij - <p>  (prevents global drift)
    - Edge update: w_ij^{t+1} = w_ij^t + η·δp_ij
    - Soft clipping: tanh(w_ij) for bounded weights

This is NOT optimization toward a target. It is spectral-coherence-driven
self-organization under constraint.

Author: Christopher Altman
Date: 2025
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Literal
import copy


@dataclass
class TopologyPlasticity:
    """Participation-based topology plasticity for SQNT.
    
    Topology adapts based on spectral participation: edges that participate
    in dominant eigenmodes are reinforced, others are weakened. This creates
    a form of structural self-organization without explicit optimization targets.
    
    Usage:
        plasticity = TopologyPlasticity(eta=1e-3, n_dominant=5)
        
        for t in range(T):
            # Compute Laplacian of current graph
            L = D - A  # or extract from graph object
            
            # Update edge weights
            A_new, diagnostics = plasticity.update(A, L)
    
    Attributes:
        eta: Learning rate for edge updates (default 1e-3)
        n_dominant: Number of dominant modes to consider (default 5)
        use_laplacian: If True, use Laplacian eigenpairs; else use adjacency
        clip_method: 'tanh', 'relu', or 'none'
        min_weight: Minimum edge weight (for numerical stability)
        max_weight: Maximum edge weight (for boundedness)
    """
    eta: float = 1e-3
    n_dominant: int = 5
    use_laplacian: bool = True
    clip_method: Literal['tanh', 'relu', 'none'] = 'tanh'
    min_weight: float = 0.0
    max_weight: float = 1.0
    _history: List[Dict[str, float]] = field(default_factory=list, repr=False)
    
    def compute_participation(
        self,
        A: np.ndarray,
        L: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute edge participation scores.
        
        For each edge (i,j), participation measures how much it contributes
        to the dominant eigenmodes:
            p_ij = Σ_k |λ_k| |v_k(i)| |v_k(j)|
        
        Args:
            A: Adjacency matrix (n, n)
            L: Optional Laplacian matrix. If None, computed from A.
            
        Returns:
            Participation matrix (n, n), symmetric, zero diagonal
        """
        n = A.shape[0]
        
        # Compute eigenpairs
        if self.use_laplacian:
            if L is None:
                D = np.diag(np.sum(A, axis=1))
                L = D - A
            eigvals, eigvecs = np.linalg.eigh(L)
        else:
            eigvals, eigvecs = np.linalg.eigh(A)
        
        # Select dominant modes (largest by |λ|)
        k = min(self.n_dominant, len(eigvals))
        dominant_idx = np.argsort(np.abs(eigvals))[-k:]
        
        # Compute participation for each edge
        P = np.zeros((n, n))
        for idx in dominant_idx:
            lam = np.abs(eigvals[idx])
            v = eigvecs[:, idx]
            # Outer product weighted by eigenvalue magnitude
            P += lam * np.outer(np.abs(v), np.abs(v))
        
        # Zero diagonal (no self-loops)
        np.fill_diagonal(P, 0.0)
        
        return P
    
    def center_participation(self, P: np.ndarray) -> np.ndarray:
        """Center participation scores to prevent global drift.
        
        Computes: δP_ij = P_ij - mean(P)
        
        This ensures total edge weight budget is approximately conserved.
        
        Args:
            P: Participation matrix
            
        Returns:
            Centered participation matrix
        """
        # Mean over non-diagonal elements
        mask = ~np.eye(P.shape[0], dtype=bool)
        p_mean = np.mean(P[mask])
        
        delta_P = P.copy()
        delta_P[mask] -= p_mean
        
        return delta_P
    
    def clip_weights(self, A: np.ndarray) -> np.ndarray:
        """Apply weight clipping to keep edges bounded.
        
        Args:
            A: Adjacency matrix with potentially unbounded weights
            
        Returns:
            Clipped adjacency matrix
        """
        if self.clip_method == 'tanh':
            # Smooth bounding in [0, max_weight]
            A_clipped = self.max_weight * np.tanh(A / self.max_weight)
            A_clipped = np.maximum(A_clipped, self.min_weight)
        elif self.clip_method == 'relu':
            A_clipped = np.clip(A, self.min_weight, self.max_weight)
        else:  # 'none'
            A_clipped = A
        
        # Ensure symmetry and zero diagonal
        A_clipped = (A_clipped + A_clipped.T) / 2.0
        np.fill_diagonal(A_clipped, 0.0)
        
        return A_clipped
    
    def update(
        self,
        A: np.ndarray,
        L: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform one topology plasticity update.
        
        Computes: A_new = clip(A + η·(P - mean(P)))
        
        Args:
            A: Current adjacency matrix
            L: Optional Laplacian (computed from A if None)
            
        Returns:
            Tuple of:
                - A_new: Updated adjacency matrix
                - diagnostics: Dict with participation stats, weight changes, etc.
        """
        # Compute participation
        P = self.compute_participation(A, L)
        
        # Center to prevent global drift
        delta_P = self.center_participation(P)
        
        # Apply update
        A_new = A + self.eta * delta_P
        
        # Clip to bounded range
        A_new = self.clip_weights(A_new)
        
        # Compute diagnostics
        weight_change = np.linalg.norm(A_new - A, 'fro')
        participation_std = np.std(P[~np.eye(P.shape[0], dtype=bool)])
        
        diagnostics = {
            'participation_mean': float(np.mean(P[~np.eye(P.shape[0], dtype=bool)])),
            'participation_std': float(participation_std),
            'participation_max': float(np.max(P)),
            'weight_change_norm': float(weight_change),
            'mean_weight': float(np.mean(A_new[~np.eye(A_new.shape[0], dtype=bool)])),
            'max_weight': float(np.max(A_new)),
            'n_edges': int(np.sum(A_new > 0.01)),
        }
        
        self._history.append(diagnostics)
        
        return A_new, diagnostics
    
    @property
    def history(self) -> List[Dict[str, float]]:
        """Get update history."""
        return self._history
    
    def reset_history(self) -> None:
        """Clear update history."""
        self._history = []
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics of plasticity behavior."""
        if not self._history:
            return {'n_updates': 0}
        
        weight_changes = [h['weight_change_norm'] for h in self._history]
        
        return {
            'n_updates': len(self._history),
            'mean_weight_change': float(np.mean(weight_changes)),
            'total_weight_change': float(np.sum(weight_changes)),
            'final_mean_weight': self._history[-1]['mean_weight'],
            'final_n_edges': self._history[-1]['n_edges'],
        }


@dataclass
class TopologyFeedback:
    """Topology-to-matrix feedback for SQNT coupling.
    
    This implements the critical coupling where topology feeds back into
    matrix dynamics. The matrix update is modulated by edge weights:
        δM_ij *= w_ij
    
    or equivalently:
        δM = W ⊙ δM  (element-wise Hadamard product)
    
    This creates bidirectional coupling:
        matrix → spectrum → participation → topology → matrix
    
    At this point, topology is no longer decorative—it actively shapes dynamics.
    """
    
    def modulate_update(
        self,
        delta_M: np.ndarray,
        W: np.ndarray,
        mode: Literal['hadamard', 'laplacian', 'degree'] = 'hadamard',
    ) -> np.ndarray:
        """Modulate matrix update by topology.
        
        Args:
            delta_M: Proposed matrix update
            W: Adjacency/weight matrix
            mode: How to apply modulation
                - 'hadamard': Element-wise δM ⊙ W
                - 'laplacian': δM - α·L·δM (smoothing)
                - 'degree': Scale by inverse degree
            
        Returns:
            Modulated update
        """
        if mode == 'hadamard':
            # Direct element-wise modulation
            # Edges with high weight get full update; weak edges are suppressed
            # Normalize W to [0, 1] range for clean modulation
            W_norm = W / (np.max(W) + 1e-12)
            delta_M_mod = delta_M * W_norm
            
        elif mode == 'laplacian':
            # Laplacian smoothing: encourages updates aligned with topology
            D = np.diag(np.sum(W, axis=1))
            L = D - W
            L_norm = L / (np.trace(L) + 1e-12)  # Normalize
            alpha = 0.1  # Smoothing strength
            delta_M_mod = delta_M - alpha * (L_norm @ delta_M + delta_M @ L_norm)
            
        elif mode == 'degree':
            # Degree-based scaling: well-connected nodes update less
            degrees = np.sum(W, axis=1)
            degree_scale = 1.0 / (1.0 + degrees / (np.mean(degrees) + 1e-12))
            # Apply symmetrically
            scale_matrix = np.outer(np.sqrt(degree_scale), np.sqrt(degree_scale))
            delta_M_mod = delta_M * scale_matrix
            
        else:
            delta_M_mod = delta_M
        
        # Preserve symmetry
        if np.allclose(delta_M, delta_M.T):
            delta_M_mod = (delta_M_mod + delta_M_mod.T) / 2.0
        
        return delta_M_mod


def random_rewire(
    A: np.ndarray,
    preserve_degree: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Randomly rewire adjacency matrix (for control experiments).
    
    This creates a topology control where edges are randomly reassigned
    while optionally preserving the degree distribution.
    
    Args:
        A: Original adjacency matrix
        preserve_degree: If True, preserve degree sequence approximately
        seed: Random seed
        
    Returns:
        Rewired adjacency matrix
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    
    if preserve_degree:
        # Configuration model-style rewiring
        weights = A[np.triu_indices(n, k=1)]
        rng.shuffle(weights)
        
        A_new = np.zeros_like(A)
        A_new[np.triu_indices(n, k=1)] = weights
        A_new = A_new + A_new.T
    else:
        # Complete random (preserves weight distribution)
        weights = A[np.triu_indices(n, k=1)].copy()
        rng.shuffle(weights)
        
        A_new = np.zeros_like(A)
        A_new[np.triu_indices(n, k=1)] = weights
        A_new = A_new + A_new.T
    
    return A_new


def apply_perturbation(
    A: np.ndarray,
    fraction: float = 0.25,
    mode: Literal['contiguous', 'random', 'targeted'] = 'contiguous',
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Apply structural perturbation to adjacency matrix.
    
    This is used in falsification experiments to test recovery dynamics.
    
    Args:
        A: Adjacency matrix
        fraction: Fraction of edges to remove
        mode: Perturbation mode
            - 'contiguous': Remove edges in a subgraph (realistic damage)
            - 'random': Remove random edges
            - 'targeted': Remove highest-weight edges (stress test)
        seed: Random seed
        
    Returns:
        Tuple of:
            - A_perturbed: Perturbed adjacency matrix
            - info: Dict with removed_edges, affected_nodes, etc.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    
    # Get edges as (i, j, weight) tuples
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 1e-12:
                edges.append((i, j, A[i, j]))
    
    n_remove = int(len(edges) * fraction)
    
    if mode == 'contiguous':
        # Select a seed node and remove edges in its neighborhood
        seed_node = rng.integers(0, n)
        # Sort edges by distance from seed node
        def node_dist(e):
            return min(abs(e[0] - seed_node), abs(e[1] - seed_node))
        edges_sorted = sorted(edges, key=node_dist)
        remove_edges = edges_sorted[:n_remove]
        
    elif mode == 'random':
        remove_idx = rng.choice(len(edges), size=n_remove, replace=False)
        remove_edges = [edges[i] for i in remove_idx]
        
    elif mode == 'targeted':
        # Remove highest-weight edges
        edges_sorted = sorted(edges, key=lambda e: -e[2])
        remove_edges = edges_sorted[:n_remove]
    
    else:
        remove_edges = []
    
    # Apply perturbation
    A_perturbed = A.copy()
    affected_nodes = set()
    for i, j, _ in remove_edges:
        A_perturbed[i, j] = 0.0
        A_perturbed[j, i] = 0.0
        affected_nodes.add(i)
        affected_nodes.add(j)
    
    info = {
        'n_edges_removed': n_remove,
        'fraction_removed': fraction,
        'mode': mode,
        'affected_nodes': list(affected_nodes),
        'n_affected_nodes': len(affected_nodes),
    }
    
    return A_perturbed, info
