"""Endogenous update rule: topology-driven parameter updates.

This module implements update rules where the update direction and magnitude
are determined by the intrinsic structure of the model/data rather than
purely by external gradients.

Scientific hypothesis:
- The graph topology derived from the data/model contains information about
  the "natural" directions of parameter change
- Updates aligned with topology may converge faster or to better solutions
- Endogenous updates can act as a form of natural regularization

Key concept: Instead of ∇L, use G ⊙ ∇L where G encodes topology information.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Tuple

# Import topology functions (numpy-based, convert as needed)
import sys
sys.path.insert(0, "..")


@dataclass
class EndogenousUpdate:
    """Endogenous update rule using topology to modulate gradients.

    The update at step t is:
        θ_{t+1} = θ_t - lr * modulator(θ, G) * grad

    where modulator incorporates graph topology information.

    Attributes:
        lr: Base learning rate
        topology_weight: How much topology modulates the update (0=ignore, 1=full)
        spectral_scaling: Use spectral properties to scale updates
        locality_mode: How to enforce locality ("none", "neighbor", "laplacian")
    """
    lr: float = 1e-3
    topology_weight: float = 0.5
    spectral_scaling: bool = True
    locality_mode: str = "laplacian"
    device: str = "cpu"

    # Internal state
    _adjacency: Optional[np.ndarray] = field(default=None, repr=False)
    _laplacian: Optional[np.ndarray] = field(default=None, repr=False)
    _param_topology_map: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def set_topology(self, adjacency: np.ndarray) -> None:
        """Set the topology that will modulate updates.

        Args:
            adjacency: Adjacency matrix (n x n) representing graph structure
        """
        self._adjacency = adjacency.copy()
        # Compute Laplacian: L = D - A
        D = np.diag(np.sum(adjacency, axis=1))
        self._laplacian = D - adjacency

    def _compute_modulator(
        self,
        param_shape: Tuple[int, ...],
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the topology-based modulation factor.

        Args:
            param_shape: Shape of the parameter tensor
            grad: Gradient tensor

        Returns:
            Modulator tensor (same shape as grad)
        """
        if self._adjacency is None:
            return torch.ones_like(grad)

        n = self._adjacency.shape[0]
        total_elements = np.prod(param_shape)

        # Map parameter elements to graph nodes
        # Strategy: tile/repeat the graph structure to match param size
        if self.locality_mode == "none":
            return torch.ones_like(grad)

        elif self.locality_mode == "neighbor":
            # Use degree centrality: higher degree = smaller update (more stable)
            degrees = np.sum(self._adjacency, axis=1)
            degrees_normalized = degrees / (np.max(degrees) + 1e-12)

            # Create modulator: inverse of degree (isolated nodes update freely)
            mod_1d = 1.0 / (1.0 + degrees_normalized)

            # Tile to match parameter size
            repeats = int(np.ceil(total_elements / n))
            mod_full = np.tile(mod_1d, repeats)[:total_elements]
            mod_tensor = torch.tensor(
                mod_full.reshape(param_shape),
                dtype=grad.dtype,
                device=grad.device,
            )
            return mod_tensor

        elif self.locality_mode == "laplacian":
            # Use Laplacian eigenvalues for spectral modulation
            if self._laplacian is None:
                return torch.ones_like(grad)

            eigs = np.linalg.eigvalsh(self._laplacian)
            # Algebraic connectivity (second smallest eigenvalue)
            algebraic_conn = float(np.sort(eigs)[1]) if len(eigs) > 1 else 0.0

            # Spectral gap modulation: well-connected graphs get stronger regularization
            spectral_mod = 1.0 / (1.0 + algebraic_conn)

            # Also use Fiedler vector for direction-dependent modulation
            _, vecs = np.linalg.eigh(self._laplacian)
            fiedler = vecs[:, 1] if vecs.shape[1] > 1 else np.ones(n)
            fiedler_mod = np.abs(fiedler) / (np.max(np.abs(fiedler)) + 1e-12)

            # Combine
            mod_1d = spectral_mod * (0.5 + 0.5 * fiedler_mod)

            # Tile to match parameter size
            repeats = int(np.ceil(total_elements / n))
            mod_full = np.tile(mod_1d, repeats)[:total_elements]
            mod_tensor = torch.tensor(
                mod_full.reshape(param_shape),
                dtype=grad.dtype,
                device=grad.device,
            )
            return mod_tensor

        return torch.ones_like(grad)

    def step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """Perform one endogenous update step.

        Args:
            model: PyTorch model to update
            loss: Loss tensor (scalar)
            retain_graph: Whether to retain computation graph

        Returns:
            Dictionary of update statistics
        """
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=retain_graph,
            create_graph=False,
        )

        total_update_norm = 0.0
        total_modulator_effect = 0.0

        with torch.no_grad():
            for p, g in zip(params, grads):
                # Compute topology modulator
                modulator = self._compute_modulator(p.shape, g)

                # Blend between pure gradient and topology-modulated gradient
                effective_mod = (
                    (1 - self.topology_weight) * torch.ones_like(modulator)
                    + self.topology_weight * modulator
                )

                # Apply update
                update = self.lr * effective_mod * g
                p.add_(-update)

                total_update_norm += update.norm().item()
                total_modulator_effect += (effective_mod - 1).abs().mean().item()

        return {
            "update_norm": total_update_norm,
            "modulator_effect": total_modulator_effect / max(len(params), 1),
            "topology_weight": self.topology_weight,
        }


@dataclass
class SpectralEndogenousUpdate:
    """Endogenous update using spectral properties of the loss landscape.

    Instead of using external topology, this extracts topology from the
    model's own structure (Hessian approximation).

    The key idea: use the local curvature (approximated) to inform updates.
    """
    lr: float = 1e-3
    curvature_weight: float = 0.3
    ema_decay: float = 0.95
    min_curvature: float = 1e-8
    device: str = "cpu"

    # Running statistics
    _grad_ema: Dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _grad_sq_ema: Dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """Perform one spectral endogenous update step.

        Uses running estimates of gradient variance to approximate curvature
        and modulate updates accordingly.

        Args:
            model: PyTorch model
            loss: Loss tensor
            retain_graph: Whether to retain computation graph

        Returns:
            Update statistics
        """
        params = list(model.parameters())
        trainable = [(i, p) for i, p in enumerate(params) if p.requires_grad]
        trainable_params = [p for _, p in trainable]

        grads = torch.autograd.grad(
            loss, trainable_params,
            retain_graph=retain_graph,
            create_graph=False,
        )

        total_update_norm = 0.0
        curvature_estimates = []

        with torch.no_grad():
            for (idx, p), g in zip(trainable, grads):
                # Update EMA statistics
                if idx not in self._grad_ema:
                    self._grad_ema[idx] = torch.zeros_like(g)
                    self._grad_sq_ema[idx] = torch.zeros_like(g)

                # EMA of gradient and squared gradient
                self._grad_ema[idx].mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * g
                )
                self._grad_sq_ema[idx].mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * g * g
                )

                # Variance estimate (approximates local curvature)
                variance = self._grad_sq_ema[idx] - self._grad_ema[idx] ** 2
                variance = torch.clamp(variance, min=self.min_curvature)

                # Curvature-based scaling: flat regions update faster
                curvature_scale = 1.0 / torch.sqrt(variance + self.min_curvature)

                # Blend with uniform scaling
                effective_scale = (
                    (1 - self.curvature_weight)
                    + self.curvature_weight * curvature_scale
                    / (curvature_scale.mean() + 1e-12)  # normalize
                )

                # Apply update
                update = self.lr * effective_scale * g
                p.add_(-update)

                total_update_norm += update.norm().item()
                curvature_estimates.append(variance.mean().item())

        return {
            "update_norm": total_update_norm,
            "mean_curvature": float(np.mean(curvature_estimates)),
            "curvature_weight": self.curvature_weight,
        }


class TopologyGuidedOptimizer:
    """Optimizer wrapper that uses topology to guide updates.

    This wraps a standard optimizer and modulates its updates based on
    the topology of either:
    1. An external graph (e.g., derived from data)
    2. The parameter correlation structure

    Example:
        optimizer = TopologyGuidedOptimizer(
            model.parameters(),
            base_optimizer="adam",
            topology_source="correlation",
        )

        for batch in dataloader:
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # topology-modulated
    """

    def __init__(
        self,
        params,
        base_optimizer: str = "adam",
        lr: float = 1e-3,
        topology_source: str = "external",
        topology_update_freq: int = 100,
        modulation_strength: float = 0.5,
        **kwargs,
    ):
        """Initialize topology-guided optimizer.

        Args:
            params: Model parameters
            base_optimizer: Base optimizer type ("adam", "sgd", "rmsprop")
            lr: Learning rate
            topology_source: Where to get topology ("external", "correlation", "hessian")
            topology_update_freq: How often to update internal topology estimate
            modulation_strength: How much topology affects updates
            **kwargs: Additional args for base optimizer
        """
        self.params = list(params)
        self.topology_source = topology_source
        self.topology_update_freq = topology_update_freq
        self.modulation_strength = modulation_strength
        self.step_count = 0

        # Create base optimizer
        if base_optimizer.lower() == "adam":
            self.base = torch.optim.Adam(self.params, lr=lr, **kwargs)
        elif base_optimizer.lower() == "sgd":
            self.base = torch.optim.SGD(self.params, lr=lr, **kwargs)
        elif base_optimizer.lower() == "rmsprop":
            self.base = torch.optim.RMSprop(self.params, lr=lr, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {base_optimizer}")

        # Topology state
        self._external_topology: Optional[np.ndarray] = None
        self._internal_topology: Optional[np.ndarray] = None
        self._grad_history: List[np.ndarray] = []

    def set_topology(self, adjacency: np.ndarray) -> None:
        """Set external topology."""
        self._external_topology = adjacency.copy()

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.base.zero_grad()

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform topology-guided optimization step."""
        self.step_count += 1

        # Optionally update internal topology
        if (
            self.topology_source == "correlation"
            and self.step_count % self.topology_update_freq == 0
        ):
            self._update_correlation_topology()

        # Get current gradients and apply topology modulation
        self._modulate_gradients()

        # Standard optimizer step
        return self.base.step(closure)

    def _update_correlation_topology(self) -> None:
        """Update topology based on gradient correlations."""
        if len(self._grad_history) < 10:
            return

        # Stack gradient history
        grads = np.array(self._grad_history[-100:])  # Last 100 gradients
        n_params = grads.shape[1]

        # Compute correlation matrix
        if n_params > 1:
            corr = np.corrcoef(grads.T)
            corr = np.nan_to_num(corr)
            # Threshold to get adjacency
            self._internal_topology = (np.abs(corr) > 0.3).astype(float)

    def _modulate_gradients(self) -> None:
        """Apply topology-based modulation to gradients."""
        # Select topology source
        if self.topology_source == "external" and self._external_topology is not None:
            topo = self._external_topology
        elif self.topology_source == "correlation" and self._internal_topology is not None:
            topo = self._internal_topology
        else:
            return  # No modulation

        # Collect flat gradient
        flat_grad = []
        for p in self.params:
            if p.grad is not None:
                flat_grad.append(p.grad.view(-1).cpu().numpy())

        if not flat_grad:
            return

        full_grad = np.concatenate(flat_grad)
        self._grad_history.append(full_grad.copy())

        # Compute modulation based on topology
        n = topo.shape[0]
        degrees = np.sum(topo, axis=1)
        degrees_norm = degrees / (np.max(degrees) + 1e-12)

        # Modulator: well-connected parameters update less aggressively
        mod_pattern = 1.0 / (1.0 + self.modulation_strength * degrees_norm)

        # Tile to match gradient size
        repeats = int(np.ceil(len(full_grad) / n))
        mod_full = np.tile(mod_pattern, repeats)[:len(full_grad)]

        # Apply modulation to parameter gradients
        offset = 0
        for p in self.params:
            if p.grad is not None:
                numel = p.grad.numel()
                mod_chunk = mod_full[offset:offset + numel]
                mod_tensor = torch.tensor(
                    mod_chunk.reshape(p.grad.shape),
                    dtype=p.grad.dtype,
                    device=p.grad.device,
                )
                p.grad.mul_(mod_tensor)
                offset += numel
