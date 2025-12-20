from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from .matrix_to_rnn import MatrixToCyclicRNN
from .neural_to_matrix import rnn_to_matrix


@dataclass
class RoundTripMapper:
    """End-to-end M -> model -> M_hat pipeline.

    Now topology-aware: accepts optional Graph object and logs
    topology-conditioned invariants during round-trip.
    """
    width: int = 64
    seed: int = 0

    def build_model(
        self,
        M: np.ndarray,
        graph: Optional[Any] = None,
    ) -> Tuple[nn.Module, nn.Module]:
        """Build model from matrix, optionally conditioned on graph topology.

        Args:
            M: Source matrix
            graph: Optional Graph object from topology module

        Returns:
            (core, wrapped_model) tuple
        """
        mapper = MatrixToCyclicRNN(
            hidden_size=self.width,
            input_size=self.width,
            seed=self.seed,
        )
        core = mapper.build(M)

        # Store graph reference for topology-aware metrics
        graph_ref = graph

        class Wrapped(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
                self.input_size = core.input_size
                self.hidden_size = core.hidden_size
                self.fc1 = nn.Linear(self.input_size, self.hidden_size)
                self.fc2 = nn.Linear(self.hidden_size, self.input_size)
                self.graph = graph_ref  # Store topology reference

            def forward(self, x: torch.Tensor, steps: int = 4):
                xh = self.fc1(x)
                _, stats = self.core(xh, steps=steps)
                out = self.fc2(stats["h_final"])
                return out

        model = Wrapped(core)
        return core, model

    def extract_matrix(self, core: nn.Module) -> np.ndarray:
        return rnn_to_matrix(core)

    def round_trip(
        self,
        M: np.ndarray,
        graph: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Full round-trip: M -> model -> M_hat with topology metrics.

        Args:
            M: Source matrix
            graph: Optional Graph object for topology-conditioned invariants

        Returns:
            Dictionary with M_hat, invariant comparisons, and topology metrics
        """
        core, model = self.build_model(M, graph=graph)
        M_hat = self.extract_matrix(core)

        # Compute invariant comparison
        from analysis import InvariantComparator
        comparator = InvariantComparator(max_power=4)
        invariant_comparison = comparator.compare(M, M_hat)

        result = {
            "M_hat": M_hat,
            "invariant_comparison": invariant_comparison,
        }

        # Add topology-conditioned metrics if graph provided
        if graph is not None:
            result["topology_metrics"] = {
                "n_edges": graph.n_edges() if hasattr(graph, 'n_edges') else None,
                "algebraic_connectivity": graph.algebraic_connectivity() if hasattr(graph, 'algebraic_connectivity') else None,
                "spectral_gap": graph.spectral_gap() if hasattr(graph, 'spectral_gap') else None,
            }

        return result
