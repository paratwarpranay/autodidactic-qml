"""Discrete topology hypotheses - graph family samplers.

This module provides samplers for generating graphs from well-known
graph families, independent of any specific matrix draw. This enables
testing topology hypotheses directly.

Supported families:
- Erdos-Renyi (ER): Random graphs with edge probability p
- Barabasi-Albert (BA): Scale-free preferential attachment
- Watts-Strogatz (WS): Small-world with rewiring
- Ring lattice: Regular ring with k neighbors
- Complete: Fully connected
- Block model: Community structure with inter/intra probabilities

Scientific motivation: Different graph topologies may confer different
learning capacities to autodidactic systems. By sampling from graph
families directly, we can test whether topology alone (independent of
the matrix model dynamics) affects kernel expressivity and phase structure.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal
from enum import Enum

from .entanglement_graph import Graph


class GraphFamily(Enum):
    """Enumeration of supported graph families."""
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert"
    WATTS_STROGATZ = "watts_strogatz"
    RING_LATTICE = "ring_lattice"
    COMPLETE = "complete"
    BLOCK_MODEL = "block_model"
    RANDOM_REGULAR = "random_regular"


def _erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Erdos-Renyi random graph G(n, p)."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.uniform(0.5, 1.0)  # Random weight
                A[i, j] = A[j, i] = w
    return A


def _barabasi_albert(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Barabasi-Albert scale-free graph.

    Starts with m+1 connected nodes, then adds n-(m+1) nodes,
    each connecting to m existing nodes with probability proportional to degree.
    """
    A = np.zeros((n, n))

    # Start with m+1 fully connected nodes
    m0 = min(m + 1, n)
    for i in range(m0):
        for j in range(i + 1, m0):
            w = rng.uniform(0.5, 1.0)
            A[i, j] = A[j, i] = w

    # Add remaining nodes
    for new_node in range(m0, n):
        degrees = np.sum(A[:new_node, :new_node], axis=1)
        total_degree = np.sum(degrees)

        if total_degree > 0:
            probs = degrees / total_degree
        else:
            probs = np.ones(new_node) / new_node

        # Choose m targets (without replacement)
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)

        for t in targets:
            w = rng.uniform(0.5, 1.0)
            A[new_node, t] = A[t, new_node] = w

    return A


def _watts_strogatz(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Watts-Strogatz small-world graph.

    Starts with ring lattice of k neighbors, then rewires edges with probability p.
    """
    A = np.zeros((n, n))

    # Create ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            w = rng.uniform(0.5, 1.0)
            A[i, neighbor] = A[neighbor, i] = w

    # Rewire edges
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                neighbor = (i + j) % n
                if A[i, neighbor] > 0:
                    # Remove old edge
                    A[i, neighbor] = A[neighbor, i] = 0

                    # Add new random edge (avoiding self-loops and duplicates)
                    candidates = [c for c in range(n) if c != i and A[i, c] == 0]
                    if candidates:
                        new_neighbor = rng.choice(candidates)
                        w = rng.uniform(0.5, 1.0)
                        A[i, new_neighbor] = A[new_neighbor, i] = w

    return A


def _ring_lattice(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate regular ring lattice with k neighbors on each side."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k + 1):
            neighbor = (i + j) % n
            w = rng.uniform(0.5, 1.0)
            A[i, neighbor] = A[neighbor, i] = w
    return A


def _complete(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate complete graph (all nodes connected)."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = rng.uniform(0.5, 1.0)
            A[i, j] = A[j, i] = w
    return A


def _block_model(
    n: int,
    n_blocks: int,
    p_intra: float,
    p_inter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stochastic block model with community structure.

    Args:
        n: Total number of nodes
        n_blocks: Number of communities
        p_intra: Edge probability within communities
        p_inter: Edge probability between communities
    """
    A = np.zeros((n, n))
    block_size = n // n_blocks

    for i in range(n):
        block_i = i // block_size
        for j in range(i + 1, n):
            block_j = j // block_size
            p = p_intra if block_i == block_j else p_inter
            if rng.random() < p:
                w = rng.uniform(0.5, 1.0)
                A[i, j] = A[j, i] = w

    return A


def _random_regular(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random d-regular graph (each node has exactly d neighbors).

    Uses a simple pairing algorithm with rejection sampling.
    """
    max_attempts = 1000

    for _ in range(max_attempts):
        A = np.zeros((n, n))

        # Create d "stubs" per node
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)

        rng.shuffle(stubs)

        # Pair stubs
        valid = True
        for k in range(0, len(stubs), 2):
            if k + 1 >= len(stubs):
                break
            i, j = stubs[k], stubs[k + 1]
            if i == j or A[i, j] > 0:
                valid = False
                break
            w = rng.uniform(0.5, 1.0)
            A[i, j] = A[j, i] = w

        if valid:
            return A

    # Fallback to approximate
    return _ring_lattice(n, d // 2, rng)


def sample_graph_family(
    family: GraphFamily,
    n: int,
    params: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> Tuple[GraphFamily, Dict, Graph]:
    """Sample a graph from a specified family.

    Args:
        family: Which graph family to sample from
        n: Number of nodes
        params: Family-specific parameters
        seed: Random seed for reproducibility

    Returns:
        Tuple of (family, params_used, Graph)
    """
    rng = np.random.default_rng(seed)
    params = params or {}

    if family == GraphFamily.ERDOS_RENYI:
        p = params.get("p", 0.3)
        A = _erdos_renyi(n, p, rng)
        params_used = {"p": p}

    elif family == GraphFamily.BARABASI_ALBERT:
        m = params.get("m", 2)
        A = _barabasi_albert(n, m, rng)
        params_used = {"m": m}

    elif family == GraphFamily.WATTS_STROGATZ:
        k = params.get("k", 4)
        p = params.get("p", 0.3)
        A = _watts_strogatz(n, k, p, rng)
        params_used = {"k": k, "p": p}

    elif family == GraphFamily.RING_LATTICE:
        k = params.get("k", 2)
        A = _ring_lattice(n, k, rng)
        params_used = {"k": k}

    elif family == GraphFamily.COMPLETE:
        A = _complete(n, rng)
        params_used = {}

    elif family == GraphFamily.BLOCK_MODEL:
        n_blocks = params.get("n_blocks", 2)
        p_intra = params.get("p_intra", 0.7)
        p_inter = params.get("p_inter", 0.1)
        A = _block_model(n, n_blocks, p_intra, p_inter, rng)
        params_used = {"n_blocks": n_blocks, "p_intra": p_intra, "p_inter": p_inter}

    elif family == GraphFamily.RANDOM_REGULAR:
        d = params.get("d", 3)
        A = _random_regular(n, d, rng)
        params_used = {"d": d}

    else:
        raise ValueError(f"Unknown graph family: {family}")

    # Normalize
    if np.max(A) > 0:
        A = A / np.max(A)

    metadata = {
        "family": family.value,
        "params": params_used,
        "n": n,
        "seed": seed,
        "source": "sample_graph_family",
    }

    return (family, params_used, Graph(adjacency=A, metadata=metadata))


def all_families() -> list:
    """Return list of all graph families."""
    return list(GraphFamily)


def sample_all_families(
    n: int,
    seed: int = 0,
) -> Dict[str, Graph]:
    """Sample one graph from each family with default parameters.

    Useful for comparative experiments across topology hypotheses.
    """
    results = {}
    for family in GraphFamily:
        _, _, G = sample_graph_family(family, n, seed=seed)
        results[family.value] = G
    return results
