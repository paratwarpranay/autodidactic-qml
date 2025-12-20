from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch.nn as nn

from .invariant_metrics import InvariantComparator
from .spectral_diagnostics import SpectralDiagnostics
from .phase_transition_metrics import PhaseTransitionMetrics
from correspondence_maps.emergent_geometry import SpectralGeometry


def _safe_rel(a: float, b: float, eps: float = 1e-12) -> float:
    return float(abs(a - b) / (abs(a) + eps))


def rnn_stability_metrics(model: Optional[nn.Module]) -> Dict[str, float]:
    """Lightweight stability observables for cyclic RNNs.

    If we can see a recurrent matrix W_hh, we report its spectral radius (max |eig|),
    plus Frobenius norm. These correlate with stability/chaos in many RNN settings.
    """
    if model is None:
        return {}

    core = getattr(model, "core", model)
    W_hh = getattr(core, "W_hh", None)
    if W_hh is None:
        return {}

    W = np.asarray(W_hh.detach().cpu().numpy(), dtype=float)
    try:
        eig = np.linalg.eigvals(W)
        spec_radius = float(np.max(np.abs(eig)))
    except np.linalg.LinAlgError:
        spec_radius = float("nan")

    return {
        "rnn_spec_radius": spec_radius,
        "rnn_fro_norm": float(np.linalg.norm(W, ord="fro")),
        "rnn_mean_abs": float(np.mean(np.abs(W))),
    }


@dataclass(frozen=True)
class CorrespondenceDiagnostics:
    max_power: int = 6
    laplacian_k: int = 12
    laplacian_tau: float = 1.0
    geom_bandwidth: float = 1.0

    def compare(
        self,
        M: np.ndarray,
        M_hat: np.ndarray,
        model: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compare invariants + spectra + emergent-geometry observables.

        Returns a flat dict with keys used by tests and experiment scripts.
        """
        M = np.asarray(M, dtype=float)
        M_hat = np.asarray(M_hat, dtype=float)

        out: Dict[str, float] = {}

        # Gauge-invariant summaries
        inv = InvariantComparator(max_power=self.max_power).compare(M, M_hat)
        out.update(inv)

        # Eigenvalue summary
        sd = SpectralDiagnostics()
        out.update(sd.summarize(M, prefix="M_"))
        out.update(sd.summarize(M_hat, prefix="Mhat_"))

        # Emergent geometry: Laplacian spectrum in eigenvalue-space
        geom = SpectralGeometry(bandwidth=float(self.geom_bandwidth))
        spec_M = geom.laplacian_spectrum(M)
        spec_H = geom.laplacian_spectrum(M_hat)

        k = int(self.laplacian_k)
        spec_Mk = spec_M[:k] if spec_M.size >= k else spec_M
        spec_Hk = spec_H[:k] if spec_H.size >= k else spec_H

        # pad to same length for L2 comparisons
        L = max(spec_Mk.size, spec_Hk.size)
        if spec_Mk.size < L:
            spec_Mk = np.pad(spec_Mk, (0, L - spec_Mk.size), mode="edge")
        if spec_Hk.size < L:
            spec_Hk = np.pad(spec_Hk, (0, L - spec_Hk.size), mode="edge")

        l2 = float(np.linalg.norm(spec_Hk - spec_Mk))
        denom = float(np.linalg.norm(spec_Mk) + 1e-12)
        out["lap_spec_l2"] = l2
        out["lap_spec_l2_rel"] = float(l2 / denom)

        # scalar Laplacian diagnostics
        gap_M = float(spec_M[1] - spec_M[0]) if spec_M.size > 1 else 0.0
        gap_H = float(spec_H[1] - spec_H[0]) if spec_H.size > 1 else 0.0
        out["lap_gap_M"] = gap_M
        out["lap_gap_Mhat"] = gap_H
        out["lap_gap_rel"] = _safe_rel(gap_M, gap_H)

        eff_M = float(geom.effective_dimension(spec_M, tau=float(self.laplacian_tau)))
        eff_H = float(geom.effective_dimension(spec_H, tau=float(self.laplacian_tau)))
        out["lap_effdim_M"] = eff_M
        out["lap_effdim_Mhat"] = eff_H
        out["lap_effdim_rel"] = _safe_rel(eff_M, eff_H)

        # RNN stability (if present)
        out.update(rnn_stability_metrics(model))

        # Training metrics (optional)
        if train_metrics:
            for k, v in train_metrics.items():
                # keep names stable across scripts
                out[f"train_{k}"] = float(v)

        return out


def phase_scan_metrics(xs: np.ndarray) -> Dict[str, float]:
    """Heuristics for 1D scans (kept for back-compat)."""
    x = np.asarray(xs, dtype=float).reshape(-1)
    return PhaseTransitionMetrics().metric(x)
