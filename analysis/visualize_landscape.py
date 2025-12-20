from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class LandscapeVisualizer:
    """Simple plotting for trajectories in observable space."""

    def plot_trajectory(self, obs: Dict[str, np.ndarray], title: str = "Trajectory") -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError("matplotlib is required for plotting. Install it to use this method.") from exc
        keys = list(obs.keys())
        if len(keys) < 2:
            raise ValueError("Need at least two observables to plot a 2D trajectory.")
        xk, yk = keys[0], keys[1]
        x, y = obs[xk], obs[yk]
        plt.figure()
        plt.plot(x, y, marker="o", linewidth=1)
        plt.xlabel(xk)
        plt.ylabel(yk)
        plt.title(title)
        plt.tight_layout()
        plt.show()
