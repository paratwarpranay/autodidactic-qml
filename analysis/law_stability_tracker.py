from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class LawStabilityTracker:
    """Track stability of a learned dynamics across training."""
    window: int = 20

    def __post_init__(self):
        self.history: List[float] = []

    def update(self, loss: float) -> Dict[str, float]:
        self.history.append(float(loss))
        w = self.history[-self.window:]
        return {"loss": float(loss), "loss_ma": float(np.mean(w)), "loss_std": float(np.std(w))}
