from __future__ import annotations
import numpy as np
from typing import Dict

def running_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x))}
