from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

@dataclass
class InterruptionChannel:
    """Apply structured interruptions during training.

    Modes:
    - freeze: temporarily disables gradients for a random subset of parameters
    - reset: partially resets parameters to noise

    The purpose is to test *recoverability* dynamics: does training adapt so that the
    system becomes easier to recover after interruptions?
    """
    freeze_frac: float = 0.10
    freeze_duration: int = 1
    reset_frac: float = 0.05
    reset_std: float = 0.05
    p_freeze: float = 0.50
    seed: int = 0

    # internal state
    _frozen: List[torch.nn.Parameter] = None  # type: ignore
    _freeze_left: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._frozen = []
        self._freeze_left = 0

    def _choose_params(self, model: nn.Module, frac: float) -> List[torch.nn.Parameter]:
        params = [p for p in model.parameters()]
        if not params:
            return []
        k = max(1, int(round(frac * len(params))))
        idx = self.rng.choice(len(params), size=k, replace=False)
        return [params[i] for i in idx]

    def step(self, model: nn.Module) -> Dict[str, float]:
        # Handle ongoing freeze countdown
        if self._freeze_left > 0:
            self._freeze_left -= 1
            if self._freeze_left == 0:
                for p in self._frozen:
                    p.requires_grad_(True)
                self._frozen = []
            return {"interruption": 0.0, "freeze_active": float(self._freeze_left > 0)}

        # Decide new interruption
        if self.rng.random() < self.p_freeze:
            chosen = self._choose_params(model, self.freeze_frac)
            for p in chosen:
                p.requires_grad_(False)
            self._frozen = chosen
            self._freeze_left = max(1, int(self.freeze_duration))
            return {"interruption": 1.0, "did_freeze": 1.0, "did_reset": 0.0, "freeze_active": 1.0}
        else:
            chosen = self._choose_params(model, self.reset_frac)
            with torch.no_grad():
                for p in chosen:
                    p.add_(self.reset_std * torch.randn_like(p))
            return {"interruption": 1.0, "did_freeze": 0.0, "did_reset": 1.0, "freeze_active": 0.0}
