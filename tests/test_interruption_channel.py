import numpy as np
import torch
import torch.nn as nn
from ucip_detection.interruption_channel import InterruptionChannel

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))

def test_interruption_channel_runs():
    m = Tiny()
    ch = InterruptionChannel(seed=0, freeze_frac=0.5, reset_frac=0.5, freeze_duration=1, p_freeze=1.0)
    r = ch.step(m)
    assert r.get("did_freeze", 0.0) == 1.0
    # next step should unfreeze or progress countdown
    r2 = ch.step(m)
    assert "freeze_active" in r2
