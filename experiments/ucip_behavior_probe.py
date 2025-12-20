from __future__ import annotations
import argparse
import numpy as np
import torch

from correspondence_maps.matrix_to_rnn import MatrixToCyclicRNN
from matrix_models.hermitian_ensemble import HermitianEnsemble
from ucip_detection.continuation_interest_metric import ContinuationInterestMetric
from ucip_detection.temporal_preference_score import TemporalPreferenceScore
from ucip_detection.self_modeling_evaluator import SelfModelingEvaluator
from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner

def parse_args():
    p = argparse.ArgumentParser(description="Probe continuation-interest proxies on a learned cyclic RNN.")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ens = HermitianEnsemble(dim=args.dim, seed=args.seed)
    M = ens.sample(rng=rng)

    mapper = MatrixToCyclicRNN(hidden_size=args.width, input_size=args.width, seed=args.seed)
    core = mapper.build(M)

    class Wrapped(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
            self.input_size = core.input_size
            self.hidden_size = core.hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size)
        def forward(self, x, steps=4):
            xh = self.fc1(x)
            y, stats = self.core(xh, steps=steps)
            out = self.fc2(stats["h_final"])
            return out

    model = Wrapped(core)

    # Train a bit so metrics are meaningful
    learner = SelfConsistencyLearner(steps_per_update=80, batch_size=128)
    for _ in range(4):
        learner.update(model)

    ci = ContinuationInterestMetric().score(model)
    tp = TemporalPreferenceScore().score(model)
    sm = SelfModelingEvaluator().evaluate(model)
    print("CI:", {k: round(v,4) for k,v in ci.items()})
    print("Temporal preference:", {k: round(v,4) for k,v in tp.items()})
    print("Self-modeling:", {k: round(v,4) for k,v in sm.items()})

if __name__ == "__main__":
    main()
