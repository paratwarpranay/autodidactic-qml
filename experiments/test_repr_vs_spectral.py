#!/usr/bin/env python
"""Test representation vs spectral constraints for functional basin creation.

This is the DECISIVE experiment to determine if Gram/CKA representation 
constraints create functional basins where spectral invariants fail.
"""

import numpy as np
import torch
from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from correspondence_maps.matrix_to_rnn import MatrixToCyclicRNN
from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from autodidactic_protocols.mutual_information_drive import MutualInformationLearner
from analysis.law_stability_tracker import LawStabilityTracker
from ucip_detection import (
    InvariantConstrainedCI,
    run_repr_vs_spectral_comparison,
    compute_model_invariants,
)


class WrappedRNN(torch.nn.Module):
    """Wrapper for cyclic RNN with fc adapters for CI metric compatibility."""
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
        return out, stats


def format_inv(inv: dict) -> str:
    """Compact invariant format."""
    keys = ["fro_norm", "spec_radius", "tr_M2"]
    return " | ".join(f"{k}={inv.get(k, 0):.3f}" for k in keys if k in inv)


def main():
    # Configuration
    dim = 12
    net_width = 64
    epochs = 6
    seed = 0
    
    print("=" * 70)
    print("REPRESENTATION vs SPECTRAL CONSTRAINTS")
    print("=" * 70)
    
    # === Build and train model ===
    print("\n[1] MATRIX SAMPLING & RNN CONSTRUCTION")
    rng = np.random.default_rng(seed)
    ens = HermitianEnsemble(dim=dim, scale=1.0, seed=seed)
    M0 = ens.sample(rng=rng)
    action = CubicAction(g=0.1)
    sampler = LangevinSampler(action=action, dt=1e-3, temperature=1e-2, seed=seed)
    M = sampler.run(M0, steps=800, rng=rng)
    
    mapper = MatrixToCyclicRNN(hidden_size=net_width, input_size=net_width, seed=seed)
    net = mapper.build(M)
    model = WrappedRNN(net)
    
    print(f"    Matrix dim={dim}, RNN hidden_size={net_width}")
    print(f"    Initial: {format_inv(compute_model_invariants(model))}")
    
    print("\n[2] AUTODIDACTIC TRAINING")
    learner = SelfConsistencyLearner(lr=1e-3, steps_per_update=120, batch_size=128)
    mi_learner = MutualInformationLearner(lr=1e-3, steps=120, mi_weight=0.15)
    tracker = LawStabilityTracker(window=25)
    
    for epoch in range(epochs):
        res = learner.update(model)
        res2 = mi_learner.update(model)
        stats = tracker.update(res["loss"])
        print(f"    Epoch {epoch}: loss={stats['loss']:.4f} ma={stats['loss_ma']:.4f} mi={res2['mi_proxy']:.3f}")
    
    print(f"    Trained:  {format_inv(compute_model_invariants(model))}")
    
    # === Run the decisive 2×2 comparison ===
    print("\n[3] RUNNING 2×2 COMPARISON")
    print("    Constraint types: spectral (shape) vs repr (Gram/CKA)")
    print("    Recovery types: mixed vs projected")
    
    results = run_repr_vs_spectral_comparison(
        model=model,
        InvariantConstrainedCI_class=InvariantConstrainedCI,
        recovery_steps=10,
        recovery_seed=2025,
        invariant_weight=0.1,
        verbose=True,
    )
    
    # === Analysis ===
    print("\n" + "=" * 70)
    print("DECISIVE RESULT")
    print("=" * 70)
    
    spectral_ci = results["spectral"]["ci_score"]
    repr_ci = results["repr"]["ci_score"]
    
    print(f"\n  Spectral CI (shape):           {spectral_ci:.4f}")
    print(f"  Representation CI (Gram/CKA):  {repr_ci:.4f}")
    print(f"  Improvement:                   {repr_ci - spectral_ci:+.4f}")
    
    if "final_h_cosine" in results["repr"]:
        print(f"\n  Final h_cosine:                {results['repr']['final_h_cosine']:.4f}")
    if "final_cka" in results["repr"]:
        print(f"  Final CKA:                     {results['repr']['final_cka']:.4f}")
    
    print("\n" + "-" * 70)
    
    if repr_ci - spectral_ci > 0.05:
        print("  ✓ BREAKTHROUGH: Gram/CKA creates functional basins!")
        print("    Representation constraints are the missing ingredient.")
        print("    → Publish: 'Functional basins require representation constraints'")
    elif repr_ci > 0.1 and spectral_ci < 0.05:
        print("  ✓ SIGNIFICANT: Repr constraint works, spectral doesn't")
        print("    → Representation geometry matters for function recovery")
    elif repr_ci - spectral_ci > 0.01:
        print("  ~ MODEST: Repr constraint helps somewhat")
        print("    → May need longer recovery or different λ")
    else:
        print("  ? INCONCLUSIVE: Both approaches show minimal recovery")
        print("    Next steps:")
        print("    - Try longer recovery_steps (20-50)")
        print("    - Try different invariant_weight (0.3, 0.5)")
        print("    - May need Jacobian-level constraints")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
