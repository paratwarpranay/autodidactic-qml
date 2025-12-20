from __future__ import annotations
import argparse
import numpy as np
from tqdm import tqdm

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from correspondence_maps.emergent_geometry import SpectralGeometry
from analysis.spectral_diagnostics import SpectralDiagnostics

def parse_args():
    p = argparse.ArgumentParser(description="Compute emergent geometry proxies from matrix dynamics.")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--g", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ens = HermitianEnsemble(dim=args.dim, seed=args.seed)
    M0 = ens.sample(rng=rng)

    act = CubicAction(g=args.g)
    sampler = LangevinSampler(action=act, seed=args.seed)
    M = sampler.run(M0, steps=args.steps, rng=rng)

    geom = SpectralGeometry(bandwidth=1.0)
    summ = geom.summarize(M)
    spec = SpectralDiagnostics().summarize(M)
    print("Spectral diag:", {k: round(v,4) for k,v in spec.items()})
    print("Emergent geometry:", {k: round(v,4) for k,v in summ.items()})

if __name__ == "__main__":
    main()
