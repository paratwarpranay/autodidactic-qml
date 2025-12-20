from __future__ import annotations
import argparse
import numpy as np

from quantum_kernels.quantum_law_encoder import VariationalLawEncoder
from quantum_kernels.kernel_expressivity import QuantumKernel, random_fourier_features_kernel, kernel_alignment

def make_synthetic(n: int = 60, d: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # non-linear decision boundary
    y = np.sign(np.sin(X[:,0]) + 0.5*np.cos(X[:,1]) + 0.25*X[:,2])
    y[y==0] = 1
    return X, y.astype(float)

def parse_args():
    p = argparse.ArgumentParser(description="Compare quantum-kernel vs classical RFF kernel alignment.")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--d", type=int, default=12)
    p.add_argument("--qubits", type=int, default=5)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rff-features", type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    X, y = make_synthetic(args.n, args.d, args.seed)

    enc = VariationalLawEncoder(n_qubits=args.qubits, n_layers=args.layers, seed=args.seed)
    qk = QuantumKernel(encode=enc.encode)
    Kq = qk.gram(X)
    aq = kernel_alignment(Kq, y)

    Kc = random_fourier_features_kernel(X, gamma=1.0, n_features=args.rff_features, seed=args.seed)
    ac = kernel_alignment(Kc, y)

    print(f"Quantum kernel alignment:   {aq:.4f}")
    print(f"Classical RFF alignment:    {ac:.4f}")
    print("Interpretation: higher alignment => kernel better matches label geometry (proxy for learnability).")

if __name__ == "__main__":
    main()
