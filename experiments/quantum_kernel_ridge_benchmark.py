from __future__ import annotations
import argparse
import csv
import numpy as np

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import QuarticAction, CubicAction
from matrix_models.sampler_langevin import LangevinSampler
from matrix_models.gauge_symmetries import invariants as gauge_invariants

from quantum_kernels.quantum_law_encoder import VariationalLawEncoder

def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X2 = np.sum(X*X, axis=1, keepdims=True)
    Y2 = np.sum(Y*Y, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.exp(-D2 / (2.0 * sigma * sigma + 1e-12))

def median_heuristic_sigma(X: np.ndarray, max_pairs: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        return 1.0
    idx = rng.integers(0, n, size=(min(max_pairs, n*(n-1)//2), 2))
    diffs = X[idx[:,0]] - X[idx[:,1]]
    d = np.sqrt(np.sum(diffs*diffs, axis=1) + 1e-12)
    med = float(np.median(d))
    return med if med > 1e-8 else 1.0

def krr_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = K.shape[0]
    A = K + lam * np.eye(n)
    return np.linalg.solve(A, y)

def krr_predict(K_test_train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return K_test_train @ alpha

def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    yy = np.outer(y, y)
    num = float(np.sum(K * yy))
    den = float(np.linalg.norm(K, ord="fro") * np.linalg.norm(yy, ord="fro") + 1e-12)
    return num / den

def margin_stats(y: np.ndarray, scores: np.ndarray) -> dict:
    signed = y * scores
    return {
        "margin_mean": float(np.mean(signed)),
        "margin_min": float(np.min(signed)),
        "margin_std": float(np.std(signed)),
    }

def parse_args():
    p = argparse.ArgumentParser(description="Quantum kernel ridge regression benchmark (with exact RBF baseline).")
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-test", type=int, default=80)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)

    # dataset: two couplings/classes
    p.add_argument("--g0", type=float, default=0.06, help="class -1 coupling")
    p.add_argument("--g1", type=float, default=0.18, help="class +1 coupling")
    p.add_argument("--use-quartic", action="store_true")

    # quantum kernel
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.0, help="depolarizing-style noise strength in [0,1]")
    p.add_argument("--lam", type=float, default=1e-3)

    # output
    p.add_argument("--out", type=str, default="experiments/outputs/krr_benchmark.csv")
    return p.parse_args()

def sample_matrix(dim: int, steps: int, seed: int, g: float, use_quartic: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ens = HermitianEnsemble(dim=dim, seed=seed)
    M0 = ens.sample(rng=rng)
    if use_quartic:
        act = QuarticAction(a=1.0, b=float(0.05 + 0.35 * abs(g)))
    else:
        act = CubicAction(g=float(g))
    sampler = LangevinSampler(action=act, seed=seed)
    M = sampler.run(M0, steps=steps, rng=rng)
    return M

def featurize(M: np.ndarray) -> np.ndarray:
    # robust, gauge-invariant-ish features: trace moments + spectrum summary + flattened upper triangle
    inv = gauge_invariants(M, max_power=4)
    v_inv = np.array([inv[f"tr_M{k}"] for k in range(1, 5)] + [inv["fro_norm"]], dtype=float)
    # add compact raw info for kernels
    iu = np.triu_indices(M.shape[0])
    v_raw = M[iu].astype(float)
    v = np.concatenate([v_inv, v_raw], axis=0)
    v = v / (np.std(v) + 1e-12)
    return v

def quantum_kernel_matrix(X: np.ndarray, encoder: VariationalLawEncoder, noise: float) -> np.ndarray:
    n = X.shape[0]
    K = np.zeros((n, n), dtype=float)
    d = 2 ** encoder.n_qubits
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i+1, n):
            f = float(encoder.kernel(X[i], X[j]))
            # depolarizing-ish noise: fidelity pulled toward 1/d
            f_noisy = (1.0 - noise) * f + noise * (1.0 / d)
            K[i, j] = f_noisy
            K[j, i] = f_noisy
    return K

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Build dataset
    def build_split(n, g, label, seed_offset):
        X = []
        y = []
        for i in range(n):
            M = sample_matrix(args.dim, args.steps, args.seed + seed_offset + i, g, args.use_quartic)
            X.append(featurize(M))
            y.append(label)
        return np.stack(X, axis=0), np.array(y, dtype=float)

    Xtr0, ytr0 = build_split(args.n_train//2, args.g0, -1.0, 10_000)
    Xtr1, ytr1 = build_split(args.n_train - args.n_train//2, args.g1, +1.0, 20_000)
    Xte0, yte0 = build_split(args.n_test//2, args.g0, -1.0, 30_000)
    Xte1, yte1 = build_split(args.n_test - args.n_test//2, args.g1, +1.0, 40_000)

    Xtr = np.concatenate([Xtr0, Xtr1], axis=0)
    ytr = np.concatenate([ytr0, ytr1], axis=0)
    Xte = np.concatenate([Xte0, Xte1], axis=0)
    yte = np.concatenate([yte0, yte1], axis=0)

    # Shuffle
    perm = rng.permutation(Xtr.shape[0])
    Xtr, ytr = Xtr[perm], ytr[perm]
    perm = rng.permutation(Xte.shape[0])
    Xte, yte = Xte[perm], yte[perm]

    # Quantum kernel
    encoder = VariationalLawEncoder(n_qubits=args.qubits, n_layers=args.layers, seed=args.seed)
    Ktr_q = quantum_kernel_matrix(Xtr, encoder, noise=args.noise)
    Kte_q = np.zeros((Xte.shape[0], Xtr.shape[0]), dtype=float)
    d = 2 ** args.qubits
    for i in range(Xte.shape[0]):
        for j in range(Xtr.shape[0]):
            f = float(encoder.kernel(Xte[i], Xtr[j]))
            Kte_q[i, j] = (1.0 - args.noise) * f + args.noise * (1.0 / d)

    alpha_q = krr_fit(Ktr_q, ytr, lam=args.lam)
    scores_q = krr_predict(Kte_q, alpha_q)
    pred_q = np.sign(scores_q)
    pred_q[pred_q == 0] = 1.0
    acc_q = float(np.mean(pred_q == yte))
    align_q = kernel_alignment(Ktr_q, ytr)
    marg_q = margin_stats(yte, scores_q)

    # Exact RBF baseline
    sigma = median_heuristic_sigma(Xtr, seed=args.seed)
    Ktr_rbf = rbf_kernel(Xtr, Xtr, sigma=sigma)
    Kte_rbf = rbf_kernel(Xte, Xtr, sigma=sigma)
    alpha_r = krr_fit(Ktr_rbf, ytr, lam=args.lam)
    scores_r = krr_predict(Kte_rbf, alpha_r)
    pred_r = np.sign(scores_r); pred_r[pred_r == 0] = 1.0
    acc_r = float(np.mean(pred_r == yte))
    align_r = kernel_alignment(Ktr_rbf, ytr)
    marg_r = margin_stats(yte, scores_r)

    row = {
        "dim": args.dim,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "g0": args.g0,
        "g1": args.g1,
        "use_quartic": float(args.use_quartic),
        "qubits": args.qubits,
        "layers": args.layers,
        "noise": args.noise,
        "lam": args.lam,
        "acc_quantum": acc_q,
        "align_quantum": align_q,
        **{f"q_{k}": v for k, v in marg_q.items()},
        "acc_rbf": acc_r,
        "align_rbf": align_r,
        "sigma_rbf": sigma,
        **{f"rbf_{k}": v for k, v in marg_r.items()},
    }

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("\n=== Kernel Ridge Regression Benchmark ===")
    print("Quantum:", {k: row[k] for k in ["acc_quantum", "align_quantum", "q_margin_mean", "q_margin_min"]})
    print("RBF   :", {k: row[k] for k in ["acc_rbf", "align_rbf", "rbf_margin_mean", "rbf_margin_min", "sigma_rbf"]})
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
