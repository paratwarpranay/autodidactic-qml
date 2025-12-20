from __future__ import annotations
import argparse
import itertools
import subprocess
import sys
import os

def parse_args():
    p = argparse.ArgumentParser(description="Sweep quantum kernel settings and append to benchmark CSV.")
    p.add_argument("--qubits", type=str, default="4,6,8")
    p.add_argument("--layers", type=str, default="2,3,4")
    p.add_argument("--noise", type=str, default="0.0,0.05,0.10")
    p.add_argument("--out", type=str, default="experiments/outputs/krr_benchmark.csv")
    return p.parse_args()

def to_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def main():
    args = parse_args()
    q_list = [int(x) for x in args.qubits.split(",") if x.strip()]
    l_list = [int(x) for x in args.layers.split(",") if x.strip()]
    n_list = to_list(args.noise)

    for q, l, n in itertools.product(q_list, l_list, n_list):
        cmd = [
            sys.executable, "-m", "experiments.quantum_kernel_ridge_benchmark",
            "--qubits", str(q),
            "--layers", str(l),
            "--noise", str(n),
            "--out", args.out,
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

    print("Sweep complete:", args.out)

if __name__ == "__main__":
    main()
