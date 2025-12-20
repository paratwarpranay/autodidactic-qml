from __future__ import annotations
import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it to use this script.") from exc
    return plt

def parse_args():
    p = argparse.ArgumentParser(description="Plot coupling sweep order parameter with detected breakpoints.")
    p.add_argument("--csv", type=str, default="experiments/outputs/coupling_sweep.csv")
    p.add_argument("--order", type=str, default="lap_effdim_Mhat")
    p.add_argument("--stage", type=str, default="post")
    p.add_argument("--report", type=str, default="experiments/outputs/coupling_sweep_report.json")
    p.add_argument("--out", type=str, default="experiments/outputs/coupling_sweep.png")
    return p.parse_args()

def load_csv(path: str):
    import csv
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    args = parse_args()
    plt = _require_matplotlib()
    rows = load_csv(args.csv)
    rows = [r for r in rows if r.get("stage") == args.stage]

    # group by g and compute mean/std across repeats
    g_vals = sorted(set(float(r["g"]) for r in rows))
    y_mean = []
    y_std = []
    for g in g_vals:
        vals = []
        for r in rows:
            if float(r["g"]) != g:
                continue
            key = args.order
            if key == "mi_proxy":
                key = "train_mi_proxy"
            if key in r and r[key] not in ("", None):
                vals.append(float(r[key]))
        y_mean.append(float(np.mean(vals)) if vals else float("nan"))
        y_std.append(float(np.std(vals)) if vals else float("nan"))

    g = np.array(g_vals, dtype=float)
    y = np.array(y_mean, dtype=float)
    s = np.array(y_std, dtype=float)

    plt.figure()
    plt.errorbar(g, y, yerr=s, fmt="o-", capsize=3)
    plt.xlabel("coupling g")
    plt.ylabel(args.order)
    plt.title(f"Phase scan: {args.order} ({args.stage})")

    # overlay breakpoints from report json if present
    if os.path.exists(args.report):
        rep = json.load(open(args.report, "r", encoding="utf-8"))
        best = rep.get("best", {})
        for bx in best.get("breaks_x", []):
            plt.axvline(float(bx), linestyle="--")

        # bootstrap CI intervals
        boot = rep.get("bootstrap", {})
        for ci in boot.get("ci", []):
            if np.isfinite(ci.get("median", float("nan"))):
                plt.axvspan(float(ci["p05"]), float(ci["p95"]), alpha=0.15)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=160)
    print("Saved plot:", args.out)

if __name__ == "__main__":
    main()
