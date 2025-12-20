from __future__ import annotations
import argparse
import json
import os
import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it to use this script.") from exc
    return plt

def parse_args():
    p = argparse.ArgumentParser(description="Plot probabilistic phase boundary (logistic) with uncertainty band.")
    p.add_argument("--report", type=str, default="experiments/outputs/phase_diagram_report.json")
    p.add_argument("--out", type=str, default="experiments/outputs/phase_diagram_probabilistic.png")
    p.add_argument("--field", type=str, default="logistic_score", choices=["logistic_score","kmeans_score"],
                   help="Which score surface to plot.")
    return p.parse_args()

def main():
    args = parse_args()
    plt = _require_matplotlib()
    rep = json.load(open(args.report, "r", encoding="utf-8"))
    axis2 = rep["axis2"]
    a2 = np.array(rep["axis2_values"], dtype=float)
    g = np.array(rep["g_values"], dtype=float)

    vp = rep.get("vector_phase", None)
    if vp is None:
        raise SystemExit("No vector_phase found. Run phase_diagram_scan with --vector-classifier.")

    if args.field == "logistic_score":
        Z = vp.get("logistic_score_grid", None)
        if Z is None:
            raise SystemExit("No logistic_score_grid. Run with --prob-boundary.")
        Z = np.array(Z, dtype=float)
        title = "Logistic p(phase=1)"
    else:
        Z = vp.get("score_grid", None)
        if Z is None:
            raise SystemExit("No score_grid (kmeans).")
        Z = np.array(Z, dtype=float)
        title = "KMeans soft score p(phase=1)"

    extent = [float(g.min()), float(g.max()), float(a2.min()), float(a2.max())]
    plt.figure()
    plt.imshow(Z, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label=title)

    # Boundary line (vector_boundary_line in report)
    bl = rep.get("vector_boundary_line", None)
    if bl:
        a2_line = np.array([p["axis2"] for p in bl], dtype=float)
        g_star = np.array([p["g_star"] for p in bl], dtype=float)
        mask = np.isfinite(a2_line) & np.isfinite(g_star)
        if mask.any():
            plt.plot(g_star[mask], a2_line[mask], "w--", linewidth=2.0)

    # Uncertainty band from logistic bootstrap (if present)
    band = vp.get("logistic_band", None)
    if band and band.get("g_star_p05") and band.get("g_star_p95"):
        p05 = np.array(band["g_star_p05"], dtype=float)
        p95 = np.array(band["g_star_p95"], dtype=float)
        mask = np.isfinite(p05) & np.isfinite(p95)
        if mask.any():
            plt.fill_betweenx(a2[mask], p05[mask], p95[mask], alpha=0.20)

    plt.xlabel("coupling g")
    plt.ylabel("T" if axis2 == "T" else "noise")
    plt.title(f"Probabilistic phase boundary ({title}) axis2={axis2}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=170)
    print("Saved plot:", args.out)

if __name__ == "__main__":
    main()
