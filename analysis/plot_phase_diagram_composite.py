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
    p = argparse.ArgumentParser(description="Plot 2D phase diagram using composite vector classifier.")
    p.add_argument("--report", type=str, default="experiments/outputs/phase_diagram_report.json")
    p.add_argument("--out", type=str, default="experiments/outputs/phase_diagram_composite.png")
    p.add_argument("--field", type=str, default="score", choices=["score","labels"],
                   help="Plot composite score p(phase=1) or hard labels.")
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
        raise SystemExit("No vector_phase found in report. Run phase_diagram_scan with --vector-classifier.")

    if args.field == "labels":
        Z = np.array(vp["label_grid"], dtype=float)
        cmap = None
    else:
        if "score_grid" not in vp:
            raise SystemExit("No score_grid. Requires k=2 in vector classifier.")
        Z = np.array(vp["score_grid"], dtype=float)
        cmap = None

    extent = [float(g.min()), float(g.max()), float(a2.min()), float(a2.max())]
    plt.figure()
    plt.imshow(Z, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    plt.colorbar(label=("label" if args.field=="labels" else "p(phase=1)"))

    # overlay composite boundary line if present
    bl = rep.get("vector_boundary_line", None)
    if bl:
        a2_line = np.array([p["axis2"] for p in bl], dtype=float)
        g_star = np.array([p["g_star"] for p in bl], dtype=float)
        mask = np.isfinite(a2_line) & np.isfinite(g_star)
        if mask.any():
            plt.plot(g_star[mask], a2_line[mask], "w--", linewidth=2.0)
            plt.scatter(g_star[mask], a2_line[mask], s=18)

    plt.xlabel("coupling g")
    plt.ylabel("T" if axis2 == "T" else "noise")
    plt.title(f"Composite phase diagram ({args.field})  axis2={axis2}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=170)
    print("Saved plot:", args.out)

if __name__ == "__main__":
    main()
