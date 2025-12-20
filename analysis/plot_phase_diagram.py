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
    p = argparse.ArgumentParser(description="Plot 2D phase diagram heatmap + transition line overlay.")
    p.add_argument("--report", type=str, default="experiments/outputs/phase_diagram_report.json")
    p.add_argument("--out", type=str, default="experiments/outputs/phase_diagram.png")
    return p.parse_args()

def main():
    args = parse_args()
    plt = _require_matplotlib()
    rep = json.load(open(args.report, "r", encoding="utf-8"))

    axis2 = rep["axis2"]
    a2 = np.array(rep["axis2_values"], dtype=float)
    g = np.array(rep["g_values"], dtype=float)
    Z = np.array(rep["order_grid"], dtype=float)

    # Build transition arrays
    tl = rep.get("transition_line", [])
    a2_line = np.array([p["axis2"] for p in tl], dtype=float)
    g_star = np.array([p["g_star"] for p in tl], dtype=float)

    plt.figure()
    # imshow expects row-major; axis2 on y, g on x
    extent = [float(g.min()), float(g.max()), float(a2.min()), float(a2.max())]
    plt.imshow(Z, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label=rep.get("order_key", "order"))

    # overlay transition line
    mask = np.isfinite(a2_line) & np.isfinite(g_star)
    if mask.any():
        plt.plot(g_star[mask], a2_line[mask], "w--", linewidth=2.0)
        plt.scatter(g_star[mask], a2_line[mask], s=18)

    plt.xlabel("coupling g")
    plt.ylabel("T" if axis2 == "T" else "noise")
    plt.title(f"Phase diagram: {rep.get('order_key','order')}  (axis2={axis2})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=170)
    print("Saved plot:", args.out)

if __name__ == "__main__":
    main()
