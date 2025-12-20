from __future__ import annotations
import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import QuarticAction, CubicAction
from matrix_models.sampler_langevin import LangevinSampler

from correspondence_maps.roundtrip import RoundTripMapper
from analysis.correspondence_diagnostics import CorrespondenceDiagnostics
from analysis.phase_diagram import compute_transition_line
from analysis.composite_phase_boundary import score_grid_to_transition_line
from analysis.vector_phase_classifier import classify_phases_from_feature_grids, choose_cluster_semantics
from analysis.logistic_boundary import fit_logistic_regression, boundary_gstar_from_scores, bootstrap_logistic_boundary

from autodidactic_protocols.unsupervised_update import SelfConsistencyLearner
from autodidactic_protocols.mutual_information_drive import MutualInformationLearner
from autodidactic_protocols.meta_objective import MetaObjectiveLearner


def parse_args():
    p = argparse.ArgumentParser(description="Compute a 2D phase diagram over (g x T) or (g x noise), with transition line detection.")
    p.add_argument("--axis2", type=str, default="T", choices=["T", "noise"])
    p.add_argument("--axis2-min", type=float, default=0.002)
    p.add_argument("--axis2-max", type=float, default=0.02)
    p.add_argument("--axis2-n", type=int, default=6)

    p.add_argument("--g-min", type=float, default=-0.25)
    p.add_argument("--g-max", type=float, default=0.25)
    p.add_argument("--g-n", type=int, default=11)

    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--sampler-steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=2)

    p.add_argument("--use-quartic", action="store_true", help="Use quartic action for stability.")
    p.add_argument("--learner", type=str, default="meta", choices=["self", "mi", "meta"])
    p.add_argument("--train-epochs", type=int, default=8)

    p.add_argument("--T", type=float, default=1e-2, help="Base Langevin temperature (if axis2 != T).")
    p.add_argument("--noise-std", type=float, default=0.15, help="Training noise std (if axis2 != noise).")

    p.add_argument("--order", type=str, default="lap_gap_Mhat",
                   choices=["lap_effdim_Mhat", "lap_gap_Mhat", "lap_spec_l2_rel", "rel_tr_M2", "mi_proxy"],
                   help="Scalar order parameter (mean over repeats).")

    p.add_argument("--model-type", type=str, default="continuous", choices=["linear", "continuous", "constant"])
    p.add_argument("--max-breaks", type=int, default=2)
    p.add_argument("--min-seg", type=int, default=3)
    p.add_argument("--strategy", type=str, default="largest_jump", choices=["largest_jump", "first", "middle"],
                   help="If multiple breakpoints, choose which defines g*(axis2).")

    p.add_argument("--out-csv", type=str, default="experiments/outputs/phase_diagram_grid.csv")
    p.add_argument("--out-json", type=str, default="experiments/outputs/phase_diagram_report.json")

    # Vector phase classifier & probabilistic boundary
    p.add_argument("--vector-classifier", action="store_true",
                   help="Cluster phases in vector feature space (gap, effdim, MI proxy) and map a composite boundary line.")
    p.add_argument("--vector-features", type=str, default="lap_gap_Mhat,lap_effdim_Mhat,mi_proxy",
                   help="Comma-separated feature keys (mi_proxy resolves to train_mi_proxy internally).")
    p.add_argument("--k", type=int, default=2, help="Clusters for composite classifier (2 recommended).")
    p.add_argument("--soft-temp", type=float, default=1.0, help="Soft assignment temperature for k=2 composite score.")

    p.add_argument("--prob-boundary", action="store_true",
                   help="Fit a tiny logistic model on clustered labels to get a probabilistic phase boundary + uncertainty band.")
    p.add_argument("--l2", type=float, default=1e-2, help="L2 regularization for logistic boundary fit.")
    p.add_argument("--boot-boundary", type=int, default=200, help="Bootstrap samples for boundary band (0 disables).")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for boundary contour.")
    return p.parse_args()


def _resolve_feature_keys(raw: str) -> List[str]:
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    # map user-facing "mi_proxy" to stored train metric key
    out = []
    for k in keys:
        out.append("train_mi_proxy" if k == "mi_proxy" else k)
    return out


def _choose_action(use_quartic: bool, g: float):
    return QuarticAction(g=float(g)) if use_quartic else CubicAction(g=float(g))


def _choose_learner(name: str):
    if name == "self":
        return SelfConsistencyLearner(steps_per_update=10, batch_size=64)
    if name == "mi":
        return MutualInformationLearner(steps_per_update=10, batch_size=64)
    return MetaObjectiveLearner(steps_per_update=10, batch_size=64)


def _run_one_point(
    dim: int,
    width: int,
    g: float,
    T: float,
    sampler_steps: int,
    dt: float,
    seed: int,
    learner_name: str,
    train_epochs: int,
    noise_std: float,
    use_quartic: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    # sample an initial matrix and run Langevin
    ens = HermitianEnsemble(dim=dim, seed=seed)
    M0 = ens.sample()
    sampler = LangevinSampler(action=_choose_action(use_quartic, g=g), temperature=float(T), dt=float(dt), seed=seed)
    M = sampler.run(M0, steps=int(sampler_steps))

    # round-trip map
    mapper = RoundTripMapper(hidden_width=width, seed=seed)
    model = mapper.build_model_from_matrix(M)

    learner = _choose_learner(learner_name)
    train_metrics = learner.update(model, epochs=int(train_epochs), noise_std=float(noise_std))

    M_hat = mapper.extract_matrix_from_model(model, dim=dim)

    # diagnostics
    diag = CorrespondenceDiagnostics(max_power=6, laplacian_k=12).compare(M, M_hat, model=model, train_metrics=train_metrics)

    # also expose MI proxy consistently (even though it is stored as train_mi_proxy)
    if "train_mi_proxy" in diag:
        diag["mi_proxy"] = float(diag["train_mi_proxy"])

    return M, M_hat, train_metrics, diag


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    axis2_vals = np.linspace(float(args.axis2_min), float(args.axis2_max), int(args.axis2_n))
    g_vals = np.linspace(float(args.g_min), float(args.g_max), int(args.g_n))

    order_grid = np.full((axis2_vals.size, g_vals.size), np.nan, dtype=float)

    # Always collect these three as the "vector" basis (even if you choose a different order scalar)
    feature_grids: Dict[str, np.ndarray] = {
        "lap_gap_Mhat": np.full_like(order_grid, np.nan),
        "lap_effdim_Mhat": np.full_like(order_grid, np.nan),
        "train_mi_proxy": np.full_like(order_grid, np.nan),
    }

    # For report/debug: keep a small metric subset
    metric_keys_keep = ["lap_gap_Mhat", "lap_effdim_Mhat", "lap_spec_l2_rel", "rel_tr_M2", "train_mi_proxy", "rnn_spec_radius"]
    metrics_grid: Dict[str, np.ndarray] = {k: np.full_like(order_grid, np.nan) for k in metric_keys_keep}

    rows = []
    for i, a2 in enumerate(axis2_vals):
        for j, g in enumerate(g_vals):
            # axis2 sets either Langevin temperature or training noise
            T = float(a2) if args.axis2 == "T" else float(args.T)
            noise = float(a2) if args.axis2 == "noise" else float(args.noise_std)

            diags = []
            for r in range(int(args.repeats)):
                seed = int(args.seed) + 1000*i + 17*j + r
                _, _, _, diag = _run_one_point(
                    dim=int(args.dim),
                    width=int(args.width),
                    g=float(g),
                    T=T,
                    sampler_steps=int(args.sampler_steps),
                    dt=float(args.dt),
                    seed=seed,
                    learner_name=str(args.learner),
                    train_epochs=int(args.train_epochs),
                    noise_std=noise,
                    use_quartic=bool(args.use_quartic),
                )
                diags.append(diag)

            # aggregate metrics over repeats
            agg: Dict[str, float] = {}
            keys = set().union(*[d.keys() for d in diags])
            for k in keys:
                vals = [d[k] for d in diags if k in d and np.isfinite(d[k])]
                if vals:
                    agg[k] = float(np.mean(vals))

            # scalar order parameter selection
            if args.order == "mi_proxy":
                order = agg.get("mi_proxy", agg.get("train_mi_proxy", float("nan")))
            else:
                order = agg.get(args.order, float("nan"))

            order_grid[i, j] = float(order)

            # vector features
            feature_grids["lap_gap_Mhat"][i, j] = float(agg.get("lap_gap_Mhat", float("nan")))
            feature_grids["lap_effdim_Mhat"][i, j] = float(agg.get("lap_effdim_Mhat", float("nan")))
            feature_grids["train_mi_proxy"][i, j] = float(agg.get("train_mi_proxy", float("nan")))

            # keep selected metrics
            for k in metric_keys_keep:
                metrics_grid[k][i, j] = float(agg.get(k, float("nan")))

            rows.append({
                "axis2": float(a2),
                "g": float(g),
                "T": float(T),
                "noise_std": float(noise),
                "order": float(order),
                **{f"m_{k}": float(agg.get(k, float('nan'))) for k in metric_keys_keep},
            })

    # scalar transition line
    scalar_pts = compute_transition_line(
        g_values=g_vals,
        axis2_values=axis2_vals,
        order_grid=order_grid,
        model_type=args.model_type,
        max_breaks=int(args.max_breaks),
        min_seg=int(args.min_seg),
        strategy=args.strategy,
    )
    scalar_boundary_line = [{"axis2": p.axis2, "g_star": p.g_star, "breaks_x": p.breaks_x, "bic": p.bic, "sse": p.sse} for p in scalar_pts]

    vector_phase = None
    vector_boundary_line = None

    if bool(args.vector_classifier):
        feat_keys = _resolve_feature_keys(str(args.vector_features))
        vp = classify_phases_from_feature_grids(
            feature_grids={**feature_grids},  # copy
            axis2_values=axis2_vals,
            g_values=g_vals,
            feature_keys=feat_keys,
            k=int(args.k),
            seed=int(args.seed),
            soft_temp=float(args.soft_temp),
        )

        # Choose consistent semantics: phase=1 should correspond to "higher" lap_gap_Mhat by default.
        sem_key = "lap_gap_Mhat" if "lap_gap_Mhat" in feature_grids else feat_keys[0]
        mapping = choose_cluster_semantics(feature_grids, axis2_vals, g_vals, np.array(vp["label_grid"], dtype=int), feature_key=sem_key)

        label_grid = np.array(vp["label_grid"], dtype=int)
        labels_flat = np.array(vp["labels_flat"], dtype=int)
        score_grid = np.array(vp.get("score_grid", np.full_like(order_grid, np.nan)), dtype=float)

        # remap labels to semantics
        label_grid_sem = np.vectorize(lambda x: mapping.get(int(x), int(x)))(label_grid)
        labels_flat_sem = np.array([mapping.get(int(x), int(x)) for x in labels_flat], dtype=int)

        # if "cluster 1" got mapped to semantic 0, flip scores so score ~ p(phase=1)
        # by construction, score_grid is prob of raw label==1
        if mapping.get(1, 1) == 0:
            score_grid = 1.0 - score_grid

        vp["label_grid"] = label_grid_sem.tolist()
        vp["labels_flat"] = labels_flat_sem.tolist()
        vp["score_grid"] = score_grid.tolist()
        vp["cluster_semantics"] = {str(k): int(v) for k, v in mapping.items()}

        # boundary line from either score_grid (kmeans soft) or logistic probability
        logistic = None
        logistic_band = None
        logistic_score_grid = None

        if bool(args.prob_boundary) and int(args.k) == 2:
            X_raw = np.array(vp["X_raw"], dtype=float)
            idx_map = np.array(vp["idx_map"], dtype=int)
            y = np.array(vp["labels_flat"], dtype=int)
            y = (y > 0).astype(int)

            logistic = fit_logistic_regression(X_raw, y, l2=float(args.l2), max_iter=80, tol=1e-6, seed=int(args.seed))
            p = logistic.predict_proba(X_raw)

            logistic_score_grid = np.full_like(order_grid, np.nan, dtype=float)
            for pp, (ia2, ig) in zip(p, idx_map):
                logistic_score_grid[int(ia2), int(ig)] = float(pp)

            # direct contour crossing
            _, g_star = boundary_gstar_from_scores(g_vals, axis2_vals, logistic_score_grid, threshold=float(args.threshold))
            vector_boundary_line = [{"axis2": float(axis2_vals[ii]), "g_star": float(g_star[ii]), "breaks_x": [], "bic": float("nan"), "sse": float("nan")} for ii in range(len(axis2_vals))]

            if int(args.boot_boundary) > 0:
                logistic_band = bootstrap_logistic_boundary(
                    X=X_raw, y=y, g_values=g_vals, axis2_values=axis2_vals, feature_index_map=idx_map,
                    l2=float(args.l2), n_boot=int(args.boot_boundary), seed=int(args.seed) + 2025,
                    threshold=float(args.threshold)
                )

            vp["logistic"] = {"w": logistic.w.tolist(), "b": float(logistic.b), "mu": logistic.mu.tolist(), "sd": logistic.sd.tolist()}
            vp["logistic_score_grid"] = logistic_score_grid.tolist()
            vp["logistic_band"] = logistic_band
        else:
            # breakpoint machinery on the soft score grid
            vb_pts = score_grid_to_transition_line(
                g_values=g_vals,
                axis2_values=axis2_vals,
                score_grid=score_grid,
                model_type=args.model_type,
                max_breaks=int(args.max_breaks),
                min_seg=int(args.min_seg),
                strategy=args.strategy,
            )
            vector_boundary_line = [{"axis2": p.axis2, "g_star": p.g_star, "breaks_x": p.breaks_x, "bic": p.bic, "sse": p.sse} for p in vb_pts]

        vector_phase = vp

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    report = {
        "axis2": str(args.axis2),
        "axis2_values": axis2_vals.tolist(),
        "g_values": g_vals.tolist(),
        "order_key": str(args.order),
        "order_grid": order_grid.tolist(),
        "scalar_boundary_line": scalar_boundary_line,
        "metrics_grid": {k: v.tolist() for k, v in metrics_grid.items()},
        "feature_grids": {k: v.tolist() for k, v in feature_grids.items()},
        "vector_phase": vector_phase,
        "vector_boundary_line": vector_boundary_line,
        "n_repeats": int(args.repeats),
        "params": {
            "dim": int(args.dim),
            "width": int(args.width),
            "sampler_steps": int(args.sampler_steps),
            "dt": float(args.dt),
            "learner": str(args.learner),
            "train_epochs": int(args.train_epochs),
            "use_quartic": bool(args.use_quartic),
            "model_type": str(args.model_type),
            "max_breaks": int(args.max_breaks),
            "min_seg": int(args.min_seg),
            "strategy": str(args.strategy),
            "prob_boundary": bool(args.prob_boundary),
            "threshold": float(args.threshold),
            "l2": float(args.l2),
            "boot_boundary": int(args.boot_boundary),
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Wrote:", args.out_csv)
    print("Wrote:", args.out_json)


if __name__ == "__main__":
    main()
