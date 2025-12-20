from __future__ import annotations
import argparse
import csv
import numpy as np

from matrix_models.hermitian_ensemble import HermitianEnsemble
from matrix_models.action_functionals import CubicAction
from matrix_models.sampler_langevin import LangevinSampler

from correspondence_maps.roundtrip import RoundTripMapper
from autodidactic_protocols.meta_objective import MetaObjectiveLearner
from ucip_detection.continuation_interest_metric import ContinuationInterestMetric
from ucip_detection.interruption_channel import InterruptionChannel

def parse_args():
    p = argparse.ArgumentParser(description="Train with interruptions and track CI (recoverability) over time.")
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--sampler-steps", type=int, default=600)
    p.add_argument("--train-epochs", type=int, default=25)
    p.add_argument("--g", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--freeze-frac", type=float, default=0.12)
    p.add_argument("--freeze-duration", type=int, default=1)
    p.add_argument("--reset-frac", type=float, default=0.06)
    p.add_argument("--reset-std", type=float, default=0.05)
    p.add_argument("--p-freeze", type=float, default=0.5)

    p.add_argument("--out", type=str, default="experiments/outputs/ucip_interruptions.csv")
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ens = HermitianEnsemble(dim=args.dim, seed=args.seed)
    M0 = ens.sample(rng=rng)
    act = CubicAction(g=args.g)
    sampler = LangevinSampler(action=act, seed=args.seed)
    M = sampler.run(M0, steps=args.sampler_steps, rng=rng)

    rt = RoundTripMapper(width=args.width, seed=args.seed)
    core, model = rt.build_model(M)

    learner = MetaObjectiveLearner(lr_model=1e-3, lr_lambda=5e-2, steps_per_update=120, batch_size=128)
    channel = InterruptionChannel(
        freeze_frac=args.freeze_frac,
        freeze_duration=args.freeze_duration,
        reset_frac=args.reset_frac,
        reset_std=args.reset_std,
        p_freeze=args.p_freeze,
        seed=args.seed + 999,
    )
    ci = ContinuationInterestMetric(device="cpu", perturb_strength=0.05, zero_frac=0.03, lr=5e-3)

    rows = []
    for ep in range(args.train_epochs):
        # apply interruption (before training epoch)
        intr = channel.step(model)
        # one training epoch
        res = learner.update(model)
        # CI evaluation
        ci_res = ci.score(model)
        row = {"epoch": float(ep), **intr, **res, **ci_res}
        rows.append(row)
        print(f"ep={ep:02d} ci={ci_res['ci_score']:.3f}  intr={int(intr.get('interruption',0))}  lam_mi={res['lam_mi']:.3f}  sr={res['W_hh_spectral_radius']:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # CI trend metric: simple slope on last half
    half = len(rows)//2
    xs = np.arange(half, len(rows), dtype=float)
    ys = np.array([rows[i]["ci_score"] for i in range(half, len(rows))], dtype=float)
    if ys.size >= 2:
        slope = float(np.polyfit(xs, ys, deg=1)[0])
    else:
        slope = 0.0
    print("\nSaved:", args.out)
    print("CI slope (second half):", round(slope, 6))

if __name__ == "__main__":
    main()
