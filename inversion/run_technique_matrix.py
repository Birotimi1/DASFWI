"""Technique-combination search: which stack is best for deployment?

FWI has many composable choices (misfit x optimizer x regularizer x
starting-model x source-independence). The full cross-product is large, so a
brute-force grid is neither affordable nor necessary. This driver runs a
CONFIGURABLE grid over the axes you name, scores each run with a single
deployment metric (inversion.config.deployment_score), and ranks them - so you
find the best combination empirically instead of guessing.

Recommended STAGED search (cheap -> refine), not full factorial:
  1. misfit x optimizer, regularizer=none        (the 45-cell base grid)
        python inversion/run_technique_matrix.py
  2. take the top few combos, sweep regularizers on those
        python inversion/run_technique_matrix.py --misfits gc,sinkhorn \
               --optimizers sgd --regularizers none,tikhonov1,tv1
  3. take the winner, sweep starting-model quality (the degradation ladder,
        inversion/run_starting_model_ladder.py)

Each axis is a comma list; the driver runs their cartesian product. Uses the
Marmousi crop (known truth -> deployment_score is meaningful). --quick shrinks
everything for a local wiring check.
"""

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from inversion import config
from inversion.run_inverse_crime import run_inverse_crime
from inversion.run_marmousi_demo import load_crop

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "technique_matrix")

# per-optimizer learning rates for the search (experiment lrs, not Liu's fixed
# campaign lr=10): sgd is gradient-proportional (peak lr*vmax), the Adam family
# moves ~lr per cell so needs a small value on this DAS setup.
SEARCH_LR = {"sgd": 0.004, "adagrad": 5.0, "adam": 2.0, "adamw": 2.0,
             "nadam": 2.0}


def pick_device(arg):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"           # NOT mps: envelope/weci/sdtw/convsi need FFT/pysdtw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misfits", default=",".join(config.MISFITS))
    ap.add_argument("--optimizers", default=",".join(config.OPTIMIZER_NAMES))
    ap.add_argument("--regularizers", default="none")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    misfits = [m for m in args.misfits.split(",") if m]
    optimizers = [o for o in args.optimizers.split(",") if o]
    regs = [r for r in args.regularizers.split(",") if r]
    for m in misfits:
        assert m in config.MISFITS, f"unknown misfit {m}"
    for o in optimizers:
        assert o in config.OPTIMIZER_NAMES, f"unknown optimizer {o}"
    for r in regs:
        assert r in config.REGULARIZERS, f"unknown regularizer {r}"
    device = pick_device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)

    vp_true, rho_true = load_crop()
    if args.quick:
        vp_true, rho_true = vp_true[:60, :60], rho_true[:60, :60]
    vp_init = gaussian_filter(vp_true, sigma=12.0)
    bands = [(5.0, 1), (10.0, 1)] if args.quick else [(5.0, 12), (10.0, 13)]
    nt = 700 if args.quick else 2500
    total = sum(n for _, n in bands)
    fiber = dict(x_well=(30 if args.quick else 150) * 5.0, z_top=100.0,
                 n_channels=10 if args.quick else 100)
    shots = dict(x_indices=(10, 45) if args.quick else (20, 90, 160, 240),
                 z_index=2, f0=10.0)

    results = []
    grid = list(itertools.product(misfits, optimizers, regs))
    print(f"running {len(grid)} combinations on {device}", flush=True)
    for misfit, opt, reg in grid:
        tag = config.InversionConfig(misfit=misfit, optimizer=opt,
                                     regularization=reg).tag()
        t0 = time.time()
        try:
            mis = "gc" if misfit == "gc" else config.build_misfit(
                misfit, dt=4e-4, iterations=total)
            res = run_inverse_crime(dict(
                vp_true=vp_true, vp_init=vp_init, nt=nt, dt=4e-4,
                fiber=fiber, shots=shots, optimizer=opt,
                lr=SEARCH_LR.get(opt, 1.0), misfit=mis, bands=bands,
                dtype=torch.float32, vp_bound=(1400.0, 3700.0), rho=rho_true,
                grad_mask_top=5, regularization=reg,
                waveform_normalize=config.MISFIT_SETTINGS[misfit]["normalize"],
                device=device))
            score = config.deployment_score(vp_true, vp_init, res["vp_final"],
                                            res["iter_loss"])
        except Exception as e:
            score = {"score": float("nan"), "error": f"{type(e).__name__}: {e}"}
        score.update(misfit=misfit, optimizer=opt, regularization=reg,
                     tag=tag, runtime_s=round(time.time() - t0, 1))
        results.append(score)
        print(f"  {tag:28s} score={score.get('score')!s:>7} "
              f"corr={score.get('update_corr', float('nan')):+.3f} "
              f"({score['runtime_s']:.0f}s)", flush=True)

    ranked = sorted(results, key=lambda r: (np.isnan(r.get("score", np.nan)),
                                            -(r.get("score") or -9)))
    with open(os.path.join(OUT_DIR, "ranking.json"), "w") as f:
        json.dump(ranked, f, indent=2)

    print("\n=== RANKED (best deployment stack first) ===", flush=True)
    for r in ranked[:10]:
        print(f"  {r['tag']:28s} score={r.get('score')!s:>7} "
              f"recovery={r.get('recovery', float('nan')):+.3f} "
              f"corr={r.get('update_corr', float('nan')):+.3f}", flush=True)
    print("saved ranking.json to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
