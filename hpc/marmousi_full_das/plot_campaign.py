#!/usr/bin/env python3
"""Plot the true Marmousi2 model against the DAS-inverted models.

Reads results/marmousi_full_das/setup.npz (vp_true, vp_init) and each combo's
iter_vp.npz (recovered model = last cached iteration) and lays them out with a
shared velocity colorbar, titled with each run's dRMS% and update-correlation.
Dependency-light: numpy + matplotlib only (no ADFWI/torch).

Usage (from the DASFWI repo root):
    # ALL combos, paginated: original Vp first, then every test, 10 panels/figure
    # (writes marmousi_vp_part1..N.png + test_legend.csv under <results>/figures):
    python hpc/marmousi_full_das/plot_campaign.py --all

    # true + initial + the top 4 combos by recovery score (single figure):
    python hpc/marmousi_full_das/plot_campaign.py

    # specific combos over their (rec - true) error maps:
    python hpc/marmousi_full_das/plot_campaign.py --combos l2_adam,convsi_adam,gc_adam --residual
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless-safe (login node / cluster)
import matplotlib.pyplot as plt

# geometry, matching hpc/marmousi_full_das/common.py
DX = DZ = 40.0
X0 = 5000.0

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT = os.environ.get(
    "DASFWI_RESULTS", os.path.join(_REPO, "results", "marmousi_full_das"))

# ---- decode of the test names (misfit_optimizer) for the legend CSV ----------
MISFITS = ("l2", "envelope", "gc", "sdtw", "sinkhorn", "weci",
           "traveltime", "nim", "convsi")
OPTIMIZERS = ("sgd", "adagrad", "adam", "adamw", "nadam")
MISFIT_MEANING = {
    "l2":        ("Waveform L2 (least squares)",
                  "Full amplitude+phase waveform difference; the baseline objective, "
                  "most nonlinear (cycle-skips first). Best when the source is known."),
    "envelope":  ("Envelope (instantaneous amplitude)",
                  "Matches Hilbert-envelope amplitudes; robust to phase errors / cycle-skipping."),
    "gc":        ("Global Correlation",
                  "Normalized zero-lag cross-correlation; amplitude/scale-insensitive. "
                  "Good field candidate."),
    "sdtw":      ("Soft Dynamic Time Warping",
                  "Differentiable DTW alignment; tolerant to time warps."),
    "sinkhorn":  ("Wasserstein-Sinkhorn Optimal Transport",
                  "Entropic-regularized OT distance; convex in time shift, cycle-skip robust (slow)."),
    "weci":      ("Weighted Envelope->Correlation hybrid",
                  "Sigmoid blend from envelope to global-correlation; needs long runs."),
    "traveltime":("Cross-correlation Traveltime",
                  "Kinematic time-shift measurement; amplitude-free, low-resolution starter."),
    "nim":       ("Normalized Integration Method",
                  "Normalizes+integrates to a distribution (= Wasserstein-1, p=1); cycle-skip "
                  "robust. NOTE: diverged on strain rate in this campaign."),
    "convsi":    ("Convolved-wavefields SOURCE-INDEPENDENT (Choi & Alkhalifah 2011)",
                  "Cancels the unknown source wavelet by cross-convolution; the recommended "
                  "FIELD misfit."),
}
OPT_MEANING = {
    "sgd":     ("SGD", "Gradient-proportional steepest descent (illumination-preconditioned; Liu's norm-grad)."),
    "adagrad": ("Adagrad", "Per-parameter learning rate that decays with accumulated squared gradient."),
    "adam":    ("Adam", "Adaptive moment estimation (1st+2nd moment); gradient-scale-invariant steps."),
    "adamw":   ("AdamW", "Adam with decoupled weight decay."),
    "nadam":   ("Nadam", "Adam with Nesterov-accelerated momentum."),
}


def _score(m):
    if not m.get("losses_finite", False):
        return 0.0
    ri, rf = m.get("rms_init", 0.0), m.get("rms_final", 0.0)
    frac = max(0.0, 1.0 - rf / ri) if ri > 0 else 0.0
    return m.get("update_corr", 0.0) * frac


def _load_metrics(results):
    out = {}
    for f in glob.glob(os.path.join(results, "*", "metrics.json")):
        try:
            m = json.load(open(f))
            out[m.get("tag", os.path.basename(os.path.dirname(f)))] = m
        except Exception:                 # noqa: BLE001
            pass
    return out


def _final_vp(results, tag):
    """Recovered model = last cached slice of iter_vp.npz for this combo."""
    f = os.path.join(results, tag, "iter_vp.npz")
    if not os.path.isfile(f):
        return None
    arr = np.load(f)["data"]
    return np.asarray(arr[-1] if arr.ndim == 3 else arr, dtype=float)


def _extent(nz, nx):
    return [X0 / 1000.0, (X0 + DX * nx) / 1000.0, DZ * nz / 1000.0, 0.0]


def write_legend(path, metrics):
    """Write test_legend.csv: every misfit_optimizer test name -> its meaning,
    plus this campaign's dRMS%/update-corr where the run has completed."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_name", "misfit_code", "optimizer_code", "misfit",
                    "optimizer", "dRMS_percent", "update_corr", "meaning"])
        for mis in MISFITS:
            mname, mdesc = MISFIT_MEANING[mis]
            for opt in OPTIMIZERS:
                oname, odesc = OPT_MEANING[opt]
                tag = f"{mis}_{opt}"
                m = metrics.get(tag, {})
                drms = ("" if not m.get("rms_init") else
                        round(100 * (m["rms_init"] - m["rms_final"]) / m["rms_init"], 1))
                corr = round(m["update_corr"], 3) if "update_corr" in m else ""
                w.writerow([tag, mis, opt, mname, oname, drms, corr,
                            f"{mname} misfit optimized with {oname}. {mdesc} {odesc}"])
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT)
    ap.add_argument("--combos", default=None,
                    help="comma-separated tags; overrides --best")
    ap.add_argument("--best", type=int, default=4,
                    help="show the top-N combos by recovery score (default 4)")
    ap.add_argument("--all", action="store_true",
                    help="plot ORIGINAL Vp + EVERY combo, paginated into figures "
                         "(--per-fig panels each), ranked best->worst; also writes "
                         "test_legend.csv")
    ap.add_argument("--per-fig", type=int, default=10, dest="per_fig",
                    help="recovered-model panels per figure in --all mode "
                         "(default 10; the original is added to figure 1)")
    ap.add_argument("--residual", action="store_true",
                    help="add a second row of (recovered - true) residuals")
    ap.add_argument("--fibers", action="store_true",
                    help="overlay the DAS fiber channel depths from setup.npz")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--outdir", default=None,
                    help="directory for --all figures (default <results>/figures)")
    ap.add_argument("--out", default=None,
                    help="output path for the single-figure (non --all) modes")
    args = ap.parse_args()

    setup_f = os.path.join(args.results, "setup.npz")
    if not os.path.isfile(setup_f):
        raise SystemExit(f"no setup.npz under {args.results} "
                         f"(rsync results/marmousi_full_das/ from the cluster)")
    setup = np.load(setup_f)
    vp_true, vp_init = setup["vp_true"], setup["vp_init"]
    nz, nx = vp_true.shape
    ext = _extent(nz, nx)
    vmin, vmax = float(vp_true.min()), float(vp_true.max())
    metrics = _load_metrics(args.results)
    fib_z = np.asarray(setup["channel_z"]) / 1000.0 if "channel_z" in setup else None

    def show(ax, data, title, cmap, vlo, vhi, xlab=True, ylab=True, fibers=False):
        im = ax.imshow(data, extent=ext, aspect="auto", cmap=cmap,
                       vmin=vlo, vmax=vhi)
        ax.set_title(title, fontsize=10)
        if xlab:
            ax.set_xlabel("x (km)")
        if ylab:
            ax.set_ylabel("z (km)")
        if fibers and args.fibers and fib_z is not None:
            ax.scatter(np.full_like(fib_z, ext[0] + 0.05), fib_z, s=1,
                       c="k", marker="_", alpha=0.5)
        return im

    # =====================================================================
    # --all : original Vp + every combo, paginated best->worst
    # =====================================================================
    if args.all:
        outdir = args.outdir or os.path.join(args.results, "figures")
        os.makedirs(outdir, exist_ok=True)
        ranked = sorted(metrics.values(), key=_score, reverse=True)
        items = []                        # (panel_title, vp) for each combo
        for m in ranked:
            tag = m["tag"]
            vp = _final_vp(args.results, tag)
            if vp is None:
                print(f"  skip {tag}: no iter_vp.npz")
                continue
            drms = (100 * (m["rms_init"] - m["rms_final"]) / m["rms_init"]
                    if m.get("rms_init") else float("nan"))
            items.append((f"{tag}\ndRMS {drms:.0f}%  corr "
                          f"{m.get('update_corr', float('nan')):.3f}", vp))
        if not items:
            raise SystemExit(f"no combos with iter_vp.npz under {args.results}")

        per = max(1, args.per_fig)
        n_fig = math.ceil(len(items) / per)
        for fi in range(n_fig):
            chunk = items[fi * per:(fi + 1) * per]
            # the original (true) model leads figure 1 as a reference panel
            page = ([("ORIGINAL Marmousi Vp (true model)", vp_true)] if fi == 0
                    else []) + chunk
            k = len(page)
            ncol = max(1, math.ceil(k / 2))     # ~2 rows -> uniform panel height
            nrow = math.ceil(k / ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.7 * nrow),
                                     squeeze=False, constrained_layout=True)
            flat = [axes[r][c] for r in range(nrow) for c in range(ncol)]
            im_v = None
            for i, (ax, (title, data)) in enumerate(zip(flat, page)):
                im_v = show(ax, data, title, args.cmap, vmin, vmax,
                            xlab=(i >= k - ncol), ylab=(i % ncol == 0),
                            fibers=(title.startswith("ORIGINAL")))
            for ax in flat[k:]:
                ax.axis("off")
            fig.colorbar(im_v, ax=axes.ravel().tolist(), shrink=0.7,
                         label="Vp (m/s)", pad=0.02)
            fig.suptitle(f"Marmousi2 DAS strain-rate FWI — recovered Vp "
                         f"(part {fi + 1}/{n_fig})", fontsize=13)
            out = os.path.join(outdir, f"marmousi_vp_part{fi + 1}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print("wrote", out, f"({k} panels)")
        write_legend(os.path.join(outdir, "test_legend.csv"), metrics)
        print(f"done: {n_fig} figures + legend in {outdir}")
        return

    # =====================================================================
    # single-figure modes (top-N or explicit --combos), optional residual
    # =====================================================================
    if args.combos:
        tags = [t.strip() for t in args.combos.split(",") if t.strip()]
    else:
        ranked = sorted(metrics.values(), key=_score, reverse=True)
        tags = [m["tag"] for m in ranked[:args.best]]
    if not tags:
        raise SystemExit(f"no combos found under {args.results} "
                         "(need <tag>/iter_vp.npz + metrics.json)")

    panels = [("true model", vp_true, None), ("initial model", vp_init, None)]
    for tag in tags:
        vp = _final_vp(args.results, tag)
        if vp is None:
            print(f"  skip {tag}: no iter_vp.npz")
            continue
        m = metrics.get(tag, {})
        drms = (100 * (m["rms_init"] - m["rms_final"]) / m["rms_init"]
                if m.get("rms_init") else float("nan"))
        sub = f"dRMS {drms:.0f}%  corr {m.get('update_corr', float('nan')):.3f}"
        panels.append((f"{tag}\n{sub}", vp, tag))

    recs = [(t, d, tag) for (t, d, tag) in panels if tag is not None]

    if not args.residual:
        n = len(panels)
        ncol = min(3, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow),
                                 squeeze=False, constrained_layout=True)
        flat = [axes[r][c] for r in range(nrow) for c in range(ncol)]
        im_v = None
        for i, (ax, (title, data, tag)) in enumerate(zip(flat, panels)):
            im_v = show(ax, data, title, args.cmap, vmin, vmax,
                        xlab=(i >= len(panels) - ncol), ylab=(i % ncol == 0),
                        fibers=True)
        for ax in flat[len(panels):]:
            ax.axis("off")
        fig.colorbar(im_v, ax=axes.ravel().tolist(), shrink=0.7,
                     label="Vp (m/s)", pad=0.02)
    else:
        ncol = len(recs)
        rlim = max((np.nanmax(np.abs(d - vp_true)) for (_t, d, _g) in recs),
                   default=1.0)
        fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 3.7 * 2),
                                 squeeze=False, constrained_layout=True)
        im_v = im_r = None
        for c, (title, data, tag) in enumerate(recs):
            im_v = show(axes[0][c], data, title, args.cmap, vmin, vmax,
                        xlab=False, ylab=(c == 0), fibers=True)
            im_r = show(axes[1][c], data - vp_true,
                        title.split("\n")[0] + "  (rec - true)",
                        "seismic", -rlim, rlim, xlab=True, ylab=(c == 0))
        fig.colorbar(im_v, ax=axes[0].tolist(), shrink=0.8,
                     label="Vp (m/s)", pad=0.02)
        fig.colorbar(im_r, ax=axes[1].tolist(), shrink=0.8,
                     label="Vp error (m/s)", pad=0.02)

    fig.suptitle("Marmousi2 DAS strain-rate FWI — recovered models", fontsize=13)
    out = args.out or os.path.join(args.results, "campaign_models.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out, f"({len(panels)} panels)")


if __name__ == "__main__":
    main()
