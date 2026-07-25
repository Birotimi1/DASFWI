#!/usr/bin/env python3
"""PHASE 1 RESULT: the cycle-skipping flip curve.

Answers the hypothesis gate: as the starting model degrades and cycle-skipping
sets in, does a ROBUST misfit (sinkhorn/OT, envelope, sdtw, traveltime, nim)
overtake L2 — and at which rung?

Reads every ladder rung:
    <results>/<combo>/metrics.json                 -> rung s6 (baseline campaign)
    <results>/ladder_<rung>/<combo>/metrics.json   -> the degraded rungs
and reports, for each rung, BOTH metric families (structural SSIM/MAPE and
amplitude dRMS) so a structure/amplitude disagreement is visible — plus the
measured cycle-skip fraction of the starting model.

    python hpc/marmousi_full_das/flip_curve.py
    python hpc/marmousi_full_das/flip_curve.py --csv flip.csv --plot flip.png

VERDICT logic: for each rung we find the best misfit (by mean SSIM over
optimizers, and by the single best combo). The FLIP RUNG is the first rung where
the winner is no longer an L2 variant. If L2 wins everywhere, the hypothesis is
NOT supported on DAS strain rate -> stop and redesign (do not build the adaptive
objective).
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from inversion.metrics import model_scores                       # noqa: E402

_DEFAULT = os.environ.get(
    "DASFWI_RESULTS", os.path.join(_REPO, "results", "marmousi_full_das"))
ROBUST = ("sinkhorn", "sdtw", "envelope", "nim", "traveltime", "weci", "convsi")


def _final_vp(d):
    f = os.path.join(d, "iter_vp.npz")
    if not os.path.isfile(f):
        return None
    a = np.load(f)["data"]
    return np.asarray(a[-1] if a.ndim == 3 else a, dtype=float)


def _collect(results, vp_true):
    """rung -> list of per-combo dicts."""
    rungs = defaultdict(list)
    pats = [(os.path.join(results, "*", "metrics.json"), "s6")]
    for d in sorted(glob.glob(os.path.join(results, "ladder_*"))):
        pats.append((os.path.join(d, "*", "metrics.json"),
                     os.path.basename(d).replace("ladder_", "")))
    for pat, rung_default in pats:
        for mf in sorted(glob.glob(pat)):
            try:
                m = json.load(open(mf))
            except Exception:                                    # noqa: BLE001
                continue
            d = os.path.dirname(mf)
            rung = m.get("start_rung") or rung_default
            tag = m.get("tag") or os.path.basename(d)
            m["_misfit"] = m.get("misfit") or tag.split("_")[0]
            m["_optimizer"] = m.get("optimizer") or tag.split("_")[-1]
            m["_combo"] = f"{m['_misfit']}_{m['_optimizer']}"
            if "ssim" not in m or "mape" not in m:
                vp = _final_vp(d)
                if vp is None or vp_true is None:
                    continue
                sc = model_scores(vp_true, vp)
                m["ssim"], m["mape"] = sc["ssim"], sc["mape"]
            finite = m.get("losses_finite", True)
            m["_ssim"] = m["ssim"] if finite else -1.0
            m["_drms"] = (100.0 * (m["rms_init"] - m["rms_final"]) / m["rms_init"]
                          if m.get("rms_init", 0) > 0 and finite else -1e9)
            rungs[rung].append(m)
    return rungs


def _order(rungs):
    """s6 < s12 < s16 < ... < const (worse = later)."""
    def key(r):
        return (1e9, r) if not r.startswith("s") else (int(r[1:]), r)
    return sorted(rungs, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    setup_f = os.path.join(args.results, "setup.npz")
    vp_true = (np.asarray(np.load(setup_f)["vp_true"], float)
               if os.path.isfile(setup_f) else None)
    rungs = _collect(args.results, vp_true)
    if not rungs:
        print(f"no results under {args.results}", file=sys.stderr)
        sys.exit(1)
    order = _order(rungs)

    print("=" * 78)
    print("PHASE 1 - CYCLE-SKIPPING FLIP CURVE  (acoustic Marmousi, Vp)")
    print("=" * 78)

    # ---- per-rung leaderboard (both metric families) ------------------------
    for rung in order:
        rows = rungs[rung]
        rows.sort(key=lambda m: m["_ssim"], reverse=True)
        sk = [m["skip_init"] for m in rows if m.get("skip_init") is not None]
        skmsg = f"  start skip-fraction {np.mean(sk):.2f}" if sk else ""
        print(f"\n--- rung {rung}  ({len(rows)} combos){skmsg} ---")
        print(f"    {'#':>2} {'combo':20}{'SSIM':>7}{'MAPE%':>8}{'dRMS%':>8}  ok")
        for i, m in enumerate(rows[:6], 1):                       # top 6
            ok = "OK" if m.get("losses_finite", True) else "NAN"
            print(f"    {i:2d} {m['_combo']:20}{m['ssim']:7.3f}{m['mape']:8.2f}"
                  f"{m['_drms']:8.1f}  {ok}")
        best_a = max(rows, key=lambda m: m["_drms"])
        print(f"    best by SSIM : {rows[0]['_combo']}  ({rows[0]['ssim']:.3f})")
        print(f"    best by dRMS : {best_a['_combo']}  ({best_a['_drms']:.1f}%)")

    # ---- misfit-level curve (mean SSIM over optimizers) ---------------------
    misfits = sorted({m["_misfit"] for r in order for m in rungs[r]})
    curve = {mi: [] for mi in misfits}
    for rung in order:
        by_mi = defaultdict(list)
        for m in rungs[rung]:
            by_mi[m["_misfit"]].append(m["_ssim"])
        for mi in misfits:
            curve[mi].append(float(np.mean(by_mi[mi])) if by_mi[mi] else np.nan)

    print("\n" + "=" * 78)
    print("MISFIT FLIP CURVE — mean SSIM over optimizers (higher = better)")
    print("=" * 78)
    print(f"  {'misfit':12}" + "".join(f"{r:>9}" for r in order))
    for mi in misfits:
        print(f"  {mi:12}" + "".join(
            f"{v:9.3f}" if np.isfinite(v) else f"{'-':>9}" for v in curve[mi]))
    print(f"  {'WINNER':12}" + "".join(
        f"{max(misfits, key=lambda mi: (curve[mi][j] if np.isfinite(curve[mi][j]) else -9)):>9}"
        for j in range(len(order))))

    # ---- the verdict --------------------------------------------------------
    winners = [max(misfits, key=lambda mi: (curve[mi][j] if np.isfinite(curve[mi][j]) else -9))
               for j in range(len(order))]
    # the hypothesis is specifically "a ROBUST (transport/envelope/kinematic)
    # misfit overtakes the waveform/phase misfits (l2, gc)"
    flip = next((order[j] for j, w in enumerate(winners) if w in ROBUST), None)
    print("\n" + "=" * 78)
    if flip is None:
        print("VERDICT: NO FLIP — a waveform misfit (l2/gc) wins at every rung.")
        print("  The L2->OT hypothesis is NOT supported on DAS strain rate here.")
        print("  => STOP: do not build the adaptive objective; redesign (consider")
        print("     harsher rungs, low-frequency deprivation, or noise).")
    else:
        j = order.index(flip)
        print(f"VERDICT: FLIP AT RUNG '{flip}' — winner becomes '{winners[j]}'.")
        print(f"  L2 leads at {order[:j]} and loses from {flip} onward.")
        print("  => Adaptive lambda is justified. Set the schedule's transition")
        print(f"     from the skip fraction measured at rung {flip}.")
    print("=" * 78)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rung", "combo", "misfit", "optimizer", "ssim", "mape",
                        "drms", "skip_init", "skip_final", "losses_finite",
                        "runtime_h"])
            for rung in order:
                for m in rungs[rung]:
                    w.writerow([rung, m["_combo"], m["_misfit"], m["_optimizer"],
                                m["ssim"], m["mape"], m["_drms"],
                                m.get("skip_init"), m.get("skip_final"),
                                m.get("losses_finite"), m.get("runtime_h")])
        print(f"wrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        xs = range(len(order))
        for mi in misfits:
            style = dict(lw=2.5, marker="o") if mi in ("l2", "gc") else dict(lw=1.4, marker="s", alpha=.85)
            ax.plot(xs, curve[mi], label=mi, **style)
        ax.set_xticks(list(xs)); ax.set_xticklabels(order)
        ax.set_xlabel("starting-model rung (worse ->)")
        ax.set_ylabel("mean SSIM over optimizers")
        ax.set_title("Cycle-skipping flip curve — DAS strain rate (acoustic Marmousi)")
        ax.grid(alpha=.3); ax.legend(ncol=3, fontsize=8)
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
