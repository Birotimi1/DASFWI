"""Live progress + results for any DASFWI run, finished or mid-flight.

Every driver checkpoints, so a running job already has numbers on disk. This
reads whatever is there and says what it means -- including the ACCEPTANCE
NUMBER for a Route B starter (does the traveltime leg actually reduce the skip
fraction below the blind 1-D ramp?).

    python hpc/check_progress.py                # everything it can find
    python hpc/check_progress.py --what starter # just the Route B starters
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOTS = {"acoustic": REPO / "results" / "marmousi_full_das",
         "elastic": REPO / "results" / "elastic_full_das"}


def _loss_trend(d):
    f = d / "iter_loss.npz"
    if not f.is_file():
        return ""
    a = np.asarray(np.load(f)["data"]).ravel()
    if a.size < 2:
        return f"  loss n={a.size}"
    drop = 100 * (a[0] - a[-1]) / max(abs(a[0]), 1e-30)
    tail = "falling" if a[-1] < a[max(0, len(a) - 6)] else "FLAT/rising"
    return f"  loss {a[0]:.4g} -> {a[-1]:.4g} ({drop:+.1f}%), last-5 {tail}"


def starters(quiet=False):
    """Route B starters: the acceptance number is skip@starter vs skip@1-D.

    Ranks a whole comparison matrix (misfit x optimizer x iterations) so the
    method can be judged on its best configuration, not on one under-stepped
    run."""
    for kind, root in ROOTS.items():
        sroot = root / "starter"
        if not sroot.is_dir():
            continue
        cells = sorted(sroot.glob("*/starter_metrics.json"))
        if (sroot / "starter_metrics.json").is_file():        # legacy flat layout
            cells.append(sroot / "starter_metrics.json")
        if cells:                                 # one cell or a matrix -> rank
            print(f"\n=== {kind} Route B starter MATRIX ({len(cells)} cells) ===")
            print(f"  {'cell':34s} {'skip 1-D':>9s} {'skip start':>11s} "
                  f"{'d skip':>7s} {'SSIM 1-D':>9s} {'SSIM start':>11s} {'verdict':>9s}")
            rows = []
            for f in cells:
                m = json.loads(f.read_text())
                s1, s2 = m.get("skip_1d"), m.get("skip_starter")
                # acoustic writes ssim_1d/ssim_starter, elastic ssim_1d_vp/..._vp
                a1 = m.get("ssim_1d_vp", m.get("ssim_1d"))
                a2 = m.get("ssim_starter_vp", m.get("ssim_starter"))
                ok = (isinstance(s1, (int, float)) and isinstance(s2, (int, float))
                      and isinstance(a1, (int, float)) and isinstance(a2, (int, float)))
                good = ok and s2 < s1 - 0.01 and a2 > a1 + 0.01
                rows.append((s2 if ok else 9e9, f.parent.name, s1, s2, a1, a2, good, ok))
            for _, name, s1, s2, a1, a2, good, ok in sorted(rows):
                if not ok:
                    print(f"  {name:34s} (incomplete)"); continue
                print(f"  {name:34s} {s1:9.3f} {s2:11.3f} {s2-s1:+7.3f} "
                      f"{a1:9.3f} {a2:11.3f} {'USABLE' if good else '  --':>9s}")
            # still-running cells have no starter_metrics.json yet
            for pj in sorted(sroot.glob("*/vp_partial.npz")):
                if not (pj.parent / "starter_metrics.json").is_file():
                    z = np.load(pj)
                    print(f"  {pj.parent.name:34s} RUNNING "
                          f"{int(z['iterations_done'])} iters done")
            best = [r for r in sorted(rows) if r[6]]
            # A verdict needs REAL cells: a 2-iteration smoke cannot move the
            # model, and an unreadable cell says nothing. Declaring the method
            # dead on those would be wrong.
            def _iters(name):
                m = re.match(r"i(\d+)", name)
                return int(m.group(1)) if m else 10**6      # legacy flat = real
            MIN_REAL = 20        # fewer iterations cannot move the model at all
            real = [r for r in rows
                    if r[7] and "smoke" not in r[1] and _iters(r[1]) >= MIN_REAL]
            if best:
                print(f"\n  BEST: {best[0][1]}")
            elif not real:
                print("\n  NO VERDICT YET -- only smoke/incomplete cells here. "
                      "Run the real matrix (>=100 iterations) before judging.")
            else:
                print(f"\n  NO CELL IS USABLE across {len(real)} real cell(s) -- "
                      "wave-equation xcorr did not beat the 1-D ramp in ANY "
                      "configuration; the eikonal fallback is justified")
            continue
        d = sroot
        print(f"\n=== {kind} Route B starter  ({d}) ===")
        mf = d / "starter_metrics.json"
        if mf.is_file():
            m = json.loads(mf.read_text())
            s1, s2 = m.get("skip_1d"), m.get("skip_starter")
            print(f"  DONE  {m.get('iterations')} iters @ {m.get('band')} Hz, "
                  f"{m.get('runtime_h')} h")
            if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                verdict = ("STARTER IS BETTER -- use it" if s2 < s1 - 0.01 else
                           "NO BETTER THAN THE 1-D RAMP -- the traveltime leg is "
                           "not earning its keep; do NOT build on it")
                print(f"  skip:  1-D {s1:.3f}  ->  starter {s2:.3f}   {verdict}")
                print(f"         (switch thresholds: robust above 0.58, hands "
                      f"over below 0.45 -> this start is "
                      f"{'IN the skip regime' if s2 >= 0.58 else 'in the transition band' if s2 > 0.45 else 'essentially aligned'})")
            print(f"  SSIM vp: 1-D {m.get('ssim_1d_vp'):.3f} -> starter "
                  f"{m.get('ssim_starter_vp'):.3f}   vs seed "
                  f"{m.get('ssim_starter_vs'):.3f}")
        else:
            p = d / "vp_partial.npz"
            if p.is_file():
                z = np.load(p)
                print(f"  RUNNING: {int(z['iterations_done'])} iterations done")
            else:
                print("  no checkpoint yet (first one lands at iteration 10)")
        print(_loss_trend(d))


def runs(what):
    """Inversion cells (pipeline / switch / adaptive), complete or partial."""
    pats = {"pipeline": ROOTS["elastic"] / "pipeline",
            "switch": None, "adaptive": ROOTS["acoustic"] / "adaptive"}
    for name, root in pats.items():
        dirs = []
        if name == "switch":
            dirs = sorted(ROOTS["acoustic"].glob("switch_*/*"))
        elif root and root.is_dir():
            dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if what not in (None, name) or not dirs:
            continue
        # label switch cells with their RUNG (switch_s16/switch_adam), otherwise
        # s6/s16/s20 cells are indistinguishable in the listing
        def _label(d):
            return (f"{d.parent.name.replace('switch_', '')}/{d.name}"
                    if name == "switch" else d.name)

        empty = [d for d in dirs if not (d / "metrics.json").is_file()]
        dirs = [d for d in dirs if (d / "metrics.json").is_file()]
        print(f"\n=== {name}  ({len(dirs)} cells"
              + (f", {len(empty)} empty dirs from failed/older runs" if empty else "")
              + ") ===")
        for d in sorted(dirs, key=_label):
            m = json.loads((d / "metrics.json").read_text())
            done = m.get("iterations_done", "?")
            tot = m.get("iterations", "?")
            state = "DONE" if m.get("complete") else f"{done}/{tot}"
            sc = (f"vp {m['ssim_vp']:.3f} vs {m['ssim_vs']:.3f}"
                  if "ssim_vp" in m else
                  f"ssim {m.get('ssim', float('nan')):.3f}")
            init = m.get("ssim_init_vp")
            warn = ""
            done_n = m.get("iterations_done") or 0
            if (isinstance(init, (int, float)) and m.get("ssim_vp", 1) < init
                    and done_n >= 10):        # ignore 2-iteration smokes
                warn = f"  *** DIVERGED (init {init:.3f}) ***"
            print(f"  {_label(d):46s} {state:>9s}  {sc}{warn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default=None,
                    choices=("starter", "pipeline", "switch", "adaptive"))
    args = ap.parse_args()
    if args.what in (None, "starter"):
        starters()
    if args.what != "starter":
        runs(args.what)
    print()


if __name__ == "__main__":
    main()
