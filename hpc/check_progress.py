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
    """Route B starters: the acceptance number is skip@starter vs skip@1-D."""
    for kind, root in ROOTS.items():
        d = root / "starter"
        if not d.is_dir():
            continue
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
