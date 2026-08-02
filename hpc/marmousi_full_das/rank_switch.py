"""PHASE A verdict: rank the switch arms against the gate's control cells.

The pure-misfit controls (l2 / envelope / weci at every rung x optimizer, 300
iters) were already run by the Phase-1 gate -- they are NOT re-run. This script
puts them in one table with the new switch/fixedk cells and applies the win
criterion from the verified design:

    WIN = switch >= weci-only + 0.05 SSIM   AND   switch >= l2-only

    python hpc/marmousi_full_das/rank_switch.py [--rung s16] [--results DIR]
"""
import argparse
import json
from pathlib import Path

RUNG_DIRS = {"s6": "", "s16": "ladder_s16", "s20": "ladder_s20",
             "routeb": "switch_routeb"}
OPTS = ("adam", "adagrad", "sgd", "adamw", "nadam")
#: weci IS envelope->gc staged (Weci.py composes Misfit_envelope and
#: Misfit_global_correlation), so all four of these are the meaningful controls:
#: the two components alone, the L2 refiner alone, and the staged reference.
#: Mined @s16: envelope 0.240, gc 0.210, l2 0.326, weci 0.451 (mean over opts).
CONTROLS = ("l2", "envelope", "gc", "weci")
#: switch = envelope->l2 (default); switch-gc = envelope->gc, i.e. weci's exact
#: pair under skip-driven timing -> isolates timing from the misfit pair.
#: ladder = the 3-stage envelope->gc->l2 generalisation (StagedMisfit+StageLadder)
SWITCH_ARMS = ("switch", "switch-gc", "ladder", "fixedk", "fixedk-gc")
MARGIN = 0.05


def load(path):
    f = path / "metrics.json"
    if not f.is_file():
        return None
    m = json.loads(f.read_text())
    return m if m.get("complete", True) else None


def rank_routeb(results):
    """Route B results: step 2 (converged starter) and step 3 (partial starter)
    share one directory and are told apart by the starter name in the tag. The
    controls (l2 / envelope / fixedk) are IN the group -- there are no external
    gate cells to compare against, because a Route B start has no rung."""
    root = results / "switch_routeb"
    if not root.is_dir():
        return print(f"no Route B results in {root}")
    starters = sorted({d.name for d in (results / "starter").glob("i*")}) \
        if (results / "starter").is_dir() else []
    cells = []
    for d in sorted(root.iterdir()):
        m = load(d)
        if not m or d.name.startswith("smoke_"):
            continue
        st = next((s for s in starters if s in d.name), "?")
        arm = d.name.split("_")[0]
        cells.append((st, arm, m.get("optimizer", "?"), m.get("ssim", 0.0),
                      m.get("mape", 0.0), m.get("handbacks"), m.get("reentries")))
    if not cells:
        return print(f"no complete Route B cells in {root}")
    for st in sorted({c[0] for c in cells}):
        grp = [c for c in cells if c[0] == st]
        print("=" * 74)
        print(f"ROUTE B  starter={st}   ({len(grp)} cells)")
        print("=" * 74)
        print(f"{'arm':14s} {'optimizer':10s} {'SSIM':>6s} {'MAPE%':>7s} "
              f"{'handovr':>8s} {'reentr':>7s}")
        for _, arm, o, ss, mp, hb, re_ in sorted(grp, key=lambda c: -c[3]):
            flag = "  <-- REENTRY" if isinstance(re_, int) and re_ > 0 else ""
            print(f"{arm:14s} {o:10s} {ss:6.3f} {mp:7.2f} "
                  f"{str(hb if hb is not None else '-'):>8s} "
                  f"{str(re_ if re_ is not None else '-'):>7s}{flag}")
        # the comparison that matters: switch vs the l2 control, per optimizer
        d_ = {(a, o): ss for _, a, o, ss, *_ in grp}
        print("-" * 74)
        for o in sorted({c[2] for c in grp}):
            l2 = d_.get(("l2", o))
            if l2 is None:
                continue
            for arm in ("switch", "switch-gc", "fixedk", "envelope"):
                v = d_.get((arm, o))
                if v is None:
                    continue
                print(f"  {o:8s} {arm:10s} {v:.3f} vs l2 {l2:.3f} -> "
                      f"{'+' if v >= l2 else ''}{v - l2:.3f}"
                      + ("  BEATS L2" if v >= l2 + MARGIN else
                         "  ~= L2" if abs(v - l2) < MARGIN else "  loses"))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="s16", choices=sorted(RUNG_DIRS))
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    results = Path(args.results) if args.results else \
        Path(__file__).resolve().parents[2] / "results" / "marmousi_full_das"
    if args.rung == "routeb":
        return rank_routeb(results)
    gate = results / RUNG_DIRS[args.rung] if RUNG_DIRS[args.rung] else results
    switch = results / f"switch_{args.rung}"

    print("=" * 74)
    print(f"PHASE A -- staged envelope->L2 switch vs controls @ {args.rung}")
    print("=" * 74)
    print(f"{'arm':22s} {'SSIM':>6s} {'MAPE%':>7s} {'handovr':>8s} {'reentr':>7s}")

    rows = []
    for mf in CONTROLS:                       # gate controls (already computed)
        for o in OPTS:
            m = load(gate / f"{mf}_{o}")
            if m:
                rows.append((f"{mf}_{o} (gate)", m["ssim"], m["mape"], "", ""))
    for arm in SWITCH_ARMS:                   # the new arms
        for o in OPTS:
            m = load(switch / f"{arm}_{o}")
            if m:
                rows.append((f"{arm}_{o}", m["ssim"], m["mape"],
                             m.get("handbacks", ""), m.get("reentries", "")))

    for tag, ssim, mape, hb, re_ in sorted(rows, key=lambda r: -r[1]):
        flag = "  <-- REENTRY" if isinstance(re_, int) and re_ > 0 else ""
        print(f"{tag:22s} {ssim:6.3f} {mape:7.2f} {str(hb):>8s} {str(re_):>7s}{flag}")

    # ---- win criterion, per optimizer ----------------------------------------
    # The bar is weci (the staged envelope->gc reference), NOT l2: weci already
    # scores 0.451 at s16 by staging internally, so beating l2 alone proves
    # nothing about the switch. WIN = beat the staged reference by MARGIN and
    # beat the plain refiner.
    print("-" * 74)
    d = {t.split(" ")[0]: s for t, s, *_ in rows}
    for o in OPTS:
        l2, wc = d.get(f"l2_{o}"), d.get(f"weci_{o}")
        if l2 is None or wc is None:
            continue
        for arm in ("switch", "switch-gc", "ladder"):
            sw = d.get(f"{arm}_{o}")
            if sw is None:
                continue
            # explicit map: str.replace would leave 'ladder' unchanged and then
            # compare the arm against ITSELF (always firing the false warning)
            fk_key = {"switch": "fixedk", "switch-gc": "fixedk-gc"}.get(arm)
            fk = d.get(f"{fk_key}_{o}") if fk_key else None
            win = sw >= wc + MARGIN and sw >= l2
            note = ""
            if fk is not None and win and sw - fk < 0.02:
                note = "  (but ~= fixedk: the diagnostic isn't earning its keep)"
            print(f"{o:8s} {arm:10s}: {sw:.3f} vs weci {wc:.3f} / l2 {l2:.3f}"
                  f" -> {'WIN' if win else 'no win'}{note}")
    print("=" * 74)


if __name__ == "__main__":
    main()
