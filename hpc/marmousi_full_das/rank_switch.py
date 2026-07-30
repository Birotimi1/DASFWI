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

RUNG_DIRS = {"s6": "", "s16": "ladder_s16", "s20": "ladder_s20"}
OPTS = ("adam", "adagrad", "sgd", "adamw", "nadam")
#: weci IS envelope->gc staged (Weci.py composes Misfit_envelope and
#: Misfit_global_correlation), so all four of these are the meaningful controls:
#: the two components alone, the L2 refiner alone, and the staged reference.
#: Mined @s16: envelope 0.240, gc 0.210, l2 0.326, weci 0.451 (mean over opts).
CONTROLS = ("l2", "envelope", "gc", "weci")
#: switch = envelope->l2 (default); switch-gc = envelope->gc, i.e. weci's exact
#: pair under skip-driven timing -> isolates timing from the misfit pair.
SWITCH_ARMS = ("switch", "switch-gc", "fixedk", "fixedk-gc")
MARGIN = 0.05


def load(path):
    f = path / "metrics.json"
    if not f.is_file():
        return None
    m = json.loads(f.read_text())
    return m if m.get("complete", True) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="s16", choices=sorted(RUNG_DIRS))
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    results = Path(args.results) if args.results else \
        Path(__file__).resolve().parents[2] / "results" / "marmousi_full_das"
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
        for arm in ("switch", "switch-gc"):
            sw = d.get(f"{arm}_{o}")
            if sw is None:
                continue
            fk = d.get(f"{arm.replace('switch', 'fixedk')}_{o}")
            win = sw >= wc + MARGIN and sw >= l2
            note = ""
            if fk is not None and win and sw - fk < 0.02:
                note = "  (but ~= fixedk: the diagnostic isn't earning its keep)"
            print(f"{o:8s} {arm:10s}: {sw:.3f} vs weci {wc:.3f} / l2 {l2:.3f}"
                  f" -> {'WIN' if win else 'no win'}{note}")
    print("=" * 74)


if __name__ == "__main__":
    main()
