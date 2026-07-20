#!/usr/bin/env python3
"""Rank the elastic Vp/Vs DAS campaign, with the illumination-precond A/B.

Reads results/elastic_full_das/<misfit>_<optimizer>_<illum|noillum>/metrics.json.
Per parameter p in {vp, vs} (density is held constant, not inverted):
    score_p = update_corr_p * max(0, 1 - rms_final_p / rms_init_p)
Combined ranking = mean of (vp, vs). Also reports DEEP-half dRMS (where the
illumination preconditioner acts) and an A/B summary: for each base combo,
illum vs off, and whether illumination improved deep recovery.

    python hpc/elastic_full_das/rank_campaign.py
    python hpc/elastic_full_das/rank_campaign.py --csv ranking.csv
"""
import argparse
import glob
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT = os.environ.get(
    "DASFWI_RESULTS", os.path.join(_REPO, "results", "elastic_full_das"))


def _pscore(m, p):
    if not m.get("losses_finite", False):
        return 0.0
    ri, rf = m.get(f"rms_init_{p}", 0.0), m.get(f"rms_final_{p}", 0.0)
    frac = max(0.0, 1.0 - rf / ri) if ri > 0 else 0.0
    return m.get(f"update_corr_{p}", 0.0) * frac


def _drms(m, p, deep=False):
    key = "rms_%s_deep_%s" if deep else "rms_%s_%s"
    ri, rf = m.get(key % ("init", p), 0.0), m.get(key % ("final", p), 0.0)
    return 100.0 * (ri - rf) / ri if ri > 0 else 0.0


def _load(results):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "*", "metrics.json"))):
        try:
            m = json.load(open(f))
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {f}: {e}", file=sys.stderr)
            continue
        m["_precond"] = m.get("precond") or ("illum" if m.get("tag", "").endswith("illum")
                                             and not m.get("tag", "").endswith("noillum")
                                             else "off")
        m["_base"] = f"{m.get('misfit','?')}_{m.get('optimizer','?')}"
        m["_score"] = 0.5 * (_pscore(m, "vp") + _pscore(m, "vs"))
        for p in ("vp", "vs"):
            m[f"_d_{p}"] = _drms(m, p)
            m[f"_dd_{p}"] = _drms(m, p, deep=True)      # deep-half dRMS
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = _load(args.results)
    if not rows:
        print(f"no metrics.json under {args.results}", file=sys.stderr)
        sys.exit(1)
    rows.sort(key=lambda m: m["_score"], reverse=True)

    hdr = (f"{'#':>2} {'combo':16}{'prec':>6}{'score':>7} | "
           f"{'vp dRMS%':>9}{'deep':>6} | {'vs dRMS%':>9}{'deep':>6} |"
           f"{'h':>5} ok")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", False) else "NAN"
        print(f"{i:2d} {m['_base']:16}{m['_precond']:>6}{m['_score']:7.3f} | "
              f"{m['_d_vp']:9.1f}{m['_dd_vp']:6.0f} | "
              f"{m['_d_vs']:9.1f}{m['_dd_vs']:6.0f} |"
              f"{m.get('runtime_h', 0):5.2f} {ok}")

    # --- A/B: illum vs off per base combo, on DEEP velocity recovery ----------
    by_base = {}
    for m in rows:
        by_base.setdefault(m["_base"], {})[m["_precond"]] = m
    pairs = [(b, d["illum"], d["off"]) for b, d in by_base.items()
             if "illum" in d and "off" in d]
    print(f"\n=== illumination A/B on DEEP (vp+vs) recovery "
          f"({len(pairs)} paired combos) ===")
    if pairs:
        gains = []
        print(f"{'combo':16}{'deep illum':>12}{'deep off':>10}{'gain(pts)':>11}")
        # deep vp+vs dRMS mean per side
        def deep_vpvs(m):
            return 0.5 * (m["_dd_vp"] + m["_dd_vs"])
        for b, mi, mo in sorted(pairs, key=lambda t: deep_vpvs(t[1]) - deep_vpvs(t[2]),
                                reverse=True):
            gi, go = deep_vpvs(mi), deep_vpvs(mo)
            gains.append(gi - go)
            print(f"{b:16}{gi:12.1f}{go:10.1f}{gi - go:+11.1f}")
        improved = sum(1 for g in gains if g > 0)
        print(f"\nillumination improved DEEP vp+vs recovery in "
              f"{improved}/{len(gains)} combos; "
              f"mean gain {sum(gains)/len(gains):+.1f} pts dRMS")

    n_fin = sum(1 for m in rows if m.get("losses_finite", False))
    print(f"\n{len(rows)}/90 runs complete ({n_fin} finite)")
    if rows:
        b = rows[0]
        print(f"best overall (vp+vs): {b['_base']} [{b['_precond']}]  "
              f"score={b['_score']:.3f}")

    if args.csv:
        import csv
        cols = ["_base", "_precond", "_score", "_d_vp", "_dd_vp", "_d_vs",
                "_dd_vs", "update_corr_vp", "update_corr_vs", "runtime_h",
                "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.lstrip("_") for c in cols])
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
