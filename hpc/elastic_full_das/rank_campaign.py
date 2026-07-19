#!/usr/bin/env python3
"""Rank the elastic 3-parameter DAS campaign by recovery quality.

Reads results/elastic_full_das/<misfit>_<optimizer>/metrics.json and prints a
table. Per parameter p in {vp, vs, rho}:
    score_p = update_corr_p * max(0, 1 - rms_final_p / rms_init_p)
The combined ranking uses the mean of the WELL-CONSTRAINED velocities
(vp, vs); density is weakly constrained in FWI so it is reported but does not
drive the ranking. A non-finite (diverged) run scores 0.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "*", "metrics.json")))
    if not files:
        print(f"no metrics.json under {args.results}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in files:
        try:
            m = json.load(open(f))
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {f}: {e}", file=sys.stderr)
            continue
        for p in ("vp", "vs", "rho"):
            m[f"_s_{p}"] = _pscore(m, p)
            ri = m.get(f"rms_init_{p}", 0)
            m[f"_d_{p}"] = (100.0 * (ri - m.get(f"rms_final_{p}", 0)) / ri
                            if ri > 0 else 0.0)
        m["_score"] = 0.5 * (m["_s_vp"] + m["_s_vs"])       # velocity-driven
        rows.append(m)
    rows.sort(key=lambda m: m["_score"], reverse=True)

    hdr = (f"{'#':>2} {'combo':20}{'score':>7} | "
           f"{'vp:dRMS%':>9}{'corr':>6}{'sc':>6} | "
           f"{'vs:dRMS%':>9}{'corr':>6}{'sc':>6} | "
           f"{'rho:dRMS%':>10}{'corr':>6}{'sc':>6} | {'h':>5} ok")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", False) else "NAN"
        print(f"{i:2d} {m['tag']:20}{m['_score']:7.3f} | "
              f"{m['_d_vp']:9.1f}{m.get('update_corr_vp', 0):6.2f}{m['_s_vp']:6.2f} | "
              f"{m['_d_vs']:9.1f}{m.get('update_corr_vs', 0):6.2f}{m['_s_vs']:6.2f} | "
              f"{m['_d_rho']:10.1f}{m.get('update_corr_rho', 0):6.2f}{m['_s_rho']:6.2f} | "
              f"{m.get('runtime_h', 0):5.2f} {ok}")

    nfin = sum(1 for m in rows if m.get("losses_finite", False))
    print(f"\n{len(rows)}/45 complete  ({nfin} finite)")
    if rows:
        b = rows[0]
        print(f"best (vp+vs): {b['tag']}  score={b['_score']:.3f}  "
              f"vp dRMS {b['_d_vp']:.0f}% / vs dRMS {b['_d_vs']:.0f}% / "
              f"rho dRMS {b['_d_rho']:.0f}%")

    if args.csv:
        import csv
        cols = ["tag", "_score", "_d_vp", "update_corr_vp", "_d_vs",
                "update_corr_vs", "_d_rho", "update_corr_rho", "runtime_h",
                "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
