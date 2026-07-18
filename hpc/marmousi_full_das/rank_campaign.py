#!/usr/bin/env python3
"""Rank the full-Marmousi2 DAS campaign by recovery quality.

Reads every results/marmousi_full_das/<misfit>_<optimizer>/metrics.json and
prints a table sorted by a composite deployment score:

    score = update_corr * max(0, 1 - rms_final / rms_init)

i.e. reward BOTH structural alignment with the true update (update_corr) AND
the fraction of velocity RMS error removed. A diverged/non-finite run scores 0.

Usage (from the DASFWI repo root, no args needed):
    python hpc/marmousi_full_das/rank_campaign.py
    python hpc/marmousi_full_das/rank_campaign.py --results /path/to/results_dir
    python hpc/marmousi_full_das/rank_campaign.py --csv ranking.csv
"""
import argparse
import glob
import json
import os
import sys

# default results dir mirrors common.OUT_ROOT (DASFWI_RESULTS or repo/results/...)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT = os.environ.get(
    "DASFWI_RESULTS", os.path.join(_REPO, "results", "marmousi_full_das"))


def score(m):
    if not m.get("losses_finite", False):
        return 0.0
    ri, rf = m.get("rms_init", 0.0), m.get("rms_final", 0.0)
    frac = max(0.0, 1.0 - rf / ri) if ri > 0 else 0.0
    return m.get("update_corr", 0.0) * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT,
                    help="campaign results dir (default: $DASFWI_RESULTS or repo/results/marmousi_full_das)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV")
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
        m["_score"] = score(m)
        m["_drms"] = (100.0 * (m["rms_init"] - m["rms_final"]) / m["rms_init"]
                      if m.get("rms_init", 0) > 0 else 0.0)
        rows.append(m)
    rows.sort(key=lambda m: m["_score"], reverse=True)

    hdr = f"{'#':>2} {'combo':20}{'rms_init':>9}{'rms_final':>10}{'dRMS%':>7}{'upd_corr':>9}{'score':>7}{'h':>6}  ok"
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", False) else "NAN"
        print(f"{i:2d} {m['tag']:20}{m['rms_init']:9.1f}{m['rms_final']:10.1f}"
              f"{m['_drms']:7.1f}{m.get('update_corr', 0):9.3f}{m['_score']:7.3f}"
              f"{m.get('runtime_h', 0):6.2f}  {ok}")

    nfin = sum(1 for m in rows if m.get("losses_finite", False))
    print(f"\n{len(rows)}/45 complete  ({nfin} finite)")
    if rows:
        best = rows[0]
        print(f"best: {best['tag']}  score={best['_score']:.3f}  "
              f"dRMS={best['_drms']:.1f}%  update_corr={best.get('update_corr', 0):.3f}")

    if args.csv:
        import csv
        cols = ["tag", "rms_init", "rms_final", "_drms", "update_corr",
                "_score", "runtime_h", "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
