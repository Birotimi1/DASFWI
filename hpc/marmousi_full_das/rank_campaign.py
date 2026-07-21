#!/usr/bin/env python3
"""Rank the full-Marmousi2 DAS campaign by SSIM + MAPE (Liu's metrics).

Following Liu's paper, model recovery is scored with:
    SSIM  - structural similarity of inverted vs true Vp (Wang 2004); higher
            better, 1 = identical.  <-- primary ranking metric
    MAPE  - mean absolute percentage error (Hyndman & Koehler 2006); lower better.
Both are computed HERE from setup.npz (vp_true) + each combo's iter_vp.npz
(recovered = last cached iteration), so a completed OR running campaign is
scored without re-running. dRMS%/update-corr are still shown for continuity.

    python hpc/marmousi_full_das/rank_campaign.py
    python hpc/marmousi_full_das/rank_campaign.py --results /path --csv ranking.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from inversion.metrics import model_scores                       # noqa: E402

_DEFAULT = os.environ.get(
    "DASFWI_RESULTS", os.path.join(_REPO, "results", "marmousi_full_das"))


def _final_vp(results, tag):
    f = os.path.join(results, tag, "iter_vp.npz")
    if not os.path.isfile(f):
        return None
    a = np.load(f)["data"]
    return np.asarray(a[-1] if a.ndim == 3 else a, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    setup_f = os.path.join(args.results, "setup.npz")
    vp_true = (np.asarray(np.load(setup_f)["vp_true"], float)
               if os.path.isfile(setup_f) else None)

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
        tag = m.get("tag") or os.path.basename(os.path.dirname(f))
        # prefer SSIM/MAPE from metrics.json; else compute from arrays
        if "ssim" in m and "mape" in m:
            pass
        elif vp_true is not None:
            vp = _final_vp(args.results, tag)
            if vp is None:
                continue
            sc = model_scores(vp_true, vp)
            m["ssim"], m["mape"] = sc["ssim"], sc["mape"]
        else:
            continue
        m["_drms"] = (100.0 * (m["rms_init"] - m["rms_final"]) / m["rms_init"]
                      if m.get("rms_init", 0) > 0 else 0.0)
        m["_ssim"] = m["ssim"] if m.get("losses_finite", True) else -1.0
        rows.append(m)
    rows.sort(key=lambda m: m["_ssim"], reverse=True)

    hdr = (f"{'#':>2} {'combo':20}{'SSIM':>7}{'MAPE%':>8}{'dRMS%':>7}"
           f"{'upd_corr':>9}{'h':>6}  ok")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", True) else "NAN"
        print(f"{i:2d} {m['tag']:20}{m['ssim']:7.3f}{m['mape']:8.2f}{m['_drms']:7.1f}"
              f"{m.get('update_corr', 0):9.3f}{m.get('runtime_h', 0):6.2f}  {ok}")

    nfin = sum(1 for m in rows if m.get("losses_finite", True))
    print(f"\n{len(rows)}/45 scored  ({nfin} finite)")
    if rows:
        b = rows[0]
        print(f"best (SSIM): {b['tag']}  SSIM={b['ssim']:.3f}  "
              f"MAPE={b['mape']:.2f}%  dRMS={b['_drms']:.1f}%")

    if args.csv:
        import csv
        cols = ["tag", "ssim", "mape", "_drms", "update_corr", "rms_init",
                "rms_final", "runtime_h", "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.lstrip("_") for c in cols])
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
