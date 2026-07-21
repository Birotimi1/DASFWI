#!/usr/bin/env python3
"""Rank the full-Marmousi2 DAS (acoustic Vp) campaign under TWO metric families,
separately, so their behaviour can be compared:

  STRUCTURAL  ranked by SSIM (Wang 2004; higher=better), MAPE shown alongside.
  AMPLITUDE   ranked by dRMS% (RMS error removed; higher=better), update-corr
              shown alongside.

SSIM/MAPE are computed from setup.npz (vp_true) + each combo's iter_vp.npz;
RMS/dRMS/update-corr from metrics.json. Scores a completed OR running campaign
without re-running.

    python hpc/marmousi_full_das/rank_campaign.py [--results DIR] [--csv FILE]
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
        if "ssim" not in m or "mape" not in m:
            if vp_true is None:
                continue
            vp = _final_vp(args.results, tag)
            if vp is None:
                continue
            sc = model_scores(vp_true, vp)
            m["ssim"], m["mape"] = sc["ssim"], sc["mape"]
        m["_drms"] = (100.0 * (m["rms_init"] - m["rms_final"]) / m["rms_init"]
                      if m.get("rms_init", 0) > 0 else 0.0)
        m["_ssim"] = m["ssim"] if m.get("losses_finite", True) else -1.0
        rows.append(m)

    nfin = sum(1 for m in rows if m.get("losses_finite", True))
    print(f"============ acoustic Vp campaign -- dual ranking "
          f"({len(rows)}/45 scored, {nfin} finite) ============")

    # ---- STRUCTURAL: by SSIM ----
    rows.sort(key=lambda m: m["_ssim"], reverse=True)
    print("\n########## STRUCTURAL -- ranked by SSIM (higher=better); MAPE% lower=better")
    h = f"{'#':>2} {'combo':20}{'SSIM':>7}{'MAPE%':>8}{'h':>7}  ok"
    print(h); print("-" * len(h))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", True) else "NAN"
        print(f"{i:2d} {m['tag']:20}{m['ssim']:7.3f}{m['mape']:8.2f}"
              f"{m.get('runtime_h',0):7.2f}  {ok}")
    b = rows[0]
    print(f"best (SSIM): {b['tag']}  SSIM {b['ssim']:.3f}  MAPE {b['mape']:.2f}%")

    # ---- AMPLITUDE: by dRMS ----
    rows.sort(key=lambda m: (m["_drms"] if m.get("losses_finite", True) else -1e9),
              reverse=True)
    print("\n########## AMPLITUDE -- ranked by dRMS% (RMS removed; higher=better)")
    h = (f"{'#':>2} {'combo':20}{'rms_init':>9}{'rms_final':>10}{'dRMS%':>7}"
         f"{'upd_corr':>9}{'h':>7}  ok")
    print(h); print("-" * len(h))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", True) else "NAN"
        print(f"{i:2d} {m['tag']:20}{m.get('rms_init',0):9.1f}"
              f"{m.get('rms_final',0):10.1f}{m['_drms']:7.1f}"
              f"{m.get('update_corr',0):9.3f}{m.get('runtime_h',0):7.2f}  {ok}")
    b = rows[0]
    print(f"best (dRMS): {b['tag']}  dRMS {b['_drms']:.1f}%  "
          f"update_corr {b.get('update_corr',0):.3f}")

    if args.csv:
        import csv
        cols = ["tag", "ssim", "mape", "_drms", "update_corr", "rms_init",
                "rms_final", "runtime_h", "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.lstrip("_") for c in cols])
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
