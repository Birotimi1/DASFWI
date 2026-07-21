#!/usr/bin/env python3
"""Rank the elastic Vp/Vs DAS campaign under TWO metric families, separately, so
their behaviour can be compared:

  STRUCTURAL  ranked by SSIM (Wang 2004; higher=better, 1=identical), with MAPE
              (Hyndman & Koehler 2006; % error, lower=better) shown alongside.
  AMPLITUDE   ranked by dRMS% (RMS error removed; higher=better), with the
              update-correlation shown alongside.

SSIM/MAPE are computed here from setup.npz (vp_true/vs_true) + each combo's
iter_vp.npz / iter_vs.npz; RMS/dRMS/update-corr come from metrics.json. A running
or finished campaign is scored with no re-run. The illumination A/B is reported
under BOTH families (deep SSIM and deep dRMS).

    python hpc/elastic_full_das/rank_campaign.py
    python hpc/elastic_full_das/rank_campaign.py --csv ranking.csv
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
    "DASFWI_RESULTS", os.path.join(_REPO, "results", "elastic_full_das"))


def _final(results, tag, param):
    f = os.path.join(results, tag, f"iter_{param}.npz")
    if not os.path.isfile(f):
        return None
    a = np.load(f)["data"]
    return np.asarray(a[-1] if a.ndim == 3 else a, dtype=float)


def _drms(m, p, deep=False):
    k = "rms_%s_deep_%s" if deep else "rms_%s_%s"
    ri, rf = m.get(k % ("init", p), 0.0), m.get(k % ("final", p), 0.0)
    return 100.0 * (ri - rf) / ri if ri > 0 else 0.0


def _load(results):
    setup_f = os.path.join(results, "setup.npz")
    if not os.path.isfile(setup_f):
        print(f"no setup.npz under {results}", file=sys.stderr)
        sys.exit(1)
    s = np.load(setup_f)
    truth = {"vp": np.asarray(s["vp_true"], float),
             "vs": np.asarray(s["vs_true"], float)}
    deep = slice(truth["vp"].shape[0] // 2, truth["vp"].shape[0])

    rows = []
    for mf in sorted(glob.glob(os.path.join(results, "*", "metrics.json"))):
        try:
            m = json.load(open(mf))
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {mf}: {e}", file=sys.stderr)
            continue
        tag = os.path.basename(os.path.dirname(mf))
        ok = True
        for p in ("vp", "vs"):
            inv = _final(results, tag, p)
            if inv is None:
                ok = False
                break
            sc = model_scores(truth[p], inv, deep=deep)
            m[f"ssim_{p}"], m[f"mape_{p}"] = sc["ssim"], sc["mape"]
            m[f"ssim_deep_{p}"] = sc["ssim_deep"]
            m[f"drms_{p}"], m[f"drms_deep_{p}"] = _drms(m, p), _drms(m, p, deep=True)
        if not ok:
            continue
        finite = m.get("losses_finite", True)
        m["_base"] = f"{m.get('misfit','?')}_{m.get('optimizer','?')}"
        m["_precond"] = m.get("precond") or ("illum" if tag.endswith("_illum") else "off")
        m["_ssim"] = 0.5 * (m["ssim_vp"] + m["ssim_vs"]) if finite else -1.0
        m["_ssim_deep"] = 0.5 * (m["ssim_deep_vp"] + m["ssim_deep_vs"])
        m["_drms"] = 0.5 * (m["drms_vp"] + m["drms_vs"]) if finite else -1e9
        m["_drms_deep"] = 0.5 * (m["drms_deep_vp"] + m["drms_deep_vs"])
        rows.append(m)
    return rows


def _ab(rows, key, label, fmt="{:+.3f}"):
    by_base = {}
    for m in rows:
        by_base.setdefault(m["_base"], {})[m["_precond"]] = m
    pairs = [(b, d["illum"], d["off"]) for b, d in by_base.items()
             if "illum" in d and "off" in d]
    print(f"\n=== illumination A/B -- {label} ({len(pairs)} paired combos) ===")
    if not pairs:
        return
    gains = []
    for b, mi, mo in sorted(pairs, key=lambda t: t[1][key] - t[2][key], reverse=True):
        g = mi[key] - mo[key]
        gains.append(g)
        print(f"  {b:16} illum {mi[key]:8.3f}  off {mo[key]:8.3f}  gain {fmt.format(g)}")
    imp = sum(1 for g in gains if g > 0)
    print(f"  -> illumination helped in {imp}/{len(gains)} combos; "
          f"mean gain {fmt.format(sum(gains)/len(gains))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_DEFAULT)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = _load(args.results)
    if not rows:
        print(f"no completed combos with iter_vp/vs.npz under {args.results}",
              file=sys.stderr)
        sys.exit(1)
    nfin = sum(1 for m in rows if m.get("losses_finite", True))
    print(f"==================== elastic Vp/Vs campaign -- dual ranking "
          f"({len(rows)}/90 scored, {nfin} finite) ====================")

    # ---- STRUCTURAL: by SSIM ------------------------------------------------
    rows.sort(key=lambda m: m["_ssim"], reverse=True)
    print("\n########## STRUCTURAL -- ranked by SSIM (higher=better); MAPE% lower=better")
    h = (f"{'#':>2} {'combo':16}{'prec':>6} | {'SSIM vp':>8}{'vs':>7}{'deep':>7}"
         f" | {'MAPE% vp':>9}{'vs':>7} |{'h':>5} ok")
    print(h); print("-" * len(h))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", True) else "NAN"
        print(f"{i:2d} {m['_base']:16}{m['_precond']:>6} | "
              f"{m['ssim_vp']:8.3f}{m['ssim_vs']:7.3f}{m['_ssim_deep']:7.3f} | "
              f"{m['mape_vp']:9.2f}{m['mape_vs']:7.2f} |{m.get('runtime_h',0):5.2f} {ok}")
    b = rows[0]
    print(f"best (SSIM): {b['_base']} [{b['_precond']}]  "
          f"SSIM {b['_ssim']:.3f}  MAPE vp {b['mape_vp']:.2f}% / vs {b['mape_vs']:.2f}%")

    # ---- AMPLITUDE: by dRMS -------------------------------------------------
    rows.sort(key=lambda m: m["_drms"], reverse=True)
    print("\n########## AMPLITUDE -- ranked by dRMS% (RMS removed; higher=better)")
    h = (f"{'#':>2} {'combo':16}{'prec':>6} | {'vp dRMS%':>9}{'deep':>6}"
         f" | {'vs dRMS%':>9}{'deep':>6} | {'corr vp':>8}{'vs':>6} |{'h':>5} ok")
    print(h); print("-" * len(h))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", True) else "NAN"
        print(f"{i:2d} {m['_base']:16}{m['_precond']:>6} | "
              f"{m['drms_vp']:9.1f}{m['drms_deep_vp']:6.0f} | "
              f"{m['drms_vs']:9.1f}{m['drms_deep_vs']:6.0f} | "
              f"{m.get('update_corr_vp',0):8.2f}{m.get('update_corr_vs',0):6.2f} |"
              f"{m.get('runtime_h',0):5.2f} {ok}")
    b = rows[0]
    print(f"best (dRMS): {b['_base']} [{b['_precond']}]  "
          f"vp {b['drms_vp']:.1f}% / vs {b['drms_vs']:.1f}%")

    # ---- illumination A/B under BOTH families -------------------------------
    _ab(rows, "_ssim_deep", "deep SSIM (structure)", "{:+.3f}")
    _ab(rows, "_drms_deep", "deep dRMS% (amplitude)", "{:+.1f}")

    if args.csv:
        import csv
        cols = ["_base", "_precond", "ssim_vp", "ssim_vs", "_ssim_deep",
                "mape_vp", "mape_vs", "drms_vp", "drms_vs", "_drms_deep",
                "update_corr_vp", "update_corr_vs", "runtime_h", "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.lstrip("_") for c in cols])
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
