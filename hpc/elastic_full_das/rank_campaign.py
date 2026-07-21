#!/usr/bin/env python3
"""Rank the elastic Vp/Vs DAS campaign by SSIM + MAPE, with the illumination A/B.

Following Liu's paper, model recovery is scored with (per parameter):
    SSIM  - structural similarity to the true model (Wang 2004); higher better,
            1 = identical.  <-- primary ranking metric
    MAPE  - mean absolute percentage error (Hyndman & Koehler 2006); lower better.
Both are computed HERE from the saved setup.npz (vp_true/vs_true) and each
combo's iter_vp.npz / iter_vs.npz (recovered = last cached iteration), so a
running campaign is scored without any re-run. The combined ranking uses the
mean SSIM of (vp, vs); the DEEP-half SSIM drives the illumination A/B.

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


def _load(results):
    setup_f = os.path.join(results, "setup.npz")
    if not os.path.isfile(setup_f):
        print(f"no setup.npz under {results}", file=sys.stderr)
        sys.exit(1)
    s = np.load(setup_f)
    truth = {"vp": np.asarray(s["vp_true"], float),
             "vs": np.asarray(s["vs_true"], float)}
    nz = truth["vp"].shape[0]
    deep = slice(nz // 2, nz)

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
            m[f"ssim_{p}"] = sc["ssim"]
            m[f"mape_{p}"] = sc["mape"]
            m[f"ssim_deep_{p}"] = sc["ssim_deep"]
        if not ok:
            continue
        m["_base"] = f"{m.get('misfit','?')}_{m.get('optimizer','?')}"
        m["_precond"] = m.get("precond") or ("illum" if tag.endswith("_illum")
                                             else "off")
        finite = m.get("losses_finite", True)
        m["_ssim"] = 0.5 * (m["ssim_vp"] + m["ssim_vs"]) if finite else -1.0
        m["_ssim_deep"] = 0.5 * (m["ssim_deep_vp"] + m["ssim_deep_vs"])
        m["_mape"] = 0.5 * (m["mape_vp"] + m["mape_vs"])
        rows.append(m)
    return rows


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
    rows.sort(key=lambda m: m["_ssim"], reverse=True)

    hdr = (f"{'#':>2} {'combo':16}{'prec':>6} | "
           f"{'SSIM vp':>8}{'vs':>7}{'deep':>7} | "
           f"{'MAPE% vp':>9}{'vs':>7} |{'h':>5} ok")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(rows, 1):
        ok = "OK" if m.get("losses_finite", True) else "NAN"
        print(f"{i:2d} {m['_base']:16}{m['_precond']:>6} | "
              f"{m['ssim_vp']:8.3f}{m['ssim_vs']:7.3f}{m['_ssim_deep']:7.3f} | "
              f"{m['mape_vp']:9.2f}{m['mape_vs']:7.2f} |"
              f"{m.get('runtime_h', 0):5.2f} {ok}")

    # --- illumination A/B on DEEP structural similarity (vp+vs) ---------------
    by_base = {}
    for m in rows:
        by_base.setdefault(m["_base"], {})[m["_precond"]] = m
    pairs = [(b, d["illum"], d["off"]) for b, d in by_base.items()
             if "illum" in d and "off" in d]
    print(f"\n=== illumination A/B on DEEP SSIM (vp+vs) "
          f"({len(pairs)} paired combos) ===")
    if pairs:
        gains = []
        print(f"{'combo':16}{'deep illum':>12}{'deep off':>10}{'gain':>9}")
        for b, mi, mo in sorted(pairs, key=lambda t: t[1]["_ssim_deep"] - t[2]["_ssim_deep"],
                                reverse=True):
            gi, go = mi["_ssim_deep"], mo["_ssim_deep"]
            gains.append(gi - go)
            print(f"{b:16}{gi:12.3f}{go:10.3f}{gi - go:+9.3f}")
        improved = sum(1 for g in gains if g > 0)
        print(f"\nillumination improved DEEP SSIM in {improved}/{len(gains)} "
              f"combos; mean gain {sum(gains)/len(gains):+.3f}")

    nfin = sum(1 for m in rows if m.get("losses_finite", True))
    print(f"\n{len(rows)}/90 runs scored ({nfin} finite)")
    b = rows[0]
    print(f"best (mean SSIM): {b['_base']} [{b['_precond']}]  "
          f"SSIM vp {b['ssim_vp']:.3f} / vs {b['ssim_vs']:.3f}  |  "
          f"MAPE vp {b['mape_vp']:.2f}% / vs {b['mape_vs']:.2f}%")

    if args.csv:
        import csv
        cols = ["_base", "_precond", "ssim_vp", "ssim_vs", "_ssim_deep",
                "mape_vp", "mape_vs", "runtime_h", "losses_finite"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([c.lstrip("_") for c in cols])
            for m in rows:
                w.writerow([m.get(c, "") for c in cols])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
