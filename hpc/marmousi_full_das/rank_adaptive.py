"""PHASE B verdict: rank the multiscale arms, and the elastic pipeline arms.

Reads results/marmousi_full_das/adaptive/<tag>/metrics.json, where tag is
    <objective>_<lo>-<hi>_<optimizer>_<rung>     (switch / adaptive)
    fixed_<misfit>_<optimizer>_<rung>            (control arms)

The decisive comparison is switch vs FIXED-L2: multiscale alone may already cure
the cycle skipping, in which case the switch adds nothing inside the cascade
(it would still stand on its own Phase-A result). Also reports switch vs the
frequency-scheduled `adaptive` arm, which isolates skip-driven timing.

    python hpc/marmousi_full_das/rank_adaptive.py [--rung s16] [--elastic]

--elastic instead ranks the elastic pipeline (results/elastic_full_das/pipeline/).
"""
import argparse
import json
from pathlib import Path

MARGIN = 0.02          # SSIM tolerance for "matches" (the confound threshold)


def load(d, partial=False):
    f = d / "metrics.json"
    if not f.is_file():
        return None
    m = json.loads(f.read_text())
    if not m.get("complete", True) and not partial:
        return None
    return m


def cells(root, partial=False):
    """Complete cells only by default. `partial=True` also returns checkpointed
    in-progress cells -- without it a running campaign looks like '0 cells',
    which hides both progress and misconfigured runs."""
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        m = load(d, partial) if d.is_dir() else None
        if m:
            out.append((d.name, m))
    return out


def trajectory_summary(m):
    """Per-band 'did the controller move?' line -- the evidence it works."""
    bits = []
    for b in (m.get("band_log") or []):
        traj = b.get("trajectory") or []
        lams = [t["lam"] for t in traj if t.get("lam") is not None]
        cut = "full" if b.get("cutoff") is None else f"{b['cutoff']:g}"
        if not lams:
            bits.append(f"{cut}:-")
        elif len(set(lams)) == 1:
            bits.append(f"{cut}:{'R' if lams[0] >= 0.5 else 'S'}")   # Robust/Sharp
        else:
            bits.append(f"{cut}:" + "".join("R" if l >= 0.5 else "S" for l in lams))
    return " ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default=None, help="filter acoustic cells by rung")
    ap.add_argument("--elastic", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--partial", action="store_true",
                    help="also show checkpointed in-progress cells (with "
                         "iterations_done and a VS-LIVE sanity flag)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if args.elastic:
        root = Path(args.results) if args.results else \
            repo / "results" / "elastic_full_das" / "pipeline"
        rows = cells(root, args.partial)
        print("=" * 78)
        print(f"ELASTIC PIPELINE -- {len(rows)} complete cells  ({root})")
        print("=" * 78)
        if not rows:
            return print("no complete cells yet")
        print(f"{'tag':44s} {'SSIM vp':>8s} {'SSIM vs':>8s} {'iters':>9s}  Vs?")
        for name, m in sorted(rows, key=lambda r: -(r[1].get("ssim_vp") or 0)):
            it = ("DONE" if m.get("complete")
                  else f"{m.get('iterations_done', 0)}/{m.get('iterations', 0)}")
            # a run whose Vs never leaves its starting model is Vp-only: the
            # vs_release_band never fires (e.g. 1 band but release band 2)
            vs_live = any(b.get("stage") == "vs" for b in (m.get("band_log") or []))
            flag = "yes" if vs_live else "*** NO -- Vp-only run! ***"
            print(f"{name:44s} {m.get('ssim_vp', 0):8.3f} "
                  f"{m.get('ssim_vs', 0):8.3f} {it:>9s}  {flag}")
        return

    root = Path(args.results) if args.results else \
        repo / "results" / "marmousi_full_das" / "adaptive"
    rows = [(n, m) for n, m in cells(root, args.partial)
            if args.rung is None or m.get("start_rung") == args.rung]
    print("=" * 78)
    print(f"PHASE B -- multiscale arms"
          + (f" @ {args.rung}" if args.rung else "") + f"  ({len(rows)} cells)")
    print("=" * 78)
    if not rows:
        return print(f"no complete cells in {root}")

    print(f"{'tag':40s} {'SSIM':>6s} {'MAPE%':>7s} {'skipF':>6s}  bands(R=robust,S=sharp)")
    for name, m in sorted(rows, key=lambda r: -(r[1].get("ssim") or 0)):
        sk = m.get("skip_final")
        print(f"{name:40s} {m.get('ssim', 0):6.3f} {m.get('mape', 0):7.2f} "
              f"{(f'{sk:.3f}' if isinstance(sk, (int, float)) else '   -'):>6s}"
              f"  {trajectory_summary(m)}")

    # ---- the confound test ---------------------------------------------------
    by = {}
    for name, m in rows:
        by[(m.get("objective"), m.get("optimizer"), m.get("start_rung"))] = m.get("ssim")
    print("-" * 78)
    rungs = sorted({r[1].get("start_rung") for r in rows})
    opts = sorted({r[1].get("optimizer") for r in rows})
    for rg in rungs:
        for o in opts:
            sw = by.get(("switch", o, rg))
            l2 = by.get(("l2", o, rg))
            ad = by.get(("adaptive", o, rg))
            en = by.get(("envelope", o, rg))
            if sw is None:
                continue
            parts = [f"{rg} {o:6s}: switch {sw:.3f}"]
            if l2 is not None:
                if sw >= l2 + MARGIN:
                    parts.append(f"BEATS fixed-l2 {l2:.3f} (+{sw - l2:.3f})")
                elif abs(sw - l2) < MARGIN:
                    parts.append(f"~= fixed-l2 {l2:.3f} -> MULTISCALE ALONE "
                                 "ALREADY CURES IT; switch adds nothing here")
                else:
                    parts.append(f"LOSES to fixed-l2 {l2:.3f} ({sw - l2:+.3f})")
            if ad is not None:
                parts.append(f"| vs freq-sched {ad:.3f} ({sw - ad:+.3f})")
            if en is not None:
                parts.append(f"| vs envelope {en:.3f}")
            print("  " + " ".join(parts))
    print("=" * 78)


if __name__ == "__main__":
    main()
