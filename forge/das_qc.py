"""Is FORGE DAS distortion AMPLITUDE-ONLY, or does it change waveform SHAPE?

This distinction decides the whole field strategy. Noe et al. (2025) warn:

    "poor coupling not only influences the absolute amplitudes of DAS
     measurements but may also ALTER THE SHAPE of the measured [waveform]"

  * AMPLITUDE-ONLY distortion is already handled -- per-trace normalisation is on
    in every run, plus channel weighting and amplitude-insensitive misfits (gc).
  * SHAPE distortion is NOT handled by any misfit choice. A phase misfit protects
    against a scalar, not against a filter. It has to be deconvolved out of the
    data or modelled in the forward problem (Celli et al. 2024). Neither Park et
    al. nor Noe et al. characterise it at FORGE; Park never mentions coupling.

THE PHYSICS THAT MAKES THIS MEASURABLE: DAS channels are metres apart while
seismic wavelengths are ~100 m (20 Hz at ~2000 m/s). Adjacent channels therefore
MUST record nearly the same waveform, differing only by a small moveout. Anything
else is instrumental, not geological. So each channel can be compared against the
median of its neighbours, and the comparison localises the defect.

FOUR TESTS (increasing power):
  1 residual   align to neighbours by cross-correlation, normalise, subtract.
                ~0 -> amplitude-only. Large -> shape.
  2 spectral   R_i(f) = |D_i(f)| / median_j |D_j(f)|.  FLAT in f -> a scalar
     ratio      (harmless). Frequency-DEPENDENT -> coupling is acting as a
                FILTER, i.e. genuine shape distortion. This is the decisive test.
  3 repeat-    a coupling defect belongs to the FIBRE, not the shot, so a bad
     ability    channel must distort the SAME way in every shot. Consistent ->
                coupling; random -> noise.
  4 phase      linear phase vs frequency = pure time delay (harmless);
                NON-LINEAR phase = dispersion, which corrupts a phase misfit too
                -- the case nobody, including Noe, solves.

If test 2 finds a consistent per-channel filter, it can be ESTIMATED AND
DECONVOLVED as a calibration step, which would go beyond both papers.

    python forge/das_qc.py --well 78A-32 --shots 20
"""
import argparse
import json
from pathlib import Path

import numpy as np

from forge.field_loader import (read_shot_geometry, load_strain_gathers,
                                _default_das_dir)


def _bandpass_fft(x, dt, f_lo, f_hi):
    """Zero-phase band-limit along the last axis (keeps the comparison in the
    band the inversion actually uses)."""
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, dt)
    X = np.fft.rfft(x, axis=-1)
    X[..., (f < f_lo) | (f > f_hi)] = 0
    return np.fft.irfft(X, n=n, axis=-1)


def _align(a, b, max_lag):
    """Shift b onto a by integer cross-correlation lag; returns (b_shifted, lag)."""
    n = a.size
    c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)), 2 * n)
    c = np.concatenate((c[-max_lag:], c[:max_lag + 1]))
    lag = int(np.argmax(c)) - max_lag
    return np.roll(b, lag), lag


def channel_report(gathers, dt, f_lo=5.0, f_hi=40.0, n_neigh=2, max_lag_s=0.02):
    """Per-channel diagnostics against the median of its neighbours.

    gathers: [S, nt, C] raw strain rate.
    Returns a dict of per-channel arrays, all length C.
    """
    S, nt, C = gathers.shape
    x = _bandpass_fft(np.asarray(gathers, np.float64).transpose(0, 2, 1),
                      dt, f_lo, f_hi)                       # [S, C, nt]
    max_lag = max(1, int(max_lag_s / dt))
    rms = np.sqrt((x ** 2).mean(axis=2))                    # [S, C]

    resid = np.full((S, C), np.nan)
    corr = np.full((S, C), np.nan)
    ratio_slope = np.full((S, C), np.nan)                   # spectral tilt
    nyq = 1.0 / (2 * dt)
    f = np.fft.rfftfreq(nt, dt)
    band = (f >= f_lo) & (f <= f_hi)
    logf = np.log(np.clip(f[band], 1e-9, None))

    for s in range(S):
        A = np.abs(np.fft.rfft(x[s], axis=1))               # [C, nf]
        for c in range(C):
            lo, hi = max(0, c - n_neigh), min(C, c + n_neigh + 1)
            idx = [j for j in range(lo, hi) if j != c]
            if len(idx) < 2 or rms[s, c] <= 0:
                continue
            ref = np.median(x[s, idx], axis=0)
            if not np.any(ref):
                continue
            # ---- 1. residual after ALIGNMENT and AMPLITUDE normalisation -----
            a = ref / np.sqrt((ref ** 2).mean())
            b = x[s, c] / np.sqrt((x[s, c] ** 2).mean())
            b_al, _ = _align(a, b, max_lag)
            resid[s, c] = np.sqrt(((a - b_al) ** 2).mean())
            corr[s, c] = float(np.dot(a, b_al) / (np.linalg.norm(a) *
                                                  np.linalg.norm(b_al) + 1e-30))
            # ---- 2. spectral ratio: flat (scalar) or tilted (filter)? -------
            Aref = np.median(A[idx], axis=0)
            m = band & (Aref > 0.05 * Aref.max())
            if m.sum() > 8:
                lr = np.log(np.clip(A[c][m] / Aref[m], 1e-9, None))
                # slope of log-ratio vs log-f: ~0 => scalar, else a filter
                ratio_slope[s, c] = np.polyfit(np.log(f[m]), lr, 1)[0]

    with np.errstate(invalid="ignore"):
        return dict(
            n_shots=S, n_chan=C,
            resid_med=np.nanmedian(resid, axis=0),
            corr_med=np.nanmedian(corr, axis=0),
            slope_med=np.nanmedian(ratio_slope, axis=0),
            # ---- 3. repeatability: does the channel distort the SAME way? ----
            slope_std=np.nanstd(ratio_slope, axis=0),
            rms_med=np.nanmedian(rms, axis=0),
        )


def verdict(rep, corr_bad=0.90, slope_bad=0.5):
    """Amplitude-only or shape distortion? Reported per channel and overall."""
    corr, slope, sstd = rep["corr_med"], rep["slope_med"], rep["slope_std"]
    ok = np.isfinite(corr) & np.isfinite(slope)
    n = int(ok.sum())
    if n == 0:
        return {"verdict": "NO USABLE CHANNELS"}
    shape = ok & (corr < corr_bad)                 # waveform differs after align
    tilted = ok & (np.abs(slope) > slope_bad)      # spectral ratio not flat
    # repeatable => a fibre property (coupling), not random noise
    repeatable = tilted & (sstd < np.abs(slope))
    frac_shape = float(shape.sum()) / n
    frac_tilt = float(tilted.sum()) / n
    if frac_shape < 0.05 and frac_tilt < 0.05:
        v = ("AMPLITUDE-ONLY -- already handled (per-trace normalisation, "
             "channel weighting, gc). No deconvolution needed.")
    elif frac_tilt >= 0.05 and repeatable.sum() >= 0.5 * tilted.sum():
        v = ("SHAPE DISTORTION, REPEATABLE ACROSS SHOTS -> consistent with "
             "COUPLING acting as a per-channel FILTER. No misfit choice fixes "
             "this: estimate and DECONVOLVE the transfer function, or model "
             "coupling in the forward problem (Celli et al. 2024).")
    else:
        v = ("SHAPE MISMATCH but NOT repeatable across shots -> more consistent "
             "with NOISE than coupling; channel weighting / windowing should "
             "suffice.")
    return {
        "n_channels_scored": n,
        "frac_shape_mismatch": round(frac_shape, 4),
        "frac_spectral_tilt": round(frac_tilt, 4),
        "frac_tilt_repeatable": round(float(repeatable.sum()) /
                                      max(int(tilted.sum()), 1), 4),
        "median_neighbour_corr": round(float(np.nanmedian(corr)), 4),
        "median_abs_slope": round(float(np.nanmedian(np.abs(slope))), 4),
        "verdict": v,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--well", default="78A-32")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--f-lo", type=float, default=5.0, dest="f_lo")
    ap.add_argument("--f-hi", type=float, default=40.0, dest="f_hi")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    well_dir = _default_das_dir() / args.well
    print(f"=== FORGE DAS QC: {well_dir} ===", flush=True)
    geom = read_shot_geometry(well_dir, n_shots=args.shots)
    dt, nt = geom["dt"], geom["nt"]
    C = len(geom["rcv_xyz"])
    print(f"    {len(geom['files'])} shots, {C} channels, nt={nt}, dt={dt}", flush=True)
    g = load_strain_gathers(geom["files"], C)
    print(f"    gathers {g.shape}, max|.|={np.abs(g).max():.3e}", flush=True)

    rep = channel_report(g, dt, args.f_lo, args.f_hi)
    v = verdict(rep)
    print(f"\n--- per-channel summary ({args.f_lo}-{args.f_hi} Hz) ---")
    print(f"    neighbour correlation : median {np.nanmedian(rep['corr_med']):.3f}"
          f"  p10 {np.nanpercentile(rep['corr_med'], 10):.3f}")
    print(f"    spectral-ratio slope  : median |{np.nanmedian(np.abs(rep['slope_med'])):.3f}|"
          "   (0 = pure scalar, non-zero = filter)")
    print(f"    dead/quiet channels   : "
          f"{int((rep['rms_med'] < 0.01 * np.nanmedian(rep['rms_med'])).sum())}")
    print("\n--- VERDICT ---")
    for k, val in v.items():
        print(f"    {k}: {val}")
    out = Path(args.out) if args.out else Path("results") / "forge_das_qc" / \
        f"{args.well}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {**v, "well": args.well, "shots": len(geom["files"]), "channels": C,
         "f_lo": args.f_lo, "f_hi": args.f_hi,
         "per_channel": {k: np.asarray(val).tolist()
                         for k, val in rep.items() if k.endswith("_med")}},
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
