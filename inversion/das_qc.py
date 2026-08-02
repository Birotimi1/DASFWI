"""SITE-AGNOSTIC DAS QC: is the distortion AMPLITUDE-ONLY, or does it change SHAPE?

Run this on the observed gathers BEFORE inverting at any new site. The answer
decides the whole strategy, because Noe et al. (2025) warn:

    "poor coupling not only influences the absolute amplitudes of DAS
     measurements but may also ALTER THE SHAPE of the measured [waveform]"

  * AMPLITUDE-ONLY distortion is already handled -- per-trace normalisation is on
    in every run, plus channel weighting and amplitude-insensitive misfits (gc).
  * SHAPE distortion is NOT handled by any misfit choice. A phase misfit protects
    against a scalar, not against a filter. It has to be deconvolved out of the
    data or modelled in the forward problem (Celli et al. 2024).

THE PHYSICS THAT MAKES THIS MEASURABLE: DAS channels are metres apart while
seismic wavelengths are ~100 m. Adjacent channels therefore MUST record nearly
the same waveform, differing only by a small moveout. Anything else is
instrumental, not geological. So each channel is compared against the MEDIAN of
its neighbours, which needs no true model, no source wavelet and no other
instrument -- which is exactly why it transfers to any site.

FOUR TESTS (increasing power):
  1 residual   align to neighbours by cross-correlation, normalise, subtract.
                ~0 -> amplitude-only. Large -> shape.
  2 spectral   R_i(f) = |D_i(f)| / median_j |D_j(f)|.  FLAT in f -> a scalar
     ratio      (harmless). Frequency-DEPENDENT -> coupling is acting as a
                FILTER, i.e. genuine shape distortion. Decisive for MAGNITUDE.
  3 repeat-    a coupling defect belongs to the FIBRE, not the shot, so a bad
     ability    channel must distort the SAME way in every shot. Consistent ->
                coupling; random -> noise. Applied to tests 2 and 4, and judged
                against the median's STANDARD ERROR rather than its size: with
                few shots the median of pure noise clears "std < |median|" by
                chance. Both statistics it judges are SIGNED for the same
                reason -- an unsigned one saturates, so "consistently random"
                reads as "consistent". That mistake produced a FALSE shape-
                distortion verdict on real FORGE data; see test 4.
  4 phase      cross-spectrum phase against the neighbours: LINEAR in f = a pure
                time delay (harmless); NON-LINEAR = dispersion. Test 2 sees only
                |R(f)|, so an ALL-PASS filter -- flat magnitude, curved phase --
                passes it while still deforming the waveform. This is the only
                test that catches that, and dispersion corrupts a PHASE misfit
                too, so it is the case nobody, including Noe, solves.
                The statistic is the SIGNED curvature, in radians across the
                band. An RMS departure is unsigned and saturates: on FORGE it
                read 0.50 rad and flipped both wells to a FALSE "shape
                distortion", where the signed version reads 0.004 rad -- the
                phase scatter there is random, so it cancels in the median.

>>> THE PREMISE GUARD (why this is safe to deploy blind) <<<
Everything above rests on "neighbouring channels see the same wavefield". At a
site with COARSE channel spacing, strong scattering or a poor SNR that premise
can FAIL -- and a failed premise looks exactly like shape distortion, so the
function would confidently report a defect that is not there. So the premise is
TESTED, not assumed: the same comparison is repeated in a LOW sub-band, where
the wavelength is longest and neighbour coherence is most strongly guaranteed.
If channels disagree even there, the verdict is INCONCLUSIVE rather than a false
alarm, and the report says what to change (fewer, wider-spaced neighbours; a
lower band; more stacking).

    from inversion.das_qc import qc_das
    res = qc_das(gathers, dt)              # gathers [S, nt, C], any site
    if res["shape_distortion"]: ...        # -> deconvolve or model coupling

    python inversion/das_qc.py --npz site.npz --f-lo 5 --f-hi 40
    python inversion/das_qc.py --well 78A-32 --shots 20        # FORGE
"""
import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def auto_band(gathers, dt, frac=0.02):
    """Usable band from the stacked amplitude spectrum, for an unknown site.

    Returns the frequency range where the mean spectrum exceeds `frac` of its
    peak -- so a site with a different source, sample rate or instrument
    response is analysed in ITS band, not in FORGE's.
    """
    g = np.asarray(gathers, np.float64)
    A = np.abs(np.fft.rfft(g, axis=1)).mean(axis=(0, 2))       # [S,nt,C] -> [nf]
    f = np.fft.rfftfreq(g.shape[1], dt)
    m = A > frac * A.max()
    m[0] = False                                               # never keep DC
    if m.sum() < 4:
        return 1.0 / (dt * g.shape[1]), 0.4 / (2 * dt)
    return float(f[m].min()), float(f[m].max())


def spacing_is_adequate(channel_spacing, v_min, f_hi, n_neigh=2):
    """Is the neighbour-reference premise geometrically plausible here?

    The furthest neighbour used must sit well inside one wavelength, else
    neighbours genuinely differ and the test cannot separate that from a defect.
    Returns (ok, span_over_wavelength). FORGE: 1.02 m spacing, 1500 m/s, 40 Hz
    -> span/lambda = 0.054, comfortable. Advisory only -- the measured low-band
    check in `qc_das` is the authority, since it needs no metadata.
    """
    lam = float(v_min) / float(f_hi)
    return (n_neigh * float(channel_spacing)) / lam <= 0.25, \
           (n_neigh * float(channel_spacing)) / lam


# --------------------------------------------------------------------------- #
# the four tests
# --------------------------------------------------------------------------- #
def channel_report(gathers, dt, f_lo=5.0, f_hi=40.0, n_neigh=2, max_lag_s=0.02,
                   max_shots=None):
    """Per-channel diagnostics against the median of its neighbours.

    gathers: [S, nt, C] raw strain rate (or any DAS observable).
    Returns a dict of per-channel arrays, all length C.
    """
    g = np.asarray(gathers, np.float64)
    if g.ndim != 3:
        raise ValueError(f"gathers must be [S, nt, C], got shape {g.shape}")
    if max_shots:
        g = g[:int(max_shots)]
    S, nt, C = g.shape
    if C < 2 * n_neigh + 2:
        raise ValueError(f"need > {2 * n_neigh + 1} channels for a neighbour "
                         f"reference with n_neigh={n_neigh}, got {C}")
    x = _bandpass_fft(g.transpose(0, 2, 1), dt, f_lo, f_hi)     # [S, C, nt]
    max_lag = max(1, int(max_lag_s / dt))
    rms = np.sqrt((x ** 2).mean(axis=2))                        # [S, C]

    resid = np.full((S, C), np.nan)
    corr = np.full((S, C), np.nan)
    ratio_slope = np.full((S, C), np.nan)                       # spectral tilt
    phase_nl = np.full((S, C), np.nan)                          # phase curvature
    f = np.fft.rfftfreq(nt, dt)
    band = (f >= f_lo) & (f <= f_hi)
    bidx = np.flatnonzero(band)                                 # CONTIGUOUS, for
    fb = f[bidx]                                                # phase unwrapping

    for s in range(S):
        X = np.fft.rfft(x[s], axis=1)                           # [C, nf] complex
        A = np.abs(X)
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
            # ---- 4. PHASE of the ratio: a delay, or DISPERSION? -------------
            # Test 2 sees only |R(f)|, so an ALL-PASS filter -- flat magnitude,
            # non-linear phase -- would pass it while still deforming the
            # waveform. The cross-spectrum phase against the neighbour median is
            # LINEAR in f for a pure time delay; departure from that line is
            # dispersion, which is shape distortion a phase misfit cannot undo.
            # The statistic is the SIGNED curvature of that phase, scaled to
            # radians of departure across the band. Signed is essential: an RMS
            # departure is unsigned and SATURATES, so noise-driven phase scatter
            # is "consistently ~0.5 rad" in every shot and sails through a
            # std-below-magnitude repeatability test. A signed curvature from
            # random noise flips sign shot to shot and cancels in the median,
            # exactly as the signed magnitude slope above already does.
            Xref = np.fft.rfft(ref)
            w = np.abs(Xref[bidx])                    # ignore empty frequencies
            if bidx.size > 8 and w.max() > 0:
                ph = np.unwrap(np.angle(X[c][bidx] * np.conj(Xref[bidx])))
                quad = np.polyfit(fb, ph, 2, w=w)[0]
                fbar = np.average(fb, weights=w)
                fvar = np.average((fb - fbar) ** 2, weights=w)
                phase_nl[s, c] = quad * fvar          # rad, SIGNED

    with np.errstate(invalid="ignore"):
        return dict(
            n_shots=S, n_chan=C, f_lo=f_lo, f_hi=f_hi,
            resid_med=np.nanmedian(resid, axis=0),
            corr_med=np.nanmedian(corr, axis=0),
            slope_med=np.nanmedian(ratio_slope, axis=0),
            phase_med=np.nanmedian(phase_nl, axis=0),
            # ---- 3. repeatability: does the channel distort the SAME way? ----
            slope_std=np.nanstd(ratio_slope, axis=0),
            phase_std=np.nanstd(phase_nl, axis=0),
            rms_med=np.nanmedian(rms, axis=0),
        )


def _repeatable(med, std, n_shots, k=3.0):
    """Is a per-shot statistic CONSISTENT, or merely consistently LARGE?

    The obvious test, std < |median|, is too weak: a statistic that is random
    every shot but always big -- e.g. the phase of incoherent traces, which
    saturates -- satisfies it. Worse, with only a handful of shots the median of
    pure noise lands far from zero often enough to pass by chance.

    So compare the median against ITS OWN standard error: a value that survives
    is one whose sign and size reproduce shot to shot, which is what "a property
    of the fibre rather than of the shot" actually means.
    """
    sem = np.asarray(std, float) / np.sqrt(max(int(n_shots), 1))
    with np.errstate(invalid="ignore"):
        return np.isfinite(sem) & (np.abs(med) > k * np.maximum(sem, 1e-12))


def verdict(rep, corr_bad=0.90, slope_bad=0.5, phase_bad=0.5, coherent_min=0.3):
    """Amplitude-only or shape distortion? Reported per channel and overall.

    `phase_bad` is anchored physically, not fitted: cycle skipping needs half a
    period, i.e. pi radians, so 0.5 rad of instrument dispersion is ~16% of that
    budget consumed before the physics gets a vote. A clean synthetic gather
    measures 0.002 rad, so the margin is ~250x.

    `coherent_min` sits between MEASURED values: incoherent channels correlate
    at 0.15, mildly dispersive ones at 0.45. It cannot be raised much -- at 0.5
    the detection window closes entirely, because by the time curvature clears
    0.5 rad the correlation has already fallen below the gate.

    LIMITS worth knowing when reading the output:
      * SEVERE dispersion destroys neighbour correlation outright, so it is
        reported by test 1 as a shape mismatch (or by the premise guard as
        inconclusive) rather than by test 4. It is never silently called clean.
      * A defect spanning a CONTIGUOUS run of channels wider than the neighbour
        window is invisible by construction -- the reference is distorted too.
        Widen `n_neigh`, or compare against a different fibre section.
    """
    corr, slope, sstd = rep["corr_med"], rep["slope_med"], rep["slope_std"]
    phase = rep.get("phase_med", np.full_like(slope, np.nan))
    pstd = rep.get("phase_std", np.full_like(slope, np.nan))
    ok = np.isfinite(corr) & np.isfinite(slope)
    n = int(ok.sum())
    if n == 0:
        return {"verdict": "NO USABLE CHANNELS", "n_channels_scored": 0,
                "inconclusive": True, "shape_distortion": False,
                "amplitude_only": False}
    shape = ok & (corr < corr_bad)                 # waveform differs after align
    tilted = ok & (np.abs(slope) > slope_bad)      # spectral ratio not flat
    # DISPERSIVE: flat |R(f)| but non-linear phase -- an all-pass filter, which
    # the magnitude test above cannot see at all.
    # `corr > coherent_min` is REQUIRED, not cosmetic: for incoherent traces the
    # cross-spectrum phase is random, so its departure from linear SATURATES near
    # its maximum in every shot -- large and consistent, i.e. indistinguishable
    # from a repeatable filter by magnitude+stability alone. You can only claim a
    # waveform is deformed relative to its neighbours if there IS a shared
    # waveform, so coherence is the precondition that makes the claim meaningful.
    disp = (ok & np.isfinite(phase) & (np.abs(phase) > phase_bad)
            & (corr > coherent_min))
    # repeatable => a fibre property (coupling), not random noise
    ns = rep.get("n_shots", 1)
    repeatable = tilted & _repeatable(slope, sstd, ns)
    disp_rep = disp & _repeatable(phase, pstd, ns)
    frac_shape = float(shape.sum()) / n
    frac_tilt = float(tilted.sum()) / n
    frac_disp = float(disp.sum()) / n
    filtering = (frac_tilt >= 0.05 and repeatable.sum() >= 0.5 * tilted.sum())
    dispersive = (frac_disp >= 0.05 and disp_rep.sum() >= 0.5 * disp.sum())
    if frac_shape < 0.05 and frac_tilt < 0.05 and frac_disp < 0.05:
        v = ("AMPLITUDE-ONLY -- already handled (per-trace normalisation, "
             "channel weighting, gc). No deconvolution needed.")
    elif filtering or dispersive:
        kind = ("MAGNITUDE (spectral tilt)" if filtering and not dispersive else
                "PHASE (dispersion, flat magnitude -- an ALL-PASS filter)"
                if dispersive and not filtering else "MAGNITUDE and PHASE")
        v = (f"SHAPE DISTORTION in {kind}, REPEATABLE ACROSS SHOTS -> consistent "
             "with COUPLING acting as a per-channel FILTER. No misfit choice "
             "fixes this: estimate and DECONVOLVE the transfer function, or "
             "model coupling in the forward problem (Celli et al. 2024).")
    else:
        v = ("SHAPE MISMATCH but NOT repeatable across shots -> more consistent "
             "with NOISE than coupling; channel weighting / windowing should "
             "suffice.")
    return {
        "n_channels_scored": n,
        "frac_shape_mismatch": round(frac_shape, 4),
        "frac_spectral_tilt": round(frac_tilt, 4),
        "frac_phase_nonlinear": round(frac_disp, 4),
        "frac_tilt_repeatable": round(float(repeatable.sum()) /
                                      max(int(tilted.sum()), 1), 4),
        "median_neighbour_corr": round(float(np.nanmedian(corr)), 4),
        "median_abs_slope": round(float(np.nanmedian(np.abs(slope))), 4),
        "median_abs_phase_curv_rad": (round(float(np.nanmedian(np.abs(phase))), 4)
                                      if np.isfinite(phase).any() else None),
        "n_dead_channels": int((rep["rms_med"] <
                                0.01 * np.nanmedian(rep["rms_med"])).sum()),
        # machine-readable flags for the run drivers
        "shape_distortion": bool(filtering or dispersive),
        "dispersive": bool(dispersive),
        "amplitude_only": bool(frac_shape < 0.05 and frac_tilt < 0.05
                               and frac_disp < 0.05),
        "inconclusive": False,
        "verdict": v,
    }


# --------------------------------------------------------------------------- #
# THE ONE-CALL API -- this is what a new site should use
# --------------------------------------------------------------------------- #
def qc_das(gathers, dt, f_lo=None, f_hi=None, n_neigh=2, max_shots=None,
           channel_spacing=None, v_min=None, premise_corr=0.40,
           premise_shots=4, check_premise=True):
    """Full DAS QC on raw gathers from ANY site. One call, one verdict.

    gathers          [S, nt, C] observed DAS traces (strain rate or strain).
    dt               sample interval, s.
    f_lo, f_hi       analysis band; None -> measured from the data (`auto_band`).
    channel_spacing  metres, optional -- enables the geometric premise check.
    v_min            m/s, optional -- ditto (only used with channel_spacing).

    Returns the verdict dict plus `band`, `premise`, and the per-channel arrays.
    Read `res["shape_distortion"]` / `res["inconclusive"]` to branch in code, or
    print `format_report(res)`.
    """
    g = np.asarray(gathers, np.float64)
    if g.ndim != 3:
        raise ValueError(f"gathers must be [S, nt, C], got shape {g.shape}")
    if f_lo is None or f_hi is None:
        a_lo, a_hi = auto_band(g, dt)
        f_lo = a_lo if f_lo is None else f_lo
        f_hi = a_hi if f_hi is None else f_hi
    if not f_hi > f_lo > 0:
        raise ValueError(f"bad band {f_lo}-{f_hi} Hz")

    rep = channel_report(g, dt, f_lo, f_hi, n_neigh, max_shots=max_shots)
    res = verdict(rep)
    res["band"] = [round(float(f_lo), 3), round(float(f_hi), 3)]
    res["n_shots"], res["n_channels"] = rep["n_shots"], rep["n_chan"]

    # ---- THE PREMISE GUARD ------------------------------------------------- #
    # Repeat in the BOTTOM THIRD of the band, where the wavelength is longest
    # and neighbours are most strongly obliged to agree. If they disagree even
    # there, "neighbours see the same wavefield" is false at this site and NO
    # conclusion about shape distortion can be drawn from this test.
    #
    # CALIBRATION (tests/test_das_qc.py, measured not guessed): low-band
    # neighbour correlation is 1.00 clean, 0.99 scalar, 0.99 magnitude filter,
    # 0.69 severe all-pass dispersion, 0.18 genuinely incoherent channels. The
    # threshold has to sit BELOW real distortion and ABOVE incoherence, hence
    # 0.40 -- an earlier 0.80 swallowed true dispersion as "inconclusive",
    # because severe distortion lowers this correlation too. That is also why
    # the REPEATABILITY override below exists: it is the property incoherence
    # can never fake.
    premise = {"checked": False}
    if check_premise:
        lo_hi = f_lo + (f_hi - f_lo) / 3.0
        try:
            lo_rep = channel_report(g, dt, f_lo, lo_hi, n_neigh,
                                    max_shots=min(premise_shots, g.shape[0]))
            lo_corr = float(np.nanmedian(lo_rep["corr_med"]))
        except (ValueError, np.linalg.LinAlgError) as e:
            lo_corr, premise["error"] = float("nan"), str(e)
        premise.update(checked=True, low_band=[round(float(f_lo), 3),
                                               round(float(lo_hi), 3)],
                       low_band_corr=round(lo_corr, 4) if lo_corr == lo_corr
                       else None, threshold=premise_corr,
                       holds=bool(lo_corr == lo_corr and lo_corr >= premise_corr))
        if not premise["holds"]:
            res["inconclusive"] = True
            res["shape_distortion"] = False
            res["amplitude_only"] = False
            res["verdict"] = (
                f"INCONCLUSIVE -- neighbouring channels disagree even at LOW "
                f"frequency (corr {lo_corr:.3f} < {premise_corr} over "
                f"{f_lo:.1f}-{lo_hi:.1f} Hz), so the neighbour-reference premise "
                f"does not hold at this site. This is NOT evidence of shape "
                f"distortion. Likely causes: channel spacing too coarse relative "
                f"to wavelength, strong scattering, or poor SNR. Try n_neigh=1, "
                f"a narrower/lower band, or stack more shots before concluding.")
    if channel_spacing and v_min:
        ok, ratio = spacing_is_adequate(channel_spacing, v_min, f_hi, n_neigh)
        premise["geometric_ok"] = bool(ok)
        premise["span_over_wavelength"] = round(float(ratio), 4)
    res["premise"] = premise
    res["per_channel"] = {k: np.asarray(v).tolist()
                          for k, v in rep.items() if k.endswith("_med")}
    return res


def recommended_settings(res):
    """Turn the verdict into the inversion settings it implies."""
    if res.get("inconclusive"):
        return {"note": "QC inconclusive -- keep the conservative defaults "
                        "(normalize=True, gc/phase misfit, channel weighting)."}
    if res.get("shape_distortion"):
        return {"note": "SHAPE DISTORTION: a misfit choice cannot fix this.",
                "deconvolve_channels": True, "channel_weight": True,
                "misfit": "gc", "normalize": True,
                "action": "estimate the per-channel transfer function from the "
                          "spectral ratios and deconvolve, or model coupling "
                          "in the forward problem (Celli et al. 2024)."}
    return {"note": "Amplitude-only: phase-based misfits are sound here.",
            "deconvolve_channels": False, "channel_weight": True,
            "misfit": "gc", "normalize": True}


def format_report(res):
    """Human-readable summary of a `qc_das` result."""
    L = [f"--- DAS QC: {res.get('n_shots', '?')} shots, "
         f"{res.get('n_channels', '?')} channels, "
         f"{res['band'][0]:.1f}-{res['band'][1]:.1f} Hz ---"]
    for k in ("median_neighbour_corr", "median_abs_slope",
              "median_abs_phase_curv_rad", "frac_shape_mismatch",
              "frac_spectral_tilt", "frac_phase_nonlinear", "n_dead_channels"):
        if k in res:
            L.append(f"    {k:<24}: {res[k]}")
    p = res.get("premise", {})
    if p.get("checked"):
        L.append(f"    {'premise (low-band corr)':<24}: {p.get('low_band_corr')}"
                 f"  {'HOLDS' if p.get('holds') else 'FAILS'}")
    if "span_over_wavelength" in p:
        L.append(f"    {'neighbour span / lambda':<24}: "
                 f"{p['span_over_wavelength']}  "
                 f"{'ok' if p.get('geometric_ok') else 'TOO COARSE'}")
    L.append(f"\n    VERDICT: {res['verdict']}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# site loaders -- add one per site; the tests above never change
# --------------------------------------------------------------------------- #
def load_npz(path, key=None, dt=None):
    """[S, nt, C] gathers + dt from a .npz/.npy. The portable path for a new
    site: save whatever you have as `gathers` (+ `dt`) and run."""
    p = Path(path)
    if p.suffix == ".npy":
        g = np.load(p)
    else:
        z = np.load(p, allow_pickle=True)
        if key is None:
            for k in ("gathers", "obs", "data", "strain_rate"):
                if k in z:
                    key = k
                    break
            else:
                key = list(z.keys())[0]
        g = z[key]
        if dt is None and "dt" in z:
            dt = float(np.asarray(z["dt"]).ravel()[0])
    g = np.asarray(g)
    if g.ndim == 2:                       # a single shot [nt, C]
        g = g[None]
    if dt is None:
        raise SystemExit("dt not found in the file -- pass --dt")
    return g, float(dt)


def load_forge_well(well, shots, root=None):
    """FORGE SEG-Y (kept here so `--well` still works; imported lazily so this
    module has no FORGE dependency at a new site).

    Uses the loader's DAS_VSP_DIR, which honours $FORGE_DAS_DIR -- the same
    override the inversion drivers use, so QC and inversion always read the
    SAME bytes.
    """
    from forge.field_loader import (read_shot_geometry, load_strain_gathers,
                                    DAS_VSP_DIR)
    well_dir = Path(root or DAS_VSP_DIR) / well
    if not well_dir.is_dir():
        raise SystemExit(f"no such well directory: {well_dir}\n"
                         f"set $FORGE_DAS_DIR or pass --data-root")
    geom = read_shot_geometry(well_dir, n_shots=shots)
    C = len(geom["rcv_xyz"])
    return load_strain_gathers(geom["files"], C), geom["dt"], C


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--well", default=None, help="FORGE well, e.g. 78A-32")
    src.add_argument("--npz", default=None,
                     help="any site: .npz/.npy of [S, nt, C] gathers")
    ap.add_argument("--key", default=None, help="array name inside the .npz")
    ap.add_argument("--dt", type=float, default=None, help="s (npz/npy input)")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--f-lo", type=float, default=None, dest="f_lo",
                    help="default: measured from the data")
    ap.add_argument("--f-hi", type=float, default=None, dest="f_hi")
    ap.add_argument("--n-neigh", type=int, default=2, dest="n_neigh")
    ap.add_argument("--channel-spacing", type=float, default=None,
                    help="m, enables the geometric premise check")
    ap.add_argument("--v-min", type=float, default=None, dest="v_min",
                    help="m/s, ditto")
    ap.add_argument("--data-root", default=None, dest="data_root",
                    help="directory holding the per-well SEG-Y (default "
                         "$FORGE_DAS_DIR)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.npz:
        g, dt = load_npz(args.npz, args.key, args.dt)
        name = Path(args.npz).stem
    else:
        well = args.well or "78A-32"
        g, dt, _ = load_forge_well(well, args.shots, args.data_root)
        name = well
    print(f"=== DAS QC: {name} -- gathers {g.shape}, dt={dt}, "
          f"max|.|={np.abs(g).max():.3e} ===", flush=True)

    res = qc_das(g, dt, args.f_lo, args.f_hi, n_neigh=args.n_neigh,
                 max_shots=args.shots, channel_spacing=args.channel_spacing,
                 v_min=args.v_min)
    print(format_report(res))
    print("\n    RECOMMENDED: " + json.dumps(recommended_settings(res), indent=6))

    out = Path(args.out) if args.out else Path("results") / "das_qc" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"site": name, "dt": dt, **res}, indent=2))
    print(f"\nwrote {out}")
    return 2 if res.get("shape_distortion") else 0


if __name__ == "__main__":
    raise SystemExit(main())
