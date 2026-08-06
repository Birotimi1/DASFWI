"""Verify DAS channel geometry AGAINST THE DATA, at any site.

>>> NOTHING HERE IS SITE-SPECIFIC. NO WELL, NO DEPTH, NO CONVENTION. <<<
Everything is measured from the recorded moveout and compared with whatever the
headers claim. That is the only form of this check that deploys anywhere: a
hardcoded correction for one well is not a fix, it is a second bug waiting for
the next site.

WHY THIS EXISTS
---------------
The FORGE field campaign ran 13 inversions to completion and produced nothing.
Root cause: the channel -> depth mapping was wrong, so receivers sat where no
velocity model could explain the arrivals. Symptoms, all downstream of it:

    skip = 1.000 at every iteration, never moving
    convsi drove the misfit to 1e-7 anyway (it is SOURCE-INDEPENDENT, so a
        systematic time error is absorbed into the matching filter)
    gc diverged
    the model railed to both velocity bounds and recovered no structure

Not one of those pointed at geometry. They looked like a misfit problem, an
optimizer problem, a starting-model problem. The cheapest possible check --
does the recorded moveout agree with the header geometry -- would have caught
it before a single SU was spent.

THE MEASUREMENT
---------------
Cross-correlate channel pairs a fixed STRIDE apart and take the median lag.
Apparent velocity = (stride * spacing) / median_lag, SIGNED:

    positive  arrivals get LATER along the header's increasing-depth direction
              -> header ordering agrees with the physics
    negative  arrivals get EARLIER -> the fibre is INVERTED relative to the
              header ordering

Three traps, each of which produced a confidently wrong number before this
function existed and each of which is now guarded:

1. STRIDE. Adjacent channels are ~1 m apart; at 5000 m/s that is 0.2 ms, a
   FIFTH of a 1 ms sample. Integer-lag cross-correlation returns 0 for every
   pair and the accumulated quantisation bias fabricates a large false moveout.
   `min_lag_samples` rejects any stride whose median lag is not resolvable.
2. FITTING. Every pair at a fixed stride has the SAME dz, so a straight-line
   fit of lag against dz is degenerate -- it returned exactly 2x the truth on
   synthetic data. Use the ratio, never a fit.
3. PICKING. Amplitude-threshold first breaks pick the LARGEST event, not the
   first, and on real DAS that inverted the apparent moveout. Nothing here
   picks anything.
"""
import numpy as np


def _xlag(a, b, max_lag):
    """Sub-sample lag of `a` relative to `b`; positive means `a` arrives LATER."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    n = a.size
    c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)), 2 * n)
    c = np.concatenate((c[-max_lag:], c[:max_lag + 1]))
    k = int(np.argmax(c))
    if 0 < k < c.size - 1:                       # parabolic sub-sample peak
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        k += (0.5 * (y0 - y2) / den) if den else 0.0
    return k - max_lag


def apparent_velocity(gather, spacing, stride, dt, max_lag=60,
                      min_lag_samples=3.0):
    """Signed apparent velocity along the channel axis, in m/s.

    `gather` is [nt, C], ordered as the CALLER believes depth increases.
    Returns (v, info). `v` is None when the lag is too small to resolve --
    silently returning a number there is how the 1 m-stride version fabricated
    a 179 ms moveout out of pure quantisation.
    """
    g = np.asarray(gather, float)
    step = max(stride // 2, 1)
    lags = np.array([_xlag(g[:, c + stride], g[:, c], max_lag) * dt
                     for c in range(0, g.shape[1] - stride, step)])
    if lags.size == 0:
        return None, {"reason": "stride exceeds the channel count"}
    med = float(np.median(lags))
    info = {"median_lag_s": med, "median_lag_samples": med / dt,
            "n_pairs": int(lags.size), "dz_m": float(stride * spacing),
            "frac_positive": float(np.mean(lags > 0)),
            "iqr_s": float(np.percentile(lags, 75) - np.percentile(lags, 25))}
    if abs(med / dt) < min_lag_samples:
        info["reason"] = (f"median lag {med/dt:.2f} samples < {min_lag_samples} "
                          f"-- UNRESOLVABLE at this stride, increase it")
        return None, info
    return float(stride * spacing) / med, info


def check_orientation(gathers, spacing, dt, strides=(50, 100, 200),
                      v_plausible=(500.0, 9000.0), min_usable=4):
    """Does the recorded moveout agree with the caller's channel ordering?

    `gathers` is [S, nt, C] (or [nt, C]) for shots NEAR the well, where the ray
    is closest to along-fibre. Site-agnostic: `spacing` and `dt` come from the
    file headers, and the plausible-velocity window is physics, not geology --
    no rock carries P waves outside ~0.5-9 km/s.

    Returns a dict with `agrees` (bool or None when undecidable). `None` means
    UNDECIDABLE and must not be read as agreement.
    """
    g = np.asarray(gathers, float)
    if g.ndim == 2:
        g = g[None]
    per = []
    for s in range(g.shape[0]):
        for st in strides:
            v, info = apparent_velocity(g[s], spacing, st, dt)
            per.append(dict(shot=s, stride=st, v=v, **info))
    good = [p for p in per if p["v"] is not None
            and v_plausible[0] <= abs(p["v"]) <= v_plausible[1]]
    out = {"per_measurement": per, "n_usable": len(good), "n_total": len(per)}
    # A SINGLE usable measurement MUST NOT decide. The first version required
    # only frac_positive >= 0.8, so one surviving pair scored 1.0 and returned
    # a confident "AGREES" -- on data that nine-of-nine measurements call
    # INVERTED when the right shots are used. Same failure as every saturated
    # diagnostic on this project: a statistic computed over too little data,
    # reported as if it were solid.
    if 0 < len(good) < min_usable:
        out.update(agrees=None, verdict=(
            f"UNDECIDABLE -- only {len(good)} of {len(per)} measurements were "
            f"usable (need {min_usable}). Too few to decide anything. Use "
            f"shots NEAR the well, where the ray runs along the fibre: far "
            f"offsets have no along-fibre direct arrival and their lags are "
            f"neither resolvable nor meaningful."))
        return out
    if not good:
        out.update(agrees=None, verdict=(
            "UNDECIDABLE -- no stride gave a resolvable lag with a physically "
            "plausible speed. Try larger strides, shots nearer the well, or "
            "check that the gather really contains a direct arrival."))
        return out
    vs = np.array([p["v"] for p in good])
    frac_pos = float(np.mean(vs > 0))
    out["v_median"] = float(np.median(vs))
    out["v_spread"] = float(np.percentile(np.abs(vs), 75)
                            - np.percentile(np.abs(vs), 25))
    out["frac_positive"] = frac_pos
    if frac_pos >= 0.8:
        out.update(agrees=True, verdict=(
            f"Header ordering AGREES with the data: arrivals get later with "
            f"depth, apparent speed {np.median(vs):.0f} m/s."))
    elif frac_pos <= 0.2:
        out.update(agrees=False, verdict=(
            f"*** HEADER ORDERING IS INVERTED *** The direct arrival gets "
            f"EARLIER as the headers say depth INCREASES (apparent speed "
            f"{np.median(vs):.0f} m/s, |v| = {abs(np.median(vs)):.0f} m/s is "
            f"physically sound, so it is the SIGN that is wrong, not the "
            f"measurement). The channel->depth map is reversed: receivers are "
            f"being placed at the wrong end of the fibre and NO velocity model "
            f"can fit the data. Do not invert until this is resolved."))
    else:
        out.update(agrees=None, verdict=(
            f"UNDECIDABLE -- {frac_pos:.0%} of measurements positive; the "
            f"strides disagree among themselves. Inspect per_measurement "
            f"before trusting either the headers or this check."))
    return out
