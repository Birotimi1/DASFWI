"""Acceptance criteria for FIELD DAS-FWI, where there is no true model.

Every metric we have used so far (SSIM, MAPE) needs a true model, and FORGE has
none. Without deciding acceptance IN ADVANCE we could produce a beautiful
velocity model and be unable to say whether it is right -- which is the whole
reason the FORGE synthetic exists, and it does not cover the field run itself.

FOUR CRITERIA, in decreasing order of how much external data they need:

1. FIRST-ARRIVAL MISMATCH REDUCTION -- needs NOTHING but the data.
   **This is Park's own published number: INV2 reduced it by 51.7%.** So it is
   directly comparable, and it scores our Route B starter on the SAME AXIS as
   their manually-picked tomography -- theirs from 100 hand-picked shot gathers,
   ours from none. That comparison is the transferability claim in one number.

2. TWO-WELL CROSS-VALIDATION -- needs no external data either.
   78A-32 and 78B-32 are INDEPENDENT datasets over shared geology, so models
   inverted from each separately must agree where they overlap. Disagreement is
   evidence of error requiring no ground truth at all. Park invert both wells
   TOGETHER, so this is a check they could not make.

3. WELL-LOG COMPARISON -- needs the 58-32 sonic + cuttings density, which we do
   NOT currently hold (not on the share; Park's figure shows them). Implemented
   against arrays so it is ready the moment the logs arrive.

4. ZONE-BOUNDARY ALIGNMENT -- do the recovered velocity transitions line up
   with zones I/II/III (unconsolidated alluvium / consolidated alluvium /
   granitoid basement)? Needs only the zone depths, which are readable off
   Park's figure even without the log traces.

>>> THE METHODOLOGICAL LINE <<<
Our mandate is Vp AND Vs from DAS strain rate ALONE. Logs are **VALIDATION
ONLY** -- never an inversion constraint, never a tuning target. Sliding from
"validate against the log" to "tune until it matches the log" would destroy the
transferability claim that is the entire point. **Fix the thresholds BEFORE
looking at the comparison**, which is why this module exists now, while the
synthetic is still running, rather than after the field result is in hand.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# 1. first-arrival mismatch -- Park's 51.7%, and it needs no true model
# --------------------------------------------------------------------------- #
def _xcorr_lag(a, b, max_lag):
    """Integer lag maximising the cross-correlation of a and b."""
    n = a.size
    c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)), 2 * n)
    c = np.concatenate((c[-max_lag:], c[:max_lag + 1]))
    return int(np.argmax(c)) - max_lag


def arrival_lags(syn, obs, dt, max_lag_s=0.25, min_amp_frac=0.05):
    """Per-trace arrival time mismatch [s], syn minus obs, by cross-correlation.

    NO PICKING, deliberately: Park manually pick first arrivals on 100 shot
    gathers, and not needing that is our claim. Cross-correlation measures the
    same quantity where the waveform is coherent.

    Traces whose observed amplitude is below `min_amp_frac` of the gather max
    return NaN rather than a meaningless lag -- a dead channel has no arrival to
    mismatch, and averaging its noise-driven lag into the score would be
    dishonest.
    """
    s = np.asarray(syn, float)
    o = np.asarray(obs, float)
    if s.shape != o.shape:
        raise ValueError(f"shape mismatch: syn {s.shape} vs obs {o.shape}")
    if s.ndim == 2:
        s, o = s[None], o[None]
    S, nt, C = s.shape
    ml = max(1, int(max_lag_s / dt))
    peak = np.abs(o).max()
    out = np.full((S, C), np.nan)
    for i in range(S):
        for c in range(C):
            if np.abs(o[i, :, c]).max() < min_amp_frac * peak:
                continue
            if not np.any(s[i, :, c]):
                continue
            # syn MINUS obs, as documented: a positive lag means the synthetic
            # arrives LATE. Passing (obs, syn) returns the negation -- the
            # magnitudes were exact and only the sign was wrong, which is the
            # easiest kind of error to ship and the hardest to notice later.
            out[i, c] = _xcorr_lag(s[i, :, c], o[i, :, c], ml) * dt
    return out


def mismatch_rms(syn, obs, dt, **kw):
    """RMS first-arrival mismatch in SECONDS (NaN traces excluded)."""
    L = arrival_lags(syn, obs, dt, **kw)
    good = L[np.isfinite(L)]
    return float(np.sqrt((good ** 2).mean())) if good.size else float("nan")


def mismatch_reduction(syn_init, syn_final, obs, dt, **kw):
    """Percent reduction in first-arrival mismatch, initial -> final model.

    **PARK REPORT 51.7% FOR INV2.** Matching or beating that from a starter
    built WITHOUT PICKING is the single cleanest statement of the result.
    Returns a dict so the raw values can be reported alongside the percentage --
    a percentage alone hides whether the starting mismatch was large or small.
    """
    m0 = mismatch_rms(syn_init, obs, dt, **kw)
    m1 = mismatch_rms(syn_final, obs, dt, **kw)
    red = 100.0 * (m0 - m1) / m0 if (m0 and np.isfinite(m0)) else float("nan")
    return dict(rms_init_s=m0, rms_final_s=m1, reduction_pct=red,
                park_inv2_pct=51.7, beats_park=bool(red > 51.7))


# --------------------------------------------------------------------------- #
# 2. two-well cross-validation -- independent data, shared geology
# --------------------------------------------------------------------------- #
def cross_validate(vp_a, vp_b, weights=None):
    """Agreement between models inverted INDEPENDENTLY from the two wells.

    78A-32 and 78B-32 see the same subsurface through different data, so where
    both are illuminated they must agree. This needs no truth, and Park cannot
    make this check because they invert both wells TOGETHER.

    `weights` (e.g. illumination) restricts the comparison to cells both
    datasets actually constrain -- comparing unilluminated cells would measure
    the starting model, not the inversion.
    """
    a, b = np.asarray(vp_a, float), np.asarray(vp_b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    m = (w > 0) & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return dict(rms_diff=float("nan"), max_diff=float("nan"),
                    rel_diff_pct=float("nan"), n_cells=0)
    d = a[m] - b[m]
    return dict(rms_diff=float(np.sqrt((d ** 2).mean())),
                max_diff=float(np.abs(d).max()),
                rel_diff_pct=float(100.0 * np.sqrt((d ** 2).mean())
                                   / np.abs(a[m]).mean()),
                n_cells=int(m.sum()))


# --------------------------------------------------------------------------- #
# 3. well-log comparison -- READY for the 58-32 sonic, which we do not yet hold
# --------------------------------------------------------------------------- #
def compare_to_log(vp_column, z_model, log_z, log_vp, z_range=None):
    """Recovered Vp against a sonic log at one well.

    `vp_column` is the model's velocity down the well; `z_model` its depths.
    The log is resampled onto the model depths, because a sonic log is far
    finer than a 10 m FWI grid and comparing at log resolution would score the
    inversion against detail it cannot represent -- which would be unfair, not
    rigorous.
    """
    zm = np.asarray(z_model, float)
    v = np.asarray(vp_column, float)
    lz, lv = np.asarray(log_z, float), np.asarray(log_vp, float)
    o = np.argsort(lz)
    ref = np.interp(zm, lz[o], lv[o], left=np.nan, right=np.nan)
    m = np.isfinite(ref) & np.isfinite(v)
    if z_range is not None:
        m &= (zm >= z_range[0]) & (zm <= z_range[1])
    if m.sum() < 3:
        return dict(n=int(m.sum()), rms=float("nan"), bias=float("nan"),
                    corr=float("nan"))
    d = v[m] - ref[m]
    return dict(n=int(m.sum()), rms=float(np.sqrt((d ** 2).mean())),
                bias=float(d.mean()),
                corr=float(np.corrcoef(v[m], ref[m])[0, 1]))


def zone_boundaries(vp_column, z_model, n_zones=3, smooth=3):
    """Depths of the strongest velocity increases -- the I/II/III transitions.

    Park identify three zones from cuttings density: unconsolidated alluvium,
    consolidated alluvium, granitoid basement. Their DEPTHS are readable off the
    published figure even without the log traces, so boundary alignment is
    checkable before we obtain any log data.
    """
    v = np.asarray(vp_column, float)
    z = np.asarray(z_model, float)
    if smooth > 1:
        # EDGE-PAD before smoothing. np.convolve(..., "same") ZERO-pads, which
        # drags the first sample toward 0 and fabricates a huge gradient there
        # -- it reported a zone boundary at z=0 that does not exist.
        k = np.ones(int(smooth)) / float(smooth)
        pad = int(smooth)
        v = np.convolve(np.pad(v, pad, mode="edge"), k, mode="same")[pad:-pad]
    g = np.gradient(v, z)
    idx = np.argsort(g)[::-1]
    picked = []
    for i in idx:                       # keep peaks separated by >5 samples
        if all(abs(i - j) > 5 for j in picked):
            picked.append(int(i))
        if len(picked) >= n_zones - 1:
            break
    return sorted(float(z[i]) for i in picked)


def boundary_alignment(vp_column, z_model, expected_depths, tol_m=100.0):
    """Do the recovered transitions sit at the expected zone depths?"""
    got = zone_boundaries(vp_column, z_model, n_zones=len(expected_depths) + 1)
    exp = sorted(float(d) for d in expected_depths)
    n = min(len(got), len(exp))
    errs = [abs(got[i] - exp[i]) for i in range(n)]
    return dict(expected=exp, recovered=got,
                errors_m=errs, n_matched=int(sum(e <= tol_m for e in errs)),
                all_within_tol=bool(errs and all(e <= tol_m for e in errs)))


#: ACCEPTANCE THRESHOLDS, fixed BEFORE any field result is seen.
#: Set deliberately at "comparable to the published work", not at whatever we
#: happen to achieve -- a threshold chosen after seeing the answer is not a test.
ACCEPT = dict(
    mismatch_reduction_pct=51.7,   # Park INV2; we use no picking at all
    cross_well_rel_diff_pct=10.0,  # independent wells should agree to ~10%
    log_rms_ms=400.0,              # m/s vs the 58-32 sonic, once we have it
    boundary_tol_m=100.0,          # zone I/II/III depths, ~10 grid cells
)
