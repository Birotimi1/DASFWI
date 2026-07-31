"""Cycle-skipping diagnostic for DAS strain-rate FWI.

Cycle skipping occurs when the kinematic misalignment between synthetic and
observed traces exceeds half a period:

        |dt|  >  T/2  =  1 / (2 * f_max)

so this module measures, per trace, the time shift that maximises the
cross-correlation of syn and obs, and reports the FRACTION of traces beyond that
threshold. That fraction is the quantity Phase 1 correlates with L2 failure, and
the quantity Phase 3 uses to certify a starting model as "skip-safe".

IMPORTANT - which f_max to use. The campaign source is an INTEGRATED Ricker
(cumtrapz of a Ricker), which shifts the spectrum DOWN: for the nominal
f0 = 5 Hz source the measured spectrum peaks at 3.54 Hz with 90% of the energy
below 6.25 Hz. So the honest threshold is T/2 ~ 80 ms, NOT 1/(2*f0) = 100 ms.
Use `ricker_f90()` (or the current multiscale band's cut-off) rather than the
nominal f0.

Data layout follows the campaign: (n_shots, nt, n_channels), time on axis 1.
Everything is torch, batched over shots/channels, and fully detached - this is a
diagnostic, never part of the gradient.
"""
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# source-spectrum helper
# --------------------------------------------------------------------------- #
def ricker_f90(f0, dt, nt, integrated=True, frac=0.90):
    """Frequency below which `frac` of the source energy lies.

    The campaign integrates the Ricker (Liu's setup), which lowers the spectrum;
    pass integrated=False for a plain Ricker. Returns Hz.
    """
    t = (np.arange(nt) - nt // 2) * dt
    w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)
    if integrated:
        w = np.cumsum(w) * dt                      # ~ cumtrapz, same spectrum
    P = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(nt, dt)
    c = np.cumsum(P) / P.sum()
    return float(f[np.searchsorted(c, frac)])


def skip_threshold(f_max):
    """Half-period cycle-skip threshold (seconds) for a max frequency in Hz."""
    return 1.0 / (2.0 * float(f_max))


# --------------------------------------------------------------------------- #
# per-trace lag by cross-correlation
# --------------------------------------------------------------------------- #
def trace_lags(syn, obs, dt, max_lag_s=None, time_axis=1, min_rel_amp=1e-6):
    """Per-trace cross-correlation lag, in seconds.

    Args:
        syn, obs: tensors/arrays shaped (n_shots, nt, n_chan) by default.
        dt: sample interval (s).
        max_lag_s: restrict the search to |lag| <= this (default nt*dt/4).
            Keeps the pick on the main arrival instead of a distant side lobe.
        min_rel_amp: traces whose max|.| is below this fraction of the global
            max are treated as DEAD (DAS has blind/dead channels) and returned
            as NaN so they can be excluded from statistics.

    Returns:
        lag: (n_shots, n_chan) tensor, seconds. POSITIVE means the synthetic
            arrives LATE relative to the observed (syn must shift earlier).
            NaN for dead traces.
        peak: (n_shots, n_chan) normalised correlation at that lag, in [-1, 1]
            (NaN for dead traces) - a quality/confidence measure.
    """
    syn = torch.as_tensor(syn).detach()
    obs = torch.as_tensor(obs).detach()
    if syn.shape != obs.shape:
        raise ValueError(f"shape mismatch: syn {tuple(syn.shape)} vs obs {tuple(obs.shape)}")
    # move time to the last axis -> (..., nt)
    a = torch.movedim(syn.to(torch.float64), time_axis, -1)
    b = torch.movedim(obs.to(torch.float64), time_axis, -1)
    nt = a.shape[-1]

    # dead-trace mask (before demeaning), relative to the global observed max
    gmax = torch.maximum(a.abs().amax(), b.abs().amax()).clamp_min(1e-300)
    dead = ((a.abs().amax(-1) < min_rel_amp * gmax)
            | (b.abs().amax(-1) < min_rel_amp * gmax))

    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    na = a.norm(dim=-1).clamp_min(1e-300)
    nb = b.norm(dim=-1).clamp_min(1e-300)

    # linear (non-circular) cross-correlation via zero-padded FFT
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    R = torch.fft.rfft(b, n=nfft) * torch.fft.rfft(a, n=nfft).conj()
    r = torch.fft.irfft(R, n=nfft)                 # r[k] = sum_t b[t+k] a[t]
    # roll so index nt corresponds to lag 0; lags run -nt .. nfft-nt-1
    r = torch.roll(r, shifts=nt, dims=-1)
    lags = torch.arange(nfft, device=r.device) - nt

    if max_lag_s is None:
        max_lag_s = nt * dt / 4.0
    max_lag = int(round(max_lag_s / dt))
    keep = (lags.abs() <= max_lag)
    r = r[..., keep]
    lags = lags[keep]

    k = r.argmax(dim=-1)
    peak = torch.gather(r, -1, k.unsqueeze(-1)).squeeze(-1) / (na * nb)
    # r[k] = sum_t obs[t+k] syn[t]. If syn is delayed by tau (syn[t]=obs[t-tau])
    # the peak sits at k = -tau, so NEGATE to report tau itself: positive lag
    # means the SYNTHETIC ARRIVES LATE (model too slow) - the intuitive sign.
    lag = -lags[k].to(torch.float64) * dt

    nan = torch.tensor(float("nan"), dtype=lag.dtype, device=lag.device)
    return torch.where(dead, nan, lag), torch.where(dead, nan, peak)


def skip_vs_band(syn, obs, dt, bands, max_lag_s=None, time_axis=1,
                 min_peak=None):
    """Skip fraction at SEVERAL band limits from ONE measurement.

    The arrival-time lags are a property of the two models, not of the band;
    only the threshold T/2 = 1/(2 f_max) changes. So the whole skip-vs-frequency
    curve costs one cross-correlation, not one forward per band.

    This is how to CHOOSE the non-skip and skip test bands for a given starting
    model: take a band whose skip fraction sits below the controller's off_below
    (L2 is safe -> the non-skip case) and one above its on_above (L2 should fail
    -> the skip case), rather than guessing which frequencies bracket the flip.

    Returns a list of dicts: f_max, threshold_s, skip_fraction, n_live.
    """
    lag, peak = trace_lags(syn, obs, dt, max_lag_s=max_lag_s, time_axis=time_axis)
    live = torch.isfinite(lag)
    if min_peak is not None:
        live = live & (peak > min_peak)
    n_live = int(live.sum())
    out = []
    for f in bands:
        thr = skip_threshold(float(f))
        sf = (float("nan") if n_live == 0 else
              float((lag[live].abs() > thr).to(torch.float64).mean()))
        out.append(dict(f_max=float(f), threshold_s=thr, skip_fraction=sf,
                        n_live=n_live))
    return out


def skip_fraction(syn, obs, dt, f_max, max_lag_s=None, time_axis=1,
                  min_peak=None):
    """Fraction of live traces that are cycle-skipped, plus lag statistics.

    Args:
        f_max: max frequency of the CURRENT data band (use ricker_f90() or the
            multiscale band cut-off, NOT the nominal f0).
        min_peak: if set, also ignore traces whose peak correlation is below
            this (unreliable lag estimate).

    Returns dict:
        skip_fraction : fraction of live traces with |lag| > 1/(2 f_max)
        threshold_s   : the half-period threshold used
        mean_abs_lag_s, median_abs_lag_s, p90_abs_lag_s
        mean_peak     : mean normalised correlation (alignment quality)
        n_live, n_total
    """
    lag, peak = trace_lags(syn, obs, dt, max_lag_s=max_lag_s, time_axis=time_axis)
    live = torch.isfinite(lag)
    if min_peak is not None:
        live = live & (peak > min_peak)
    n_live = int(live.sum())
    thr = skip_threshold(f_max)
    if n_live == 0:
        return dict(skip_fraction=float("nan"), threshold_s=thr,
                    mean_abs_lag_s=float("nan"), median_abs_lag_s=float("nan"),
                    p90_abs_lag_s=float("nan"), mean_peak=float("nan"),
                    n_live=0, n_total=int(lag.numel()))
    al = lag[live].abs()
    return dict(
        skip_fraction=float((al > thr).to(torch.float64).mean()),
        threshold_s=thr,
        mean_abs_lag_s=float(al.mean()),
        median_abs_lag_s=float(al.median()),
        p90_abs_lag_s=float(torch.quantile(al, 0.90)),
        mean_peak=float(peak[live].mean()),
        n_live=n_live, n_total=int(lag.numel()),
    )
