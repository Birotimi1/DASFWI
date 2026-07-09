"""Field-data preprocessing, LF diagnostic, validation metrics (T8).

CODE + UNIT TESTS ONLY at local stage - no real data is read or processed
here; run_field.py / run_ladder.py wire these into HPC campaigns later.

Gather convention: 2-D float arrays ``d[T, C]`` (time samples x channels),
matching the per-shot layout of ADFWI records and the DAS layer output.

E14 preprocessing chain, IN ORDER (each step is a pure function; apply_chain
runs them in the fixed spec order):
    1. detrend/demean (per channel)
    2. common-mode removal        d_c <- d_c - median_{c'}(d_{c'})   (per t)
    3. per-band bandpass          (zero-phase Butterworth)
    4. near-offset mute           (zero channels with |offset| < min_offset)
    5. tube-wave FK fan rejection around v = f/k ~ 1800 m/s (+- tolerance)
    6. cosine tapers so h(t0) = h(t1) = 0 (misfit windows are NEVER
       hard-edged; spec hard rule 4)

E16 LF diagnostic: SNR(f) = P_signal(f) / P_noise(f) per channel (signal
window vs pre-first-break noise window); usable f_low = min{f: SNR(f) >= 2}.

E17 validation metrics: RMS_Vp = sqrt(mean_z[(v_FWI - v_log)^2]) over the
58-32 sonic interval; granitoid boundary-depth error vs 823 m (+ air offset)
in 78B-32.

E18 ladder criterion: per-cell PASS iff
    (J_final/J_0 <= tau_J)  AND  (dz_b <= tau_b)  AND  (RMS_Vp <= tau_v),
with thresholds FROZEN before the ladder runs.
"""

import numpy as np
from scipy import signal as sp_signal

TUBE_WAVE_V = 1800.0          # m/s, FORGE tube-wave apparent velocity
GRANITOID_DEPTH_M = 823.0     # m below ground surface, well 78B-32
SNR_THRESHOLD = 2.0


# --------------------------------------------------------------------------- #
# E14 - preprocessing chain
# --------------------------------------------------------------------------- #

def detrend_demean(d):
    """Remove per-channel linear trend and mean (first chain step)."""
    d = np.asarray(d, dtype=np.float64)
    return sp_signal.detrend(d, axis=0, type="linear")


def remove_common_mode(d):
    """Common-mode removal: subtract the cross-channel median per time sample,
    d_c <- d_c - median_{c'}(d_{c'})  (kills cable-wide coherent noise)."""
    d = np.asarray(d, dtype=np.float64)
    return d - np.median(d, axis=1, keepdims=True)


def bandpass(d, dt, f_lo, f_hi, order=4):
    """Zero-phase Butterworth bandpass along time (per multiscale band)."""
    d = np.asarray(d, dtype=np.float64)
    nyq = 0.5 / dt
    b, a = sp_signal.butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    return sp_signal.filtfilt(b, a, d, axis=0)


def mute_near_offset(d, offsets_m, min_offset_m):
    """Zero channels whose |source-channel offset| is below min_offset_m."""
    d = np.asarray(d, dtype=np.float64).copy()
    offsets_m = np.asarray(offsets_m, dtype=np.float64)
    d[:, np.abs(offsets_m) < min_offset_m] = 0.0
    return d


def fk_fan_reject(d, dt, dch, v_reject=TUBE_WAVE_V, tol=0.25,
                  transition=0.05):
    """Reject the FK fan of apparent velocities around v_reject (+- tol).

    Events with moveout t = x/v map onto the FK line |f| = v |k|; the mask
    attenuates |f|/|k| inside [v_reject*(1-tol), v_reject*(1+tol)] for BOTH
    propagation directions (up- and down-going tube waves), with a cosine
    transition of relative width `transition` at the fan edges.

    Args:
        d: gather [T, C].  dt: time step [s].  dch: channel spacing [m].
    """
    d = np.asarray(d, dtype=np.float64)
    T, C = d.shape
    F = np.fft.fft2(d)
    f = np.fft.fftfreq(T, dt)[:, None]        # [T, 1] temporal frequency
    k = np.fft.fftfreq(C, dch)[None, :]       # [1, C] spatial frequency

    with np.errstate(divide="ignore", invalid="ignore"):
        v_app = np.abs(f) / np.abs(k)          # inf on k == 0 (kept)
    v_app[np.isnan(v_app)] = np.inf            # f == 0 and k == 0

    v_lo, v_hi = v_reject * (1 - tol), v_reject * (1 + tol)
    t_lo, t_hi = v_lo * (1 - transition), v_hi * (1 + transition)

    gain = np.ones_like(v_app)
    inside = (v_app >= v_lo) & (v_app <= v_hi)
    gain[inside] = 0.0
    lo_edge = (v_app >= t_lo) & (v_app < v_lo)
    gain[lo_edge] = 0.5 * (1 - np.cos(np.pi * (v_lo - v_app[lo_edge])
                                      / (v_lo - t_lo)))
    hi_edge = (v_app > v_hi) & (v_app <= t_hi)
    gain[hi_edge] = 0.5 * (1 - np.cos(np.pi * (v_app[hi_edge] - v_hi)
                                      / (t_hi - v_hi)))

    return np.real(np.fft.ifft2(F * gain))


def cosine_taper(d, taper_frac=0.05):
    """Tukey taper along time so the window ends are exactly zero:
    h(t0) = h(t1) = 0 (misfit windows must never be hard-edged)."""
    d = np.asarray(d, dtype=np.float64)
    w = sp_signal.windows.tukey(d.shape[0], alpha=2 * taper_frac)
    w[0] = w[-1] = 0.0
    return d * w[:, None]


def apply_chain(d, dt, dch, offsets_m, band=(2.0, 20.0), min_offset_m=0.0,
                v_reject=TUBE_WAVE_V, v_tol=0.25, taper_frac=0.05):
    """Full E14 chain in the fixed spec order."""
    d = detrend_demean(d)
    d = remove_common_mode(d)
    d = bandpass(d, dt, band[0], band[1])
    if min_offset_m > 0:
        d = mute_near_offset(d, offsets_m, min_offset_m)
    d = fk_fan_reject(d, dt, dch, v_reject=v_reject, tol=v_tol)
    d = cosine_taper(d, taper_frac=taper_frac)
    return d


# --------------------------------------------------------------------------- #
# E16 - low-frequency SNR diagnostic
# --------------------------------------------------------------------------- #

def snr_spectrum(d, dt, first_break_idx, signal_len, noise_gap=0):
    """Per-frequency SNR from per-channel signal vs pre-first-break noise.

    SNR(f) = P_signal(f) / P_noise(f), with P the channel-averaged
    HANN-WINDOWED periodogram of the signal window [fb, fb + signal_len) and
    of the noise window ending noise_gap samples BEFORE the first break
    (equal length). The Hann window is essential: with a rectangular window,
    sidelobe leakage of strong in-band signal inflates the low-frequency SNR
    and corrupts the f_low estimate.

    Args:
        d: gather [T, C]. first_break_idx: [C] first-break sample per channel.
        signal_len: window length in samples.

    Returns:
        (freqs, snr): rfft frequencies [Hz] and channel-averaged SNR(f).
    """
    d = np.asarray(d, dtype=np.float64)
    fb = np.asarray(first_break_idx, dtype=int)
    C = d.shape[1]
    p_sig = np.zeros((signal_len // 2 + 1,))
    p_noi = np.zeros_like(p_sig)
    hann = sp_signal.windows.hann(signal_len)
    n_used = 0
    for c in range(C):
        n_end = fb[c] - noise_gap
        if n_end - signal_len < 0 or fb[c] + signal_len > d.shape[0]:
            continue   # not enough room for equal-length windows
        sig = d[fb[c]:fb[c] + signal_len, c] * hann
        noi = d[n_end - signal_len:n_end, c] * hann
        p_sig += np.abs(np.fft.rfft(sig)) ** 2
        p_noi += np.abs(np.fft.rfft(noi)) ** 2
        n_used += 1
    if n_used == 0:
        raise ValueError("no channel has room for both windows")
    freqs = np.fft.rfftfreq(signal_len, dt)
    return freqs, p_sig / np.maximum(p_noi, 1e-300)


def usable_f_low(freqs, snr, threshold=SNR_THRESHOLD):
    """E16: usable f_low = min{f : SNR(f) >= threshold} (NaN if never)."""
    ok = np.asarray(snr) >= threshold
    return float(freqs[np.argmax(ok)]) if ok.any() else float("nan")


# --------------------------------------------------------------------------- #
# E17 - validation metrics
# --------------------------------------------------------------------------- #

def rms_vp(v_fwi, v_log):
    """RMS_Vp = sqrt(mean_z[(v_FWI - v_log)^2]) over the (pre-sliced) sonic
    interval of well 58-32."""
    v_fwi = np.asarray(v_fwi, dtype=np.float64)
    v_log = np.asarray(v_log, dtype=np.float64)
    return float(np.sqrt(np.mean((v_fwi - v_log) ** 2)))


def granitoid_boundary_depth(vp_profile, z, v_threshold=5000.0):
    """First depth at which the vp profile reaches the granitoid velocity."""
    vp_profile = np.asarray(vp_profile)
    idx = np.argmax(vp_profile >= v_threshold)
    if vp_profile[idx] < v_threshold:
        return float("nan")
    return float(np.asarray(z)[idx])


def boundary_depth_error(vp_profile, z, z_air, v_threshold=5000.0):
    """E17: |estimated - true| granitoid boundary depth in 78B-32, with the
    true boundary at 823 m below ground surface (+ z_air model offset)."""
    zb = granitoid_boundary_depth(vp_profile, z, v_threshold)
    return abs(zb - (GRANITOID_DEPTH_M + z_air))


# --------------------------------------------------------------------------- #
# E18 - degradation-ladder criterion
# --------------------------------------------------------------------------- #

def ladder_pass(j_final, j0, dz_b, rms_v, tau_j, tau_b, tau_v):
    """Per-cell PASS iff (J_final/J_0 <= tau_J) AND (dz_b <= tau_b) AND
    (RMS_Vp <= tau_v). Freeze the thresholds BEFORE the ladder runs."""
    return bool((j_final / j0 <= tau_j) and (dz_b <= tau_b)
                and (rms_v <= tau_v))
