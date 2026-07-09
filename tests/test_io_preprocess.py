"""Tests for forge/io_preprocess.py (T8) - synthetic signals only."""

import numpy as np
import pytest

from forge.io_preprocess import (detrend_demean, remove_common_mode, bandpass,
                                 mute_near_offset, fk_fan_reject, cosine_taper,
                                 apply_chain, snr_spectrum, usable_f_low,
                                 rms_vp, granitoid_boundary_depth,
                                 boundary_depth_error, ladder_pass,
                                 GRANITOID_DEPTH_M)

DT = 1e-3          # 1 kHz
DCH = 5.0          # channel spacing on the 5 m grid


def linear_event(T, C, dt, dch, v, f0=25.0, t0=0.05, amp=1.0):
    """Gaussian-enveloped wavelet with moveout t = t0 + x/v, [T, C].

    Edge-tapered across channels: without it, the hard spatial truncation
    smears the event's FK ridge across k, and no velocity fan can isolate it.
    (The k-resolution argument also sets the test aperture: with C channels
    the fan can only resolve dv/v ~ 1/(C*dch*k); the FK test below uses
    C = 256, i.e. a 1.28 km aperture, comparable to a field fiber.)"""
    from scipy.signal.windows import tukey
    t = np.arange(T) * dt
    x = np.arange(C) * dch
    tt = t[:, None] - (t0 + x[None, :] / v)
    d = amp * np.exp(-(tt / 0.008) ** 2) * np.cos(2 * np.pi * f0 * tt)
    return d * tukey(C, 0.4)[None, :]


def test_detrend_demean():
    T, C = 500, 12
    rng = np.random.default_rng(0)
    base = rng.standard_normal((T, C))
    trend = np.linspace(0, 3.0, T)[:, None] + 5.0
    out = detrend_demean(base + trend)
    assert np.abs(out.mean(axis=0)).max() < 1e-10
    # the linear trend is gone: correlation with a ramp ~ 0
    ramp = np.linspace(-1, 1, T)
    corr = np.abs((out * ramp[:, None]).mean(axis=0)) / out.std(axis=0)
    assert corr.max() < 0.05


def test_common_mode_removal_kills_common_signal():
    T, C = 400, 30
    rng = np.random.default_rng(1)
    common = np.sin(2 * np.pi * 12.0 * np.arange(T) * DT)[:, None]
    unique = 0.01 * rng.standard_normal((T, C))
    out = remove_common_mode(common + unique)
    # the injected common signal (unit amplitude) is annihilated
    resid_common = np.median(out, axis=1)
    assert np.abs(resid_common).max() < 1e-12
    assert np.abs(out).max() < 0.1   # only the small unique part remains


def test_bandpass_passes_in_band_rejects_out():
    T, C = 2000, 4
    t = np.arange(T) * DT
    in_band = np.sin(2 * np.pi * 10.0 * t)[:, None] * np.ones((1, C))
    out_band = np.sin(2 * np.pi * 90.0 * t)[:, None] * np.ones((1, C))
    f = bandpass(in_band + out_band, DT, 5.0, 20.0)
    mid = slice(400, 1600)   # avoid filter edge effects
    # 10 Hz survives, 90 Hz is strongly attenuated
    assert np.sqrt((f[mid] ** 2).mean()) > 0.5
    resid_90 = f[mid] - bandpass(in_band, DT, 5.0, 20.0)[mid]
    assert np.sqrt((resid_90 ** 2).mean()) < 0.02


def test_mute_near_offset():
    d = np.ones((100, 5))
    offsets = np.array([-120.0, -60.0, 0.0, 60.0, 120.0])
    out = mute_near_offset(d, offsets, min_offset_m=100.0)
    assert np.all(out[:, [1, 2, 3]] == 0)
    assert np.all(out[:, [0, 4]] == 1)


def test_fk_fan_rejects_tube_wave_passes_fast_event():
    # E14 chain order matters: bandpass runs BEFORE the FK fan, so the fan
    # never sees the sub-band DC pedestal of the events (near k=0 the
    # apparent-velocity fan cannot discriminate; that energy is out of band).
    T, C = 1024, 256
    tube = bandpass(linear_event(T, C, DT, DCH, v=1800.0), DT, 5.0, 60.0)
    fast = bandpass(linear_event(T, C, DT, DCH, v=5500.0), DT, 5.0, 60.0)

    tube_out = fk_fan_reject(tube, DT, DCH)
    fast_out = fk_fan_reject(fast, DT, DCH)

    def db(x, y):
        return 20 * np.log10(np.sqrt((y ** 2).mean())
                             / np.sqrt((x ** 2).mean()))

    atten_tube = -db(tube, tube_out)
    atten_fast = -db(fast, fast_out)
    assert atten_tube >= 20.0, f"tube wave only attenuated {atten_tube:.1f} dB"
    assert atten_fast <= 3.0, f"5500 m/s event lost {atten_fast:.1f} dB"


def test_cosine_taper_zeros_ends():
    d = np.ones((300, 7))
    out = cosine_taper(d, taper_frac=0.05)
    assert np.all(out[0] == 0) and np.all(out[-1] == 0)
    assert np.all(out[140:160] == 1.0)   # interior untouched


def test_apply_chain_runs_and_ends_tapered():
    T, C = 600, 32
    rng = np.random.default_rng(2)
    d = (linear_event(T, C, DT, DCH, v=5500.0, f0=12.0, t0=0.1)
         + 0.05 * rng.standard_normal((T, C)))
    offsets = np.linspace(-160, 160, C)
    out = apply_chain(d, DT, DCH, offsets, band=(5.0, 30.0),
                      min_offset_m=20.0)
    assert out.shape == (T, C)
    assert np.all(out[0] == 0) and np.all(out[-1] == 0)
    assert np.isfinite(out).all()


def test_snr_recovers_known_f_low():
    """Construct signal+noise where SNR(f) >= 2 only above ~15 Hz.

    C = 48 channels: the SNR estimate averages per-channel periodograms, and
    with too few channels the ratio's variance lets a low-f noise spike cross
    the threshold and corrupt f_low."""
    T, C = 4096, 48
    dt = 1e-3
    rng = np.random.default_rng(3)
    fb = np.full(C, 2048)
    win = 1024
    t = np.arange(T) * dt

    # white noise everywhere; signal = strong 20-40 Hz + weak 5-12 Hz band.
    # Strong amplitude kept moderate: even Hann-window sidelobes of a very
    # strong in-band line would leak into the low-f bins and fake SNR there.
    d = 1.0 * rng.standard_normal((T, C))
    strong = np.sin(2 * np.pi * 25.0 * t) + np.sin(2 * np.pi * 35.0 * t)
    weak = 0.05 * np.sin(2 * np.pi * 8.0 * t)
    for c in range(C):
        d[fb[c]:, c] += 2.0 * strong[fb[c]:] + weak[fb[c]:]

    freqs, snr = snr_spectrum(d, dt, fb, signal_len=win)
    f_low = usable_f_low(freqs, snr, threshold=2.0)
    # weak band (8 Hz) must be below threshold, strong band above
    assert 12.0 <= f_low <= 30.0, f"recovered f_low = {f_low}"
    assert snr[np.argmin(np.abs(freqs - 25.0))] >= 2.0
    assert snr[np.argmin(np.abs(freqs - 8.0))] < 2.0


def test_rms_vp_and_boundary():
    z = np.arange(0, 1500.0, 5.0)
    v_log = 2000.0 + z
    assert rms_vp(v_log + 100.0, v_log) == pytest.approx(100.0)

    z_air = 100.0
    vp = np.where(z >= GRANITOID_DEPTH_M + z_air, 5600.0, 3000.0)
    assert granitoid_boundary_depth(vp, z) == pytest.approx(925.0)  # next node
    assert boundary_depth_error(vp, z, z_air) <= 5.0                # <= dz
    assert np.isnan(granitoid_boundary_depth(np.full_like(z, 3000.0), z))


def test_ladder_pass():
    assert ladder_pass(j_final=0.1, j0=1.0, dz_b=10.0, rms_v=100.0,
                       tau_j=0.2, tau_b=25.0, tau_v=150.0)
    # each criterion individually fails the cell
    assert not ladder_pass(0.5, 1.0, 10.0, 100.0, 0.2, 25.0, 150.0)
    assert not ladder_pass(0.1, 1.0, 30.0, 100.0, 0.2, 25.0, 150.0)
    assert not ladder_pass(0.1, 1.0, 10.0, 200.0, 0.2, 25.0, 150.0)
