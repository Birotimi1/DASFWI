"""Tests for forge/traveltime_tomography.py (first-break starting model)."""

import numpy as np
import pytest

from forge.traveltime_tomography import (pick_first_breaks,
                                         vsp_checkshot_velocity,
                                         build_starting_model,
                                         starting_model_from_gathers)


def ricker(n, f0, dt):
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f0) ** 2
    return (1 - 2 * a * t ** 2) * np.exp(-a * t ** 2)


# --------------------------------------------------------------------------- #
# first-break picking
# --------------------------------------------------------------------------- #
def test_picker_recovers_known_onset():
    nt, C, dt = 600, 20, 1e-3
    onset = 120 + 6 * np.arange(C)                 # known onset sample per trace
    wav = ricker(60, 30.0, dt)
    g = np.zeros((nt, C))
    rng = np.random.default_rng(0)
    g += 0.002 * rng.standard_normal((nt, C))      # mild noise
    for c in range(C):
        s = onset[c]
        g[s:s + len(wav), c] += wav
    picks = pick_first_breaks(g, dt, sta_s=0.008, lta_s=0.04, threshold=3.0)
    pick_samp = picks / dt
    assert np.isfinite(pick_samp).all(), "some traces failed to pick"
    # STA/LTA fires partway up the energy ramp, so it lags the true onset by a
    # ~constant amount. That constant lag is harmless -- it cancels in
    # v = dz/dt -- so the physically meaningful check is that the MOVEOUT
    # (relative timing) is faithful: pick - onset is consistent across traces.
    lag = pick_samp - onset
    assert lag.std() < 4.0, f"moveout not faithful: lag std {lag.std():.1f}"
    assert 0.0 <= lag.mean() < 30.0, f"unphysical lag mean {lag.mean():.1f}"


# --------------------------------------------------------------------------- #
# 1-D velocity from first breaks
# --------------------------------------------------------------------------- #
def test_checkshot_velocity_linear_gradient():
    # true v(z) = a + b z ; vertical time t(z) = (1/b) ln((a+b z)/a)
    a, b = 2000.0, 0.6
    z = np.arange(60.0, 1000.0, 10.0)
    t_vert = (1.0 / b) * np.log((a + b * z) / a)
    z_out, v_out = vsp_checkshot_velocity(t_vert, z, x_offset=0.0, smooth_n=3)
    v_true = a + b * z_out
    # compare at mid-depths (ends noisier from the gradient/smoothing)
    mid = (z_out > 200) & (z_out < 800)
    rel = np.abs(v_out[mid] - v_true[mid]) / v_true[mid]
    assert rel.max() < 0.05, f"v(z) rel error {rel.max():.3f}"


def test_checkshot_offset_deskew():
    # picks are straight-ray times; deskew must recover the vertical profile
    a, b, x = 2000.0, 0.6, 150.0
    z = np.arange(60.0, 1200.0, 10.0)
    t_vert = (1.0 / b) * np.log((a + b * z) / a)
    t_straight = t_vert * np.sqrt(z ** 2 + x ** 2) / z    # what we'd pick
    z_out, v_out = vsp_checkshot_velocity(t_straight, z, x_offset=x, smooth_n=3)
    v_true = a + b * z_out
    mid = (z_out > 300) & (z_out < 1000)                  # deskew best when z>>x
    rel = np.abs(v_out[mid] - v_true[mid]) / v_true[mid]
    assert rel.max() < 0.08, f"deskew v(z) rel error {rel.max():.3f}"


# --------------------------------------------------------------------------- #
# 2-D assembly
# --------------------------------------------------------------------------- #
def test_build_starting_model():
    z_prof = np.arange(100.0, 1000.0, 20.0)
    v_prof = 2000.0 + 0.6 * z_prof
    vp = build_starting_model(z_prof, v_prof, nz=220, nx=80, dz=5.0)
    assert vp.shape == (220, 80)
    assert np.allclose(vp, vp[:, :1])                     # laterally constant
    assert vp.min() >= 1400.0 and vp.max() <= 6500.0
    # increases with depth in the profiled interval
    assert vp[180, 0] > vp[40, 0]


def test_end_to_end():
    # synthetic VSP: build first-break gather from a known v(z), recover it
    nt, dt, x = 900, 1e-3, 100.0
    a, b = 2000.0, 0.6
    z = np.arange(80.0, 1000.0, 12.0)
    C = z.size
    t_vert = (1.0 / b) * np.log((a + b * z) / a)
    t_pick = t_vert * np.sqrt(z ** 2 + x ** 2) / z         # straight-ray onset
    wav = ricker(60, 25.0, dt)
    g = np.zeros((nt, C))
    for c in range(C):
        s = int(round(t_pick[c] / dt))
        if s + len(wav) < nt:
            g[s:s + len(wav), c] += wav
    vp, z_prof, v_prof, picks = starting_model_from_gathers(
        g, dt, z, x_offset=x, nz=220, nx=40, dz=5.0,
        sta_s=0.008, lta_s=0.04, threshold=3.0)
    assert vp.shape == (220, 40)
    # recovered velocity tracks the truth in the well-illuminated interval
    mid = (z_prof > 300) & (z_prof < 900)
    v_true = a + b * z_prof
    rel = np.abs(v_prof[mid] - v_true[mid]) / v_true[mid]
    assert rel.max() < 0.12, f"end-to-end v(z) rel error {rel.max():.3f}"


def test_too_few_picks_raises():
    with pytest.raises(ValueError):
        vsp_checkshot_velocity(np.array([0.1, np.nan]), np.array([100.0, 200.0]))
