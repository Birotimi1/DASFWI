"""Cycle-skip diagnostic: recover known time shifts and the skip fraction."""
import numpy as np
import pytest
import torch

from inversion.skip_diagnostic import (ricker_f90, skip_threshold, trace_lags,
                                       skip_fraction)

DT, NT, F0 = 0.003, 1600, 5.0
A = 2.3e-8                       # realistic DAS strain-rate amplitude


def _gather(shifts, amp=A):
    """(1, nt, n_chan) of Ricker wavelets, one per channel, at given shifts."""
    t = np.arange(NT) * DT
    out = np.empty((1, NT, len(shifts)))
    for j, s in enumerate(shifts):
        tt = t - (0.9 + s)
        out[0, :, j] = amp * (1 - 2 * (np.pi * F0 * tt) ** 2) * \
            np.exp(-(np.pi * F0 * tt) ** 2)
    return torch.from_numpy(out)


def test_recovers_known_shifts():
    """Lag must equal the imposed shift, to within one sample."""
    shifts = np.array([0.0, 0.03, -0.03, 0.12, -0.21])
    obs = _gather(np.zeros_like(shifts))
    syn = _gather(shifts)                      # synthetic late by `shifts`
    lag, peak = trace_lags(syn, obs, DT)
    assert lag.shape == (1, len(shifts))
    assert np.allclose(lag.numpy()[0], shifts, atol=DT + 1e-12)
    assert (peak.numpy() > 0.99).all()         # identical wavelets -> corr ~ 1


def test_skip_fraction_matches_threshold():
    """Only traces beyond T/2 count as skipped."""
    f_max = 6.25                               # -> threshold 80 ms
    thr = skip_threshold(f_max)
    assert thr == pytest.approx(0.08)
    # 2 of 5 shifts exceed 80 ms
    shifts = np.array([0.0, 0.05, -0.05, 0.15, -0.30])
    obs, syn = _gather(np.zeros_like(shifts)), _gather(shifts)
    st = skip_fraction(syn, obs, DT, f_max)
    assert st["skip_fraction"] == pytest.approx(2 / 5)
    assert st["n_live"] == 5
    assert st["mean_abs_lag_s"] == pytest.approx(np.abs(shifts).mean(),
                                                 abs=DT + 1e-12)


def test_zero_shift_gives_zero_skip():
    obs = _gather(np.zeros(4))
    st = skip_fraction(obs.clone(), obs, DT, 6.25)
    assert st["skip_fraction"] == 0.0
    assert st["mean_peak"] == pytest.approx(1.0, abs=1e-6)


def test_dead_traces_excluded():
    """DAS blind/dead channels must not be counted as live (or as skipped)."""
    shifts = np.array([0.0, 0.20, 0.0])
    obs, syn = _gather(np.zeros_like(shifts)), _gather(shifts)
    obs[0, :, 2] = 0.0                          # kill one channel
    syn[0, :, 2] = 0.0
    st = skip_fraction(syn, obs, DT, 6.25)
    assert st["n_live"] == 2 and st["n_total"] == 3
    assert st["skip_fraction"] == pytest.approx(0.5)   # 1 of the 2 live traces


def test_amplitude_invariance():
    """Kinematic diagnostic: scaling either side must not change the lag."""
    shifts = np.array([0.0, 0.10])
    obs = _gather(np.zeros_like(shifts))
    syn = _gather(shifts, amp=A * 1e3)          # 1000x amplitude difference
    lag, _ = trace_lags(syn, obs, DT)
    assert np.allclose(lag.numpy()[0], shifts, atol=DT + 1e-12)


def test_integrated_ricker_f90_is_lower():
    """The campaign integrates the Ricker, which lowers the spectrum: the
    honest skip threshold is ~80 ms, not 1/(2*f0) = 100 ms."""
    f90_plain = ricker_f90(F0, DT, NT, integrated=False)
    f90_int = ricker_f90(F0, DT, NT, integrated=True)
    assert f90_int < f90_plain
    assert f90_int == pytest.approx(6.25, abs=0.3)
    assert skip_threshold(f90_int) == pytest.approx(0.080, abs=0.005)
