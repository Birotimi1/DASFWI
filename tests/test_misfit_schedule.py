"""Tests for inversion/misfit_schedule.py (T6)."""

import math

import pytest
import torch

from ADFWI.fwi.misfit import (Misfit_global_correlation, Misfit_envelope,
                              Misfit_weighted_ECI)

from inversion.misfit_schedule import (ScheduledMisfit, weci_sigmoid_weights,
                                       linear_ramp_weights,
                                       piecewise_constant_weights)


# --------------------------------------------------------------------------- #
# weights sum to 1 for every i
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fn", [
    weci_sigmoid_weights(20),
    linear_ramp_weights(20),
    piecewise_constant_weights([10, 25], [(1, 0), (0.5, 0.5), (0, 1)]),
])
def test_weights_sum_to_one(fn):
    for i in range(0, 60):
        w = fn(i)
        assert abs(sum(w) - 1.0) < 1e-12, (i, w)
        assert all(x >= 0 for x in w), (i, w)


def test_weci_sigmoid_form():
    # code-verified form: w_GC(i) = 1/(1 + exp(-(i - N/2)))
    N = 30
    fn = weci_sigmoid_weights(N)
    for i in (0, 7, 15, 23, 30, 45):
        assert fn(i)[0] == 1.0 / (1.0 + math.exp(-(i - N / 2)))
    assert fn(0)[0] < 1e-6           # envelope-dominated at the start
    assert fn(N)[0] > 1.0 - 1e-6     # GC-dominated at the end
    assert fn(N // 2)[0] == 0.5      # crossover at N/2


def test_linear_ramp_clamps():
    fn = linear_ramp_weights(11)
    assert fn(0) == (0.0, 1.0)
    assert fn(10) == (1.0, 0.0)
    assert fn(25) == (1.0, 0.0)      # clamped beyond the horizon
    assert linear_ramp_weights(1)(0) == (1.0, 0.0)


def test_piecewise_constant_bands():
    fn = piecewise_constant_weights([3, 6], [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    assert fn(0) == (1, 0, 0) and fn(2) == (1, 0, 0)
    assert fn(3) == (0, 1, 0) and fn(5) == (0, 1, 0)
    assert fn(6) == (0, 0, 1) and fn(100) == (0, 0, 1)


def test_piecewise_validation():
    with pytest.raises(ValueError):
        piecewise_constant_weights([5, 3], [(1, 0), (0, 1), (0.5, 0.5)])
    with pytest.raises(ValueError):
        piecewise_constant_weights([5], [(1, 0)])           # missing a row
    with pytest.raises(ValueError):
        piecewise_constant_weights([5], [(0.6, 0.6), (1, 0)])  # sum != 1


# --------------------------------------------------------------------------- #
# equivalence with ADFWI's Weci over an iteration sweep
# --------------------------------------------------------------------------- #

def test_reproduces_weci_loss():
    """GC+Envelope schedule with the WECI sigmoid must reproduce
    Misfit_weighted_ECI(max_iter=N) exactly, iteration by iteration."""
    N = 12
    dt, p = 1, 1.5

    weci = Misfit_weighted_ECI(max_iter=N, dt=dt, p=p, instaneous_phase=False)
    sched = ScheduledMisfit(
        components=[Misfit_global_correlation(dt=dt),
                    Misfit_envelope(dt=dt, p=p, instaneous_phase=False)],
        weight_fn=weci_sigmoid_weights(N))

    g = torch.Generator().manual_seed(3)
    for i in range(N):
        # float32: upstream GC hard-codes a float32 accumulator
        a = torch.randn(2, 100, 7, dtype=torch.float32, generator=g)
        b = torch.randn(2, 100, 7, dtype=torch.float32, generator=g)
        # Weci.forward(obs, syn) passes its args to GC/Envelope in the same
        # positional order our scheduler does with forward(syn, obs); call
        # both with identical positional arguments.
        l_ref = weci.forward(a, b)
        l_sch = sched.forward(a, b)
        assert torch.allclose(l_sch, l_ref, rtol=0.0, atol=0.0), (
            f"iteration {i}: schedule {float(l_sch)!r} != weci {float(l_ref)!r}")


# --------------------------------------------------------------------------- #
# counter ownership and reset policy
# --------------------------------------------------------------------------- #

def test_counter_and_reset():
    sched = ScheduledMisfit(
        components=[Misfit_global_correlation(dt=1),
                    Misfit_envelope(dt=1)],
        weight_fn=weci_sigmoid_weights(10))
    assert sched.iter == 0

    g = torch.Generator().manual_seed(0)
    a = torch.randn(1, 50, 3, dtype=torch.float32, generator=g)
    b = torch.randn(1, 50, 3, dtype=torch.float32, generator=g)
    for _ in range(4):
        sched.forward(a, b)
    assert sched.iter == 4          # exactly one increment per forward call

    sched.reset()
    assert sched.iter == 0          # explicit reset only
    sched.reset(7)
    assert sched.iter == 7          # band-local restart at a given iteration

    sched2 = ScheduledMisfit(
        components=[Misfit_global_correlation(dt=1), Misfit_envelope(dt=1)],
        weight_fn=weci_sigmoid_weights(10), start_iter=5)
    assert sched2.iter == 5


def test_bad_weight_fn_rejected():
    with pytest.raises(ValueError):
        ScheduledMisfit(components=[Misfit_global_correlation(dt=1)],
                        weight_fn=lambda i: (0.7, 0.7))    # wrong K AND sum
    with pytest.raises(ValueError):
        ScheduledMisfit(components=[Misfit_global_correlation(dt=1),
                                    Misfit_envelope(dt=1)],
                        weight_fn=lambda i: (0.7, 0.7))    # sum != 1
    with pytest.raises(ValueError):
        ScheduledMisfit(components=[], weight_fn=lambda i: ())
