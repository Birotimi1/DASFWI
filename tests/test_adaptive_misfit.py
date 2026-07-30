"""Adaptive L2->OT objective: blend balance, schedule, and short-circuiting."""
import math

import numpy as np
import pytest
import torch

from ADFWI.fwi.misfit import Misfit_waveform_L2
from inversion.adaptive_misfit import (BlendedMisfit, LambdaSchedule,
                                       DiagnosticLambda)
from inversion.safe_misfits import SinkhornSafe

DT, NT, F0 = 0.003, 400, 5.0
A = 2.3e-8                        # realistic DAS strain-rate amplitude


def _pair(shift=0.02, nr=12):
    """(obs, syn) at strain-rate scale; syn carries the autograd graph."""
    t = np.arange(NT) * DT
    def gather(sh):
        g = np.empty((1, NT, nr))
        for j in range(nr):
            tt = t - (0.4 + sh + 0.001 * j)
            g[0, :, j] = A * (1 - 2 * (np.pi * F0 * tt) ** 2) * \
                np.exp(-(np.pi * F0 * tt) ** 2)
        return torch.from_numpy(g)
    obs = gather(0.0)
    syn = gather(shift).clone().requires_grad_(True)
    return obs, syn


def _l2():
    return Misfit_waveform_L2(dt=DT)


def _ot():
    return SinkhornSafe(dt=0.01, sparse_sampling=2, p=1, blur=1e-2)


# --------------------------------------------------------------------------- #
# the central claim: normalisation balances the two terms' influence
# --------------------------------------------------------------------------- #
def test_normalisation_balances_adjoint_sources():
    """Each normalised term must contribute a ~unit-norm adjoint source, so
    lambda controls the real balance (value-normalisation would NOT do this:
    measured value ratio 233 vs gradient ratio 2686)."""
    norms = {}
    for name, lam in (("lo", 0.0), ("hi", 1.0)):
        obs, syn = _pair()
        m = BlendedMisfit(_l2(), _ot(), lam=lam, beta=0.0)
        e = m.forward(syn, obs)                 # FWI convention: (syn, obs)
        g, = torch.autograd.grad(e, syn)
        norms[name] = float(g.norm())
    assert norms["lo"] == pytest.approx(1.0, rel=1e-3)
    assert norms["hi"] == pytest.approx(1.0, rel=1e-3)


def test_raw_terms_are_wildly_unbalanced():
    """Guards the premise: without normalisation the gradients differ by orders
    of magnitude, which is why the blend must normalise."""
    obs, syn = _pair()
    e1 = _l2().forward(syn, obs)
    g1, = torch.autograd.grad(e1, syn, retain_graph=True)
    obs2, syn2 = _pair()
    e2 = _ot().forward(syn2, obs2)
    g2, = torch.autograd.grad(e2, syn2)
    ratio = float(g2.norm()) / float(g1.norm())
    assert ratio > 100          # orders of magnitude apart -> normalisation needed


# --------------------------------------------------------------------------- #
# reduction at the endpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam,pure", [(0.0, "lo"), (1.0, "hi")])
def test_endpoints_match_pure_misfit_direction(lam, pure):
    """At lambda=0/1 the blend must give the SAME gradient direction as the pure
    misfit (magnitude is normalised by construction)."""
    obs, syn = _pair()
    m = BlendedMisfit(_l2(), _ot(), lam=lam, beta=0.0)
    gb, = torch.autograd.grad(m.forward(syn, obs), syn)
    obs2, syn2 = _pair()
    pure_fn = _l2() if pure == "lo" else _ot()
    gp, = torch.autograd.grad(pure_fn.forward(syn2, obs2), syn2)
    cos = float(torch.dot(gb.flatten(), gp.flatten())
                / (gb.norm() * gp.norm()))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_unnormalised_endpoints_are_exact():
    """With normalize=False the endpoints reduce EXACTLY to the pure losses."""
    obs, syn = _pair()
    m = BlendedMisfit(_l2(), _ot(), lam=0.0, normalize=False)
    assert float(m.forward(syn, obs)) == pytest.approx(float(_l2().forward(syn, obs)))


def test_blend_is_finite_and_between_at_midpoint():
    """lambda=0.5 on strain-rate-scale data must stay finite (the convsi-style
    underflow trap) and produce a usable gradient."""
    obs, syn = _pair()
    m = BlendedMisfit(_l2(), _ot(), lam=0.5, beta=0.0)
    e = m.forward(syn, obs)
    g, = torch.autograd.grad(e, syn)
    assert torch.isfinite(e).all() and torch.isfinite(g).all()
    assert float(g.norm()) > 0
    # both scales were populated -> both terms really were evaluated
    assert m.scales["lo"] is not None and m.scales["hi"] is not None


def test_short_circuit_skips_the_expensive_term():
    """At lambda=0 the OT term must never be evaluated (it costs ~20x L2)."""
    calls = {"n": 0}

    class Counting(SinkhornSafe):
        def forward(self, a, b):
            calls["n"] += 1
            return super().forward(a, b)

    obs, syn = _pair()
    m = BlendedMisfit(_l2(), Counting(dt=0.01, sparse_sampling=2, p=1, blur=1e-2),
                      lam=0.0)
    m.forward(syn, obs)
    assert calls["n"] == 0
    m.set_lambda(0.5)
    m.forward(syn, obs)
    assert calls["n"] == 1


def test_lambda_validation():
    m = BlendedMisfit(_l2(), _ot())
    with pytest.raises(ValueError):
        m.set_lambda(1.5)
    with pytest.raises(ValueError):
        m.set_lambda(-0.1)


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #
def test_frequency_schedule_ramps_and_clips():
    s = LambdaSchedule(f_lo=3.0, f_hi=8.0)
    assert s.lam(2.0) == 0.0                       # below f_lo -> pure L2
    assert s.lam(3.0) == pytest.approx(0.0)
    assert s.lam(8.0) == pytest.approx(1.0)
    assert s.lam(12.0) == 1.0                      # above f_hi -> pure OT
    mid = s.lam(math.sqrt(3.0 * 8.0))              # log-midpoint
    assert mid == pytest.approx(0.5, abs=1e-6)
    fs = [3.0, 4.0, 5.0, 6.0, 8.0]
    lams = [s.lam(f) for f in fs]
    assert all(b >= a for a, b in zip(lams, lams[1:]))   # monotonic


def test_stage_override_starts_robust_then_anneals():
    """Vs release starts at lambda=1 regardless of band (sqrt(3) seed can skip
    in the sedimentary cover even at 3 Hz), then anneals to the schedule."""
    s = LambdaSchedule(f_lo=3.0, f_hi=8.0, stage_overrides={"vs": 1.0},
                       stage_anneal=2)
    assert s.lam(3.0, stage="vs", bands_since_stage_start=0) == pytest.approx(1.0)
    assert s.lam(3.0, stage="vs", bands_since_stage_start=2) == pytest.approx(
        s.lam(3.0))                                 # fully annealed
    assert s.lam(3.0, stage="vp") == pytest.approx(s.lam(3.0))   # no override


def test_schedule_validation():
    with pytest.raises(ValueError):
        LambdaSchedule(f_lo=8.0, f_hi=3.0)


def test_stage_plan_vp_lead_vs_follow():
    """The 2-D schedule: band 1 Vp-only, Vs released at band 2 with lambda
    forced high (sqrt(3) seed risk), then annealing back to the frequency ramp."""
    from inversion.adaptive_misfit import stage_plan
    bands = [2.0, 3.0, 4.5, None]
    s = LambdaSchedule(3.0, 8.0, stage_overrides={"vs": 1.0}, stage_anneal=1)
    plan = stage_plan(bands, s, f_max_source=6.25, vs_release_band=2)
    assert [p["band"] for p in plan] == [1, 2, 3, 4]
    assert [p["vs_live"] for p in plan] == [False, True, True, True]
    assert plan[0]["stage"] == "vp" and plan[1]["stage"] == "vs"
    assert plan[0]["lam"] == pytest.approx(0.0)      # band 1: low freq, pure L2
    assert plan[1]["lam"] == pytest.approx(1.0)      # Vs release: forced robust
    assert plan[2]["lam"] < 1.0                      # annealed back to schedule
    assert plan[3]["f_eff"] == pytest.approx(6.25)   # 'full' band uses source f90
    # filtering can never RAISE the frequency content
    assert all(p["f_eff"] <= 6.25 + 1e-9 for p in plan)


def test_stage_plan_acoustic_has_no_vs_stage():
    from inversion.adaptive_misfit import stage_plan
    plan = stage_plan([3.0, 6.0], LambdaSchedule(3.0, 8.0), 6.25,
                      vs_release_band=None)
    assert all(not p["vs_live"] and p["stage"] == "vp" for p in plan)
    assert plan[0]["lam"] == pytest.approx(0.0) and plan[1]["lam"] > 0


def test_skip_switch_fires_at_gate_calibration():
    """THE F2 BUG TEST: with default thresholds, a start at s16's measured skip
    (0.64) must select the ROBUST term. (The originally proposed on_above=0.65
    sat above 0.64, so the controller would have run pure L2 exactly at the rung
    where L2 collapses.)"""
    from inversion.adaptive_misfit import SkipSwitch, SKIP_ON_ABOVE, SKIP_OFF_BELOW
    assert SKIP_ON_ABOVE < 0.64          # must fire at s16
    assert SKIP_OFF_BELOW < 0.51         # must not hand back at s6's start level
    s = SkipSwitch()
    assert s.update(0.64) == pytest.approx(1.0)


def test_skip_switch_initializes_from_first_measurement():
    from inversion.adaptive_misfit import SkipSwitch
    # first obs inside the hysteresis band -> start ROBUST (conservative),
    # NOT the hardcoded-0 (pure L2) behavior the review flagged
    assert SkipSwitch().update(0.50) == pytest.approx(1.0)
    # first obs clearly aligned -> start on L2
    assert SkipSwitch().update(0.20) == pytest.approx(0.0)


def test_skip_switch_handover_and_ratchet():
    from inversion.adaptive_misfit import SkipSwitch
    s = SkipSwitch(ema=1.0, dwell=1, max_handbacks=1)  # no smoothing, for clarity
    assert s.update(0.70) == 1.0          # start robust
    assert s.update(0.50) == 1.0          # in band: held
    assert s.update(0.30) == 0.0          # below off_below: hand over to L2
    assert s.handbacks == 1
    lam = s.update(0.70)                  # skip rises again: ratchet holds L2
    assert lam == 0.0 and s.reentries == 1
    assert len(s.history) == 4


def test_skip_switch_dwell_and_smoothing_block_chatter():
    from inversion.adaptive_misfit import SkipSwitch
    # dwell: a mode younger than `dwell` updates cannot flip
    s = SkipSwitch(ema=1.0, dwell=2)
    s.update(0.70)                        # robust, age now 1
    assert s.update(0.10) == 1.0          # would hand over, but dwell blocks
    assert s.update(0.10) == 0.0          # age reached -> hand-over commits
    # EMA: one clean-looking dip must not flip the smoothed series
    s2 = SkipSwitch(ema=0.3, dwell=1)
    s2.update(0.70)
    assert s2.update(0.20) == 1.0         # smoothed 0.55 > off_below: held
    # NaN is ignored (state held)
    assert s2.update(float("nan")) == 1.0


def test_diagnostic_lambda_hysteresis():
    """Measured skipping raises lambda; good alignment lowers it; inside the
    hysteresis band the state is held (no oscillation)."""
    d = DiagnosticLambda(f_lo=3.0, f_hi=8.0, on_above=0.30, off_below=0.15,
                         blend=1.0)
    assert d.lam(4.0, skip_fraction=0.50) == pytest.approx(1.0)   # skipping
    assert d.lam(4.0, skip_fraction=0.22) == pytest.approx(1.0)   # held
    assert d.lam(4.0, skip_fraction=0.05) == pytest.approx(0.0)   # aligned
    assert d.lam(4.0, skip_fraction=0.22) == pytest.approx(0.0)   # held
    # no diagnostic -> falls back to the frequency prior
    assert d.lam(4.0) == pytest.approx(LambdaSchedule(3.0, 8.0).lam(4.0))
