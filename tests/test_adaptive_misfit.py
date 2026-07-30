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


def test_skip_timing_yields_no_band_lambda():
    """REGRESSION: under skip-driven timing stage_plan gets schedule=None and so
    returns lam=None for every band (the controller sets lambda per chunk). A
    band-level set_lambda(lam) then crashed run_pipeline on the elastic switch
    arms. The plan must report None, and set_lambda must say so clearly."""
    from inversion.adaptive_misfit import stage_plan
    plan = stage_plan([None, None], None, 3.73, vs_release_band=2)
    assert all(p["lam"] is None for p in plan)          # nothing to apply per band
    with pytest.raises(ValueError, match="set_lambda\\(None\\)"):
        BlendedMisfit(_l2(), _ot()).set_lambda(None)


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


def test_single_band_with_release_band_2_never_inverts_vs():
    """REGRESSION: `--bands full` gives ONE band while --vs-release-band defaults
    to 2, so vs_live (bi >= 2) never fires and an 'elastic' run silently updates
    Vp ONLY -- at full elastic cost. Two Bridges-2 cells did exactly this.
    run_pipeline's preflight now refuses this configuration."""
    from inversion.adaptive_misfit import stage_plan
    bad = stage_plan([None], None, 3.73, vs_release_band=2)
    assert not any(p["vs_live"] for p in bad)          # the bug
    # the fix: two full-band stages = Vp-lead/Vs-follow with no cascade
    good = stage_plan([None, None], None, 3.73, vs_release_band=2)
    assert [p["vs_live"] for p in good] == [False, True]
    # ...as does releasing Vs in the single band
    assert stage_plan([None], None, 3.73, vs_release_band=1)[0]["vs_live"]


def test_stage_plan_acoustic_has_no_vs_stage():
    from inversion.adaptive_misfit import stage_plan
    plan = stage_plan([3.0, 6.0], LambdaSchedule(3.0, 8.0), 6.25,
                      vs_release_band=None)
    assert all(not p["vs_live"] and p["stage"] == "vp" for p in plan)
    assert plan[0]["lam"] == pytest.approx(0.0) and plan[1]["lam"] > 0


def test_blend_rejects_stateful_weci():
    """weci carries its OWN iteration schedule (envelope -> global correlation,
    advanced by call count), so blending/switching it is silently wrong: the
    short-circuit freezes its counter, and past max_iter/2 the 'robust' term has
    become phase-SENSITIVE GC. BlendedMisfit must refuse it."""
    from inversion.config import build_misfit
    weci = build_misfit("weci", dt=0.003, iterations=300)
    assert hasattr(weci, "iter") and hasattr(weci, "max_iter")   # the signature
    l2 = build_misfit("l2", dt=0.003, iterations=300)
    with pytest.raises(ValueError, match="own iteration schedule"):
        BlendedMisfit(l2, weci)
    with pytest.raises(ValueError, match="own iteration schedule"):
        BlendedMisfit(weci, l2)
    # explicit opt-out still allowed (for an externally pinned weight)
    BlendedMisfit(l2, weci, allow_stateful=True)
    # the stateless envelope -- the actual robust term -- is accepted
    BlendedMisfit(l2, build_misfit("envelope", dt=0.003, iterations=300))


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


def test_stage_ladder_advances_and_never_regresses():
    from inversion.adaptive_misfit import StageLadder
    L = StageLadder([0.45, 0.20], ema=1.0, dwell=1)     # no smoothing, for clarity
    assert L.update(0.64) == 0          # robust stage
    assert L.update(0.50) == 0          # above first threshold: held
    assert L.update(0.40) == 1          # -> gentle refiner
    assert L.update(0.30) == 1          # above second threshold: held
    assert L.update(0.15) == 2          # -> sharp refiner
    # monotonic: a skip rise can NEVER walk the ladder back
    assert L.update(0.90) == 2
    assert len(L.history) == 6


def test_stage_ladder_skips_stages_on_an_easy_start():
    """An already-aligned start must jump straight to the sharp refiner rather
    than waste iterations in robust mode ('stay out of the way' at s6)."""
    from inversion.adaptive_misfit import StageLadder
    assert StageLadder([0.45, 0.20], ema=1.0).update(0.05) == 2
    # NaN is ignored (stage held)
    L = StageLadder([0.45, 0.20], ema=1.0)
    L.update(0.60)
    assert L.update(float("nan")) == 0


def test_stage_ladder_rejects_bad_thresholds():
    from inversion.adaptive_misfit import StageLadder
    with pytest.raises(ValueError):
        StageLadder([0.20, 0.45])                # ascending
    with pytest.raises(ValueError):
        StageLadder([])                          # empty


def test_staged_misfit_evaluates_only_the_active_stage():
    from inversion.adaptive_misfit import StagedMisfit
    calls = {"a": 0, "b": 0, "c": 0}

    def mk(key, scale):
        class _M(torch.nn.Module):
            def forward(self, syn, obs):
                calls[key] += 1
                return scale * ((syn - obs) ** 2).sum()
        return _M()

    m = StagedMisfit([mk("a", 1.0), mk("b", 2.0), mk("c", 3.0)],
                     names=("envelope", "gc", "l2"))
    syn = torch.randn(2, 8, 3, requires_grad=True)
    obs = torch.randn(2, 8, 3)
    assert m.active_name == "envelope"
    m.forward(syn, obs)
    assert (calls["a"], calls["b"], calls["c"]) == (1, 0, 0)   # only stage 0
    m.set_stage(2)
    assert m.active_name == "l2"
    m.forward(syn, obs)
    assert (calls["a"], calls["b"], calls["c"]) == (1, 0, 1)   # only stage 2
    with pytest.raises(ValueError):
        m.set_stage(3)
    with pytest.raises(ValueError):
        StagedMisfit([mk("a", 1.0)])                           # needs >= 2


def test_staged_misfit_normalises_each_stage_to_unit_gradient():
    """The property that keeps Adam/adagrad moments sane across a hand-over:
    every stage's adjoint source has ~unit norm despite wildly different raw
    scales."""
    from inversion.adaptive_misfit import StagedMisfit

    def mk(scale):
        class _M(torch.nn.Module):
            def forward(self, syn, obs):
                return scale * ((syn - obs) ** 2).sum()
        return _M()

    m = StagedMisfit([mk(1.0), mk(1e6)], beta=0.0)     # 1e6x raw scale gap
    obs = torch.randn(2, 8, 3)
    for stage in (0, 1):
        m.set_stage(stage)
        syn = torch.randn(2, 8, 3, requires_grad=True)
        e = m.forward(syn, obs)
        g, = torch.autograd.grad(e, syn)
        assert g.norm().item() == pytest.approx(1.0, rel=0.05)


def test_staged_misfit_rejects_stateful_weci():
    from inversion.adaptive_misfit import StagedMisfit
    from inversion.config import build_misfit
    with pytest.raises(ValueError, match="own iteration schedule"):
        StagedMisfit([build_misfit("l2", dt=0.003, iterations=300),
                      build_misfit("weci", dt=0.003, iterations=300)])


def test_multiscale_needs_a_fresh_switch_per_band():
    """PHASE B semantics. Raising the band raises f_max, so skip jumps at every
    band boundary and the controller must be free to re-enter robust mode. A
    PERSISTENT switch's hand-back ratchet blocks that (it would stay on L2 for
    the rest of the cascade); a FRESH switch per band does the right thing."""
    from inversion.adaptive_misfit import SkipSwitch
    # band 1 (low f): skip reads low -> L2 quickly. band 2 (higher f): skip jumps.
    band1, band2 = [0.60, 0.30], [0.70, 0.30]

    persistent = SkipSwitch(ema=1.0, dwell=1)
    lams = [persistent.update(s) for s in band1 + band2]
    assert lams[-2] == 0.0 and persistent.reentries == 1   # ratchet blocked it

    fresh = []
    for band in (band1, band2):
        sw = SkipSwitch(ema=1.0, dwell=1)                  # fresh per band
        fresh.append([sw.update(s) for s in band])
    assert fresh[0] == [1.0, 0.0]        # band 1: robust -> hand over
    assert fresh[1] == [1.0, 0.0]        # band 2: re-enters robust, hands over


def test_low_band_hands_over_immediately():
    """Why the Phase-A thresholds transfer to the cascade unchanged: at a low
    band T/2 is larger, so the SAME model reads as less skipped and the
    controller goes straight to the sharp refiner -- which is exactly why
    multiscale works."""
    from inversion.adaptive_misfit import SkipSwitch
    from inversion.skip_diagnostic import skip_threshold
    assert skip_threshold(3.0) > skip_threshold(6.25)      # looser at low f
    # a model reading 0.30 skip at 3 Hz starts on L2, not envelope
    assert SkipSwitch().update(0.30) == pytest.approx(0.0)


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
